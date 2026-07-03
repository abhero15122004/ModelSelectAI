import time, os, joblib, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

def train_arima(series, order=(1,1,1), run_dir=".", name="arima"):
    s = np.asarray(series, dtype="float64")
    t0 = time.time()
    model = ARIMA(s, order=order).fit()
    train_time = time.time()-t0
    path = os.path.join(run_dir, "artifacts", f"{name}.pkl")
    joblib.dump(model, path)
    size_mb = os.path.getsize(path)/(1024*1024)
    # quick hold-out forecast for metrics
    k = max(10, int(0.2*len(s)))
    preds = model.forecast(steps=k)
    trues = s[-k:]
    mae = float(mean_absolute_error(trues, preds))
    rmse = float(mean_squared_error(trues, preds))
    return dict(name=name, train_time_s=train_time, infer_time_s_per_row=0.0, mae=mae, rmse=rmse, size_mb=size_mb, model_path=path)

def train_ets(series, seasonal=None, run_dir=".", name="ets"):
    s = np.asarray(series, dtype="float64")
    t0 = time.time()
    model = ExponentialSmoothing(s, seasonal=seasonal).fit()
    train_time = time.time()-t0
    path = os.path.join(run_dir, "artifacts", f"{name}.pkl")
    joblib.dump(model, path)
    size_mb = os.path.getsize(path)/(1024*1024)
    k = max(10, int(0.2*len(s)))
    preds = model.forecast(k)
    trues = s[-k:]
    mae = float(mean_absolute_error(trues, preds))
    rmse = float(mean_squared_error(trues, preds))
    return dict(name=name, train_time_s=train_time, infer_time_s_per_row=0.0, mae=mae, rmse=rmse, size_mb=size_mb, model_path=path)

def ml_on_lags(series, lags=24, run_dir=".", name="rf_lags"):
    from .preprocess_timeseries import make_lag_features
    import numpy as np
    s = np.asarray(series, dtype="float64")
    X, y = make_lag_features(pd_series_like(s), lags=lags)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=300)
    t0 = time.time(); model.fit(X_tr, y_tr); train_time = time.time()-t0
    t1 = time.time(); preds = model.predict(X_te); infer_time = (time.time()-t1)/max(len(y_te),1)
    mae = mean_absolute_error(y_te, preds); rmse = mean_squared_error(y_te, preds)
    path = os.path.join(run_dir, "artifacts", f"{name}.joblib")
    joblib.dump(model, path)
    size_mb = os.path.getsize(path)/(1024*1024)
    return dict(name=name, train_time_s=train_time, infer_time_s_per_row=infer_time, mae=float(mae), rmse=float(rmse), size_mb=size_mb, model_path=path)

# helper to adapt numpy array to series-like
def pd_series_like(arr):
    import pandas as pd, numpy as np
    if isinstance(arr, pd.Series): return arr
    return pd.Series(np.asarray(arr).astype("float64"))

# LSTM
if HAS_TORCH:
    class SeqDataset(Dataset):
        def __init__(self, arr, seq_len=24):
            self.arr = np.asarray(arr, dtype="float32")
            self.seq_len = seq_len
        def __len__(self):
            return max(0, len(self.arr) - self.seq_len)
        def __getitem__(self, idx):
            import torch
            x = self.arr[idx: idx + self.seq_len]
            y = self.arr[idx + self.seq_len]
            return torch.tensor(x).unsqueeze(-1), torch.tensor(y)

    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out).squeeze(-1)

    def train_lstm(series, seq_len=24, epochs=10, batch_size=32, lr=1e-3, run_dir=".", name="lstm_ts"):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        arr = np.asarray(series, dtype=np.float32)
        n = len(arr)
        if n < seq_len + 50:
            raise ValueError("Series too short for LSTM")
        split = int(n*0.8)
        ds = SeqDataset(arr, seq_len)
        train_ds = torch.utils.data.Subset(ds, list(range(split - seq_len)))
        test_ds = torch.utils.data.Subset(ds, list(range(split - seq_len, len(ds))))
        tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        te = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        model = LSTMModel().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        t0 = time.time()
        for _ in range(epochs):
            model.train()
            for xb, yb in tr:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        train_time = time.time()-t0
        model.eval()
        preds, trues = [], []
        t1 = time.time()
        with torch.no_grad():
            for xb, yb in te:
                xb = xb.to(device)
                out = model(xb).cpu().numpy()
                preds.extend(out.tolist()); trues.extend(yb.numpy().tolist())
        infer_time = (time.time()-t1)/max(len(trues),1)
        mae = float(mean_absolute_error(trues, preds))
        rmse = float(mean_squared_error(trues, preds))
        path = os.path.join(run_dir, "artifacts", f"{name}.pt")
        torch.save(model.state_dict(), path)
        size_mb = os.path.getsize(path)/(1024*1024)
        return dict(name=name, train_time_s=train_time, infer_time_s_per_row=infer_time, mae=mae, rmse=rmse, size_mb=size_mb, model_path=path)