# Vision pipeline: PyTorch finetune ResNet18 + RandomForest baseline
import os, time, joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms, datasets
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def finetune_resnet_from_folder(train_folder, val_folder=None, epochs=6, batch_size=16, lr=1e-4, run_dir=".", name="resnet18"):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    full_ds = datasets.ImageFolder(train_folder, transform=tr_transforms)
    n = len(full_ds)

    if n < 10:  # 🔥 tiny dataset safeguard
        train_ds = full_ds
        val_ds = datasets.ImageFolder(train_folder, transform=val_transforms)
    else:
        split = int(n * 0.8)
        if split == 0 or split == n:  # 🔥 avoid empty train/val
            split = max(1, n-1)
        train_ds = torch.utils.data.Subset(full_ds, list(range(split)))
        val_ds = torch.utils.data.Subset(datasets.ImageFolder(train_folder, transform=val_transforms), list(range(split, n)))

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = models.resnet18(pretrained=True)
    nfeat = model.fc.in_features
    n_classes = len(full_ds.classes)
    model.fc = nn.Linear(nfeat, n_classes)
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # ---- Training ----
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0

    # ---- Validation ----
    preds, trues = [], []
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb).argmax(dim=1).cpu().numpy()
            preds.extend(out.tolist()); trues.extend(yb.numpy().tolist())
    infer_time = (time.time() - t1) / max(len(trues), 1)

    acc, f1w = 0.0, 0.0
    if trues:
        acc = float(accuracy_score(trues, preds))
        f1w = float(f1_score(trues, preds, average="weighted"))

    path = os.path.join(run_dir, "artifacts", f"{name}.pt")
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path)/(1024*1024)

    return dict(
        name=name,
        train_time_s=train_time,
        infer_time_s_per_row=infer_time,
        accuracy=acc,
        f1_weighted=f1w,
        size_mb=size_mb,
        model_path=path
    )

def sklearn_flat_images(X_images, y_labels, run_dir=".", name="rf_flat"):
    """Baseline: train a RandomForest on flattened pixel arrays."""
    ns = len(X_images)
    Xf = X_images.reshape(ns, -1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        Xf, y_labels, test_size=0.2, random_state=42,
        stratify=y_labels if len(set(y_labels)) > 1 else None
    )

    clf = RandomForestClassifier(n_estimators=300)

    t0 = time.time()
    clf.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t1 = time.time()
    pred = clf.predict(X_te)
    infer_time = (time.time() - t1) / max(len(y_te), 1)

    acc = float(accuracy_score(y_te, pred))
    f1w = float(f1_score(y_te, pred, average="weighted"))

    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)
    path = os.path.join(run_dir, "artifacts", f"{name}.joblib")
    joblib.dump(clf, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)

    return dict(
        name=name,
        train_time_s=train_time,
        infer_time_s_per_row=infer_time,
        accuracy=acc,
        f1_weighted=f1w,
        size_mb=size_mb,
        model_path=path,
    )


def train_vision_models(train_folder, X_images=None, y_labels=None,
                        run_dir=".", epochs=6, batch_size=16, lr=1e-4):
    """
    Wrapper: always try both ResNet18 (if torch available) and RandomForest baseline.
    Returns a list of trained model result dicts.
    """
    results = []

    # Train ResNet18 if possible
    if HAS_TORCH:
        try:
            resnet_out = finetune_resnet_from_folder(
                train_folder, epochs=epochs, batch_size=batch_size,
                lr=lr, run_dir=run_dir, name="resnet18"
            )
            results.append(resnet_out)
        except Exception as e:
            results.append({"name": "resnet18", "error": str(e)})

    # Train RandomForest if raw arrays provided
    if X_images is not None and y_labels is not None:
        try:
            rf_out = sklearn_flat_images(
                X_images, y_labels, run_dir=run_dir, name="rf_flat"
            )
            results.append(rf_out)
        except Exception as e:
            results.append({"name": "rf_flat", "error": str(e)})

    return results