import warnings
warnings.filterwarnings("ignore")

import os, time
from typing import Optional

from modelselectai.utils import new_run_dir, save_json, print_block, log_exception
from modelselectai.loader import load_table, try_extract_archive, is_image_folder, is_dicom_folder
from modelselectai.detect import detect_input_type
from modelselectai.config import RunConfig
from modelselectai.preprocess_tabular import build_preprocessor, split_data
from modelselectai.preprocess_timeseries import detect_datetime_column, select_ts_target
from modelselectai.preprocess_image import load_images_from_folder
from modelselectai.preprocess_dicom import load_dicom_folder
from modelselectai.models_tabular import classification_models, regression_models
from modelselectai.models_timeseries import train_arima, train_ets, ml_on_lags, train_lstm
from modelselectai.models_vision import finetune_resnet_from_folder, sklearn_flat_images
from modelselectai.evaluation import plot_top5_classification, plot_top5_regression
from modelselectai.scorer import rank_models
from modelselectai.utils_task import detect_task_type
from modelselectai.graphing import generate_all_graphs

import numpy as np, pandas as pd, joblib, time as time_module
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")

# ---- Change this to your dataset path (file or folder) ----
INPUT_PATH = None

# -------------------- TABULAR/TIMESERIES RUNNER --------------------

def run_all_tabular_models(df, target, task, cfg, run_dir):
    X = df.drop(columns=[target])
    y = df[target]

    _, _, pre, num_cols, cat_cols = build_preprocessor(df, target)

    # ---------------- SAFE STRATIFY LOGIC ----------------
    stratify_flag = None
    try:
        is_categorical_or_low_cardinality = (
            (not pd.api.types.is_numeric_dtype(y))
            or (y.nunique() <= cfg.cls_cardinality_threshold)
        )
        if is_categorical_or_low_cardinality:
            vc = y.value_counts()
            if (vc >= 2).all() and len(vc) > 1:
                stratify_flag = y
    except Exception:
        stratify_flag = None

    # ✅ Safe split with fallback (prevents ValueError)
    try:
        X_tr, X_te, y_tr, y_te = split_data(
            X, y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify_flag
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = split_data(
            X, y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=None
        )
    # ---------------------------------------------------

    results, errors = [], []
    models = classification_models() if task == "classification" else regression_models()
    
    def process_model(name, mdl):
        try:
            pipe = Pipeline([("pre", pre), ("model", mdl)])
            t0 = time_module.time(); pipe.fit(X_tr, y_tr); train_time = time_module.time()-t0
            t1 = time_module.time(); pred = pipe.predict(X_te); infer_time = (time_module.time()-t1)/max(len(y_te),1)
            
            if task == "classification":
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
                acc = float(accuracy_score(y_te, pred))
                f1w = float(f1_score(y_te, pred, average="weighted", zero_division=0))
                prec = float(precision_score(y_te, pred, average="weighted", zero_division=0))
                rec = float(recall_score(y_te, pred, average="weighted", zero_division=0))
                auc = None
                try:
                    if hasattr(pipe, "predict_proba"):
                        prob = pipe.predict_proba(X_te)
                        if prob.ndim == 2 and prob.shape[1] == 2:
                            auc = float(roc_auc_score(y_te, prob[:,1]))
                except Exception:
                    auc = None
                cm = confusion_matrix(y_te, pred).tolist()
                path_m = os.path.join(run_dir, "artifacts", f"{name}.joblib"); joblib.dump(pipe, path_m)
                size_mb = os.path.getsize(path_m)/(1024*1024)
                return dict(name=name, task=task, accuracy=acc, f1_weighted=f1w,
                            precision=prec, recall=rec, roc_auc=auc, confusion_matrix=cm,
                            train_time_s=train_time, infer_time_s_per_row=infer_time,
                            size_mb=size_mb, model_path=path_m,
                            explainability=1.0 if hasattr(pipe.named_steps["model"], "feature_importances_") else 0.6)
            else:
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                r2 = float(r2_score(y_te, pred))
                mae = float(mean_absolute_error(y_te, pred))
                rmse = float(mean_squared_error(y_te, pred))
                path_m = os.path.join(run_dir, "artifacts", f"{name}.joblib"); joblib.dump(pipe, path_m)
                size_mb = os.path.getsize(path_m)/(1024*1024)
                return dict(name=name, task=task, r2=r2, mae=mae, rmse=rmse,
                            train_time_s=train_time, infer_time_s_per_row=infer_time,
                            size_mb=size_mb, model_path=path_m,
                            explainability=1.0 if hasattr(pipe.named_steps["model"], "feature_importances_") else 0.6)
        except Exception as e:
            return {"error": f"{name}: {e}"}

    results_and_errors = Parallel(n_jobs=-1)(delayed(process_model)(name, mdl) for name, mdl in models)
    results = [res for res in results_and_errors if "error" not in res]
    errors = [res["error"] for res in results_and_errors if "error" in res]
    return results, errors


def run_tabular_and_timeseries(path, cfg: RunConfig, application, run_dir, task_hint):
    df = pd.read_csv(path, nrows=5000) if os.path.getsize(path) > 10e6 else load_table(path)
    df = df.loc[:, df.notna().sum() > 0]

    if df.shape[1] < 2:
        raise ValueError("Dataset has too few columns to process.")

    dt_col = detect_datetime_column(df)

    target_found = False
    for col in reversed(df.columns):
        if df[col].notna().sum() > 0:
            target = col
            target_found = True
            break
    if not target_found:
        raise ValueError("No suitable target column found in the dataset.")

    df = df.dropna(subset=[target])
    if df.empty or df.shape[1] < 2:
        raise ValueError("Dataset is empty or has too few columns after removing rows with missing target values.")

    task = detect_task_type(df[target], cfg)
    print(f"[INFO] Detected ML Task: {task.upper()} (Target: {target})")

    combined_results, combined_errors = [], []

    if task == "regression" and dt_col:
        ts_y = select_ts_target(df, exclude_cols=[dt_col] if dt_col else [])
        if not ts_y.empty:
            ts_results, ts_errors = [], []
            for trainer, name in [(train_arima, "arima"), (train_ets, "ets"),
                                  (ml_on_lags, "rf_lags"), (train_lstm, "lstm_ts")]:
                try:
                    res = trainer(ts_y, run_dir=run_dir, name=name)
                    ts_results.append(res)
                except Exception as e:
                    ts_errors.append(f"{name}: {e}")
            combined_results.extend(ts_results)
            combined_errors.extend(ts_errors)

    tabular_results, tabular_errors = run_all_tabular_models(df, target, task, cfg, run_dir)
    combined_results.extend(tabular_results)
    combined_errors.extend(tabular_errors)

    ranked = rank_models([r for r in combined_results if "error" not in r], task=task, cfg=cfg, application=application)
    
    if task == "regression":
        final_results = []
        top_tabular_score = max([r.get("suitability_score", 0) for r in tabular_results] or [0])
        
        for r in ranked:
            if "arima" in r["name"] or "ets" in r["name"] or "lstm_ts" in r["name"]:
                if r["suitability_score"] > top_tabular_score:
                    final_results.append(r)
            else:
                final_results.append(r)
    else:
        final_results = ranked

    final_top5 = final_results[:cfg.n_top_models]
    
    if task == "classification":
        figs = plot_top5_classification(final_top5, run_dir)
    else:
        figs = plot_top5_regression(final_top5, run_dir)
    
    return dict(task=task, ranked=ranked, top5=final_top5, figs=figs, errors=combined_errors)


# -------------------- IMAGE & MEDICAL RUNNERS --------------------
# (unchanged)
# ----------------------------------------------------------------

def run_image(folder, cfg: RunConfig, application, run_dir):
    try:
        X, y = load_images_from_folder(folder, size=(224, 224))
    except Exception:
        X, y = load_images_from_folder(folder, size=(128, 128))
    if len(X) == 0:
        raise ValueError("No images found in folder")

    results, errors = [], []
    try:
        res = finetune_resnet_from_folder(folder, epochs=6, run_dir=run_dir, name="resnet18_ft")
        results.append(res)   # ✅ always append ResNet results, like in old code
    except Exception as e:
        errors.append(f"resnet18_ft: {e}")

    try:
        res = sklearn_flat_images(X, y, run_dir=run_dir, name="rf_flat_images")
        results.append(res)
    except Exception as e2:
        errors.append(f"rf_flat_images: {e2}")

    ranked = results
    top5 = ranked[:cfg.n_top_models]
    figs = plot_top5_classification(top5, run_dir) if top5 and "accuracy" in top5[0] else []
    return dict(task="image", ranked=ranked, top5=top5, figs=figs, errors=errors)


def run_medical(folder, cfg: RunConfig, application, run_dir):
    X, y = load_dicom_folder(folder, size=(224, 224))
    if len(X) == 0:
        raise ValueError("No DICOM images loaded")

    results, errors = [], []
    try:
        res = finetune_resnet_from_folder(folder, epochs=6, run_dir=run_dir, name="resnet18_dicom")
        results.append(res)   # ✅ always append ResNet results
    except Exception as e:
        errors.append(f"resnet18_dicom: {e}")

    try:
        res = sklearn_flat_images(X, y, run_dir=run_dir, name="rf_flat_dicoms")
        results.append(res)
    except Exception as e2:
        errors.append(f"rf_flat_dicoms: {e2}")

    ranked = results
    top5 = ranked[:cfg.n_top_models]
    figs = plot_top5_classification(top5, run_dir) if top5 and "accuracy" in top5[0] else []
    return dict(task="medical", ranked=ranked, top5=top5, figs=figs, errors=errors)

# -------------------- ORCHESTRATOR --------------------
# (unchanged from your code)
# ----------------------------------------------------------------

def run_pipeline(input_path: Optional[str] = None, application: Optional[str] = None, task_hint: str = "auto"):
    cfg = RunConfig()
    run_id, run_dir = new_run_dir()
    if not input_path or not str(input_path).strip():
        input_path = INPUT_PATH
    if not input_path:
        raise ValueError("Set INPUT_PATH in modelselectai/main.py or pass input_path to run_pipeline().")

    if not application:
        application = "generic"

    input_folder = input_path if os.path.isdir(input_path) else None
    detected = detect_input_type(input_path if os.path.isfile(input_path) else None, input_folder, task_hint)

    if detected in ["tabular", "timeseries"]:
        out = run_tabular_and_timeseries(input_path, cfg, application, run_dir, detected)
    elif detected == "image":
        out = run_image(input_path if os.path.isdir(input_path) else None, cfg, application, run_dir)
    elif detected == "medical":
        out = run_medical(input_path if os.path.isdir(input_path) else None, cfg, application, run_dir)
    else:
        out = run_tabular_and_timeseries(input_path, cfg, application, run_dir, "tabular")
    
    graphs = generate_all_graphs(out, run_dir)

    figs = [os.path.basename(p) for p in out.get("figs", [])]
    graphs = [os.path.basename(p) for p in graphs]

    final_out = {
        "status": "done",
        "task": out.get("task"),
        "top5": out.get("top5", []),
        "ranked": out.get("ranked", []),
        "errors": out.get("errors", []),
        "figs": figs,
        "graphs": graphs,
    }

    lines = []
    lines.append(f"ModelSelectAI Summary | Run: {run_id} | Task: {final_out['task']} | App: {application}")
    lines.append(f"Artifacts: {run_dir}")
    lines.append("-" * 72)
    if final_out["top5"]:
        lines.append("Top 5 Models:")
        for i, r in enumerate(final_out["top5"], 1):
            if final_out["task"] == "classification":
                lines.append(f"{i}. {r['name']} | acc={r.get('accuracy',0):.4f} f1w={r.get('f1_weighted',0):.4f} auc={r.get('roc_auc') if r.get('roc_auc') is not None else 'NA'} "
                            f"| train={r.get('train_time_s',0):.3f}s infer/row={r.get('infer_time_s_per_row',0):.6f}s size={r.get('size_mb',0):.2f}MB "
                            f"| score={r.get('suitability_score',0):.4f}")
            elif final_out["task"] == "regression":
                lines.append(f"{i}. {r['name']} | R2={r.get('r2',0):.4f} MAE={r.get('mae',0):.4f} RMSE={r.get('rmse',0):.4f} "
                            f"| train={r.get('train_time_s',0):.3f}s infer/row={r.get('infer_time_s_per_row',0):.6f}s size={r.get('size_mb',0):.2f}MB "
                            f"| score={r.get('suitability_score',0):.4f}")
            else:
                if "mae" in r:
                    lines.append(f"{i}. {r['name']} | MAE={r.get('mae',0):.4f} RMSE={r.get('rmse',0):.4f} train={r.get('train_time_s',0):.3f}s size={r.get('size_mb',0):.2f}MB")
                else:
                    lines.append(f"{i}. {r['name']} | acc={r.get('accuracy',0):.4f} f1w={r.get('f1_weighted',0):.4f} train={r.get('train_time_s',0):.3f}s size={r.get('size_mb',0):.2f}MB")
    else:
        lines.append("No successful models to rank.")

    if final_out.get("errors"):
        lines.append("-" * 72)
        lines.append("Errors (skipped models):")
        for e in final_out["errors"]:
            lines.append(" - " + str(e))

    if final_out.get("figs"):
        lines.append("-" * 72)
        lines.append("Saved figures:")
        for p in final_out["figs"]:
            lines.append(" - " + p)

    if final_out.get("graphs"):
        lines.append("-" * 72)
        lines.append("Generated Graphs:")
        for gp in final_out["graphs"]:
            lines.append(" - " + gp)

    print("\n".join(lines), flush=True)
    
    return dict(run_id=run_id, run_dir=run_dir, **final_out)

if __name__ == "__main__":
    result = run_pipeline()
    import json
    print(json.dumps(result, indent=2))