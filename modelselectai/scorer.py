import numpy as np
from .config import RunConfig

# Extended priors; any unknown domain falls back to "generic".
APP_PRIORS = {
    "generic":   dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.05, w_explain=0.10),

    # Regulated / high-stakes prefer explainability & performance
    "healthcare": dict(w_perf=0.60, w_train_time=0.10, w_infer_time=0.10, w_size=0.05, w_explain=0.15),
    "finance":    dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.20, w_size=0.05, w_explain=0.05),
    "insurance":  dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.05, w_explain=0.10),
    "gov":        dict(w_perf=0.55, w_train_time=0.10, w_infer_time=0.10, w_size=0.10, w_explain=0.15),
    "legal":      dict(w_perf=0.55, w_train_time=0.10, w_infer_time=0.10, w_size=0.10, w_explain=0.15),

    # Latency-sensitive
    "ads":        dict(w_perf=0.50, w_train_time=0.20, w_infer_time=0.20, w_size=0.05, w_explain=0.05),
    "gaming":     dict(w_perf=0.50, w_train_time=0.20, w_infer_time=0.20, w_size=0.05, w_explain=0.05),
    "realtime":   dict(w_perf=0.50, w_train_time=0.20, w_infer_time=0.20, w_size=0.05, w_explain=0.05),

    # Size/edge
    "edge":       dict(w_perf=0.50, w_train_time=0.15, w_infer_time=0.20, w_size=0.15, w_explain=0.00),
    "mobile":     dict(w_perf=0.50, w_train_time=0.15, w_infer_time=0.20, w_size=0.15, w_explain=0.00),
    "iot":        dict(w_perf=0.50, w_train_time=0.15, w_infer_time=0.20, w_size=0.15, w_explain=0.00),

    # Domain examples
    "marketing":  dict(w_perf=0.50, w_train_time=0.20, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "retail":     dict(w_perf=0.50, w_train_time=0.20, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "logistics":  dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "manufacturing": dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "energy":     dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "telco":      dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "cybersecurity": dict(w_perf=0.55, w_train_time=0.10, w_infer_time=0.20, w_size=0.05, w_explain=0.10),
    "agriculture": dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "education":  dict(w_perf=0.50, w_train_time=0.20, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "sports":     dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.10, w_explain=0.05),

    # 🔥 Added exactly from Upload.js dropdown
    "retail & e-commerce": dict(w_perf=0.50, w_train_time=0.20, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "manufacturing & iot": dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "smart cities & transport": dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "energy & environment": dict(w_perf=0.55, w_train_time=0.15, w_infer_time=0.15, w_size=0.10, w_explain=0.05),
    "social media & nlp": dict(w_perf=0.60, w_train_time=0.10, w_infer_time=0.15, w_size=0.05, w_explain=0.10),
}

def _normalize(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0: return a
    if np.allclose(a.max(), a.min()):
        return np.ones_like(a) * 0.5
    return (a - a.min())/(a.max()-a.min()+1e-12)

def pick_app_weights(app: str, cfg: RunConfig):
    base = dict(w_perf=cfg.w_perf, w_train_time=cfg.w_train_time, w_infer_time=cfg.w_infer_time, w_size=cfg.w_size, w_explain=cfg.w_explain)
    if not app: return base
    key = app.strip().lower()
    w = APP_PRIORS.get(key, APP_PRIORS["generic"]).copy()
    # override missing keys with base
    for k,v in base.items():
        w.setdefault(k, v)
    return w

def rank_models(results, task, cfg: RunConfig, application="generic"):
    w = pick_app_weights(application, cfg)
    train_times = _normalize([r.get("train_time_s", 0) for r in results])
    infer_times = _normalize([r.get("infer_time_s_per_row", 0) for r in results])
    sizes = _normalize([r.get("size_mb", 0) for r in results])
    explains = _normalize([r.get("explainability", 0.5) for r in results])

    if task=="classification":
        perf = _normalize([ (r.get("accuracy",0)+r.get("f1_weighted",0)+(r.get("roc_auc") or 0))/3 for r in results ])
    else:
        r2s = np.array([r.get("r2",0) for r in results])
        maes = np.array([r.get("mae", np.inf) for r in results])
        rmses = np.array([r.get("rmse", np.inf) for r in results])
        perf = _normalize((r2s - (np.nanmin(r2s) if r2s.size else 0)) + (1/(1+maes)) + (1/(1+rmses)))

    scores = (w["w_perf"]*perf + w["w_train_time"]*(1-train_times) + w["w_infer_time"]*(1-infer_times) + w["w_size"]*(1-sizes) + w["w_explain"]*explains)
    ranked = sorted([dict(**r, suitability_score=float(s)) for r,s in zip(results, scores)], key=lambda x: x["suitability_score"], reverse=True)
    return ranked