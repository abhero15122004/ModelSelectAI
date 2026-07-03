import os, matplotlib.pyplot as plt

def _bar_labels(ax, values):
    for i,v in enumerate(values):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8, rotation=0)

def plot_top5_classification(top5, out_dir):
    names = [t["name"] for t in top5]
    acc = [t.get("accuracy",0) for t in top5]
    f1 = [t.get("f1_weighted",0) for t in top5]
    auc = [t.get("roc_auc",0) if t.get("roc_auc") is not None else 0 for t in top5]
    x = range(len(top5))
    # metrics
    plt.figure(figsize=(10,5))
    plt.bar(x, acc); plt.xticks(x, names, rotation=30); plt.title("Accuracy")
    _bar_labels(plt.gca(), acc); plt.tight_layout()
    p1 = os.path.join(out_dir, "figs", "cls_accuracy.png"); plt.savefig(p1, dpi=150); plt.close()
    plt.figure(figsize=(10,5))
    plt.bar(x, f1); plt.xticks(x, names, rotation=30); plt.title("F1-weighted")
    _bar_labels(plt.gca(), f1); plt.tight_layout()
    p2 = os.path.join(out_dir, "figs", "cls_f1.png"); plt.savefig(p2, dpi=150); plt.close()
    plt.figure(figsize=(10,5))
    plt.bar(x, auc); plt.xticks(x, names, rotation=30); plt.title("ROC-AUC (binary only)")
    _bar_labels(plt.gca(), auc); plt.tight_layout()
    p3 = os.path.join(out_dir, "figs", "cls_auc.png"); plt.savefig(p3, dpi=150); plt.close()
    # speed/size
    speed = [1.0/max(t.get("infer_time_s_per_row",1e-9),1e-9) for t in top5]
    size = [t.get("size_mb",0) for t in top5]
    plt.figure(figsize=(10,4)); plt.bar(x, speed); plt.xticks(x,names,rotation=30); plt.title("Throughput (approx rows/sec)")
    p4 = os.path.join(out_dir, "figs", "throughput.png"); plt.tight_layout(); plt.savefig(p4,dpi=150); plt.close()
    plt.figure(figsize=(10,4)); plt.bar(x, size); plt.xticks(x,names,rotation=30); plt.title("Model size (MB)")
    p5 = os.path.join(out_dir, "figs", "size.png"); plt.tight_layout(); plt.savefig(p5,dpi=150); plt.close()
    return [p1,p2,p3,p4,p5]

def plot_top5_regression(top5, out_dir):
    names = [t["name"] for t in top5]
    r2 = [t.get("r2",0) for t in top5]
    mae = [t.get("mae",0) for t in top5]
    rmse = [t.get("rmse",0) for t in top5]
    x = range(len(top5))
    # metrics
    plt.figure(figsize=(10,5)); plt.bar(x, r2); plt.xticks(x, names, rotation=30); plt.title("R2"); _bar_labels(plt.gca(), r2)
    p1 = os.path.join(out_dir, "figs", "reg_r2.png"); plt.tight_layout(); plt.savefig(p1, dpi=150); plt.close()
    plt.figure(figsize=(10,5)); plt.bar(x, mae); plt.xticks(x, names, rotation=30); plt.title("MAE"); _bar_labels(plt.gca(), mae)
    p2 = os.path.join(out_dir, "figs", "reg_mae.png"); plt.tight_layout(); plt.savefig(p2, dpi=150); plt.close()
    plt.figure(figsize=(10,5)); plt.bar(x, rmse); plt.xticks(x, names, rotation=30); plt.title("RMSE"); _bar_labels(plt.gca(), rmse)
    p3 = os.path.join(out_dir, "figs", "reg_rmse.png"); plt.tight_layout(); plt.savefig(p3, dpi=150); plt.close()
    # speed/size
    speed = [1.0/max(t.get("infer_time_s_per_row",1e-9),1e-9) for t in top5]
    size = [t.get("size_mb",0) for t in top5]
    plt.figure(figsize=(10,4)); plt.bar(x, speed); plt.xticks(x,names,rotation=30); plt.title("Throughput (approx rows/sec)")
    p4 = os.path.join(out_dir, "figs", "throughput.png"); plt.tight_layout(); plt.savefig(p4,dpi=150); plt.close()
    plt.figure(figsize=(10,4)); plt.bar(x, size); plt.xticks(x,names,rotation=30); plt.title("Model size (MB)")
    p5 = os.path.join(out_dir, "figs", "size.png"); plt.tight_layout(); plt.savefig(p5,dpi=150); plt.close()
    return [p1,p2,p3,p4,p5]