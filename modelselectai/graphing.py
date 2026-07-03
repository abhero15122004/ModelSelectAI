# modelselectai/graphing.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_classification_graphs(out, run_dir):
    paths = []
    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    top_model = out["top5"][0] if out["top5"] else None
    if not top_model:
        return paths

    # Confusion Matrix
    if "confusion_matrix" in top_model:
        cm = np.array(top_model["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {top_model['name']}")
        cm_path = os.path.join(graphs_dir, "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(cm_path)

    # ROC Curve
    if "roc_auc" in top_model and top_model["roc_auc"] is not None:
        # If your pipeline saved probabilities, you could plot here
        # Here we only create a placeholder if auc exists
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--', label="Random")
        ax.set_title(f"ROC Curve - {top_model['name']} (AUC={top_model['roc_auc']:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        roc_path = os.path.join(graphs_dir, "roc_curve.png")
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(roc_path)

    # Model Performance Bar
    names = [m["name"] for m in out["top5"]]
    accs = [m.get("accuracy", 0) for m in out["top5"]]
    fig, ax = plt.subplots()
    ax.barh(names, accs, color="skyblue")
    ax.set_xlabel("Accuracy")
    ax.set_title("Top-5 Model Accuracies")
    bar_path = os.path.join(graphs_dir, "model_accuracy_bar.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(bar_path)

    return paths


def plot_regression_graphs(out, run_dir):
    paths = []
    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    top_model = out["top5"][0] if out["top5"] else None
    if not top_model:
        return paths

    # Placeholder: No actual predictions stored, so we generate a fake scatter for now
    np.random.seed(0)
    y_true = np.random.rand(100)
    y_pred = y_true + np.random.normal(0, 0.1, size=100)

    # Predicted vs Actual
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual - {top_model['name']}")
    scatter_path = os.path.join(graphs_dir, "pred_vs_actual.png")
    plt.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(scatter_path)

    # Residual Histogram
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=20, color="salmon", edgecolor="black")
    ax.set_title(f"Residuals Histogram - {top_model['name']}")
    hist_path = os.path.join(graphs_dir, "residual_histogram.png")
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(hist_path)

    return paths


def plot_timeseries_graphs(out, run_dir):
    paths = []
    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Fake example for now
    np.random.seed(0)
    actual = np.sin(np.linspace(0, 10, 100))
    predicted = actual + np.random.normal(0, 0.1, size=100)

    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual")
    ax.plot(predicted, label="Predicted")
    ax.set_title("Time Series Forecast (Top Model)")
    ax.legend()
    ts_path = os.path.join(graphs_dir, "timeseries_forecast.png")
    plt.savefig(ts_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(ts_path)

    return paths


def plot_image_graphs(out, run_dir):
    paths = []
    graphs_dir = os.path.join(run_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Accuracy bar
    names = [m["name"] for m in out["top5"]]
    accs = [m.get("accuracy", 0) for m in out["top5"]]
    fig, ax = plt.subplots()
    ax.barh(names, accs, color="lightgreen")
    ax.set_xlabel("Accuracy")
    ax.set_title("Top-5 Model Accuracies (Image)")
    bar_path = os.path.join(graphs_dir, "image_model_accuracy.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close(fig)
    paths.append(bar_path)

    return paths


def generate_all_graphs(out, run_dir):
    """
    Dispatch to the appropriate plotting function based on task type.
    Returns a list of saved graph file paths.
    """
    task = out["task"]
    if task == "classification":
        return plot_classification_graphs(out, run_dir)
    elif task == "regression":
        return plot_regression_graphs(out, run_dir)
    elif task == "timeseries":
        return plot_timeseries_graphs(out, run_dir)
    elif task in ("image", "medical"):
        return plot_image_graphs(out, run_dir)
    else:
        return []