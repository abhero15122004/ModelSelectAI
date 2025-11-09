import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers (prevents plotting hangs)

import os
import json
import threading
import traceback
from typing import Tuple
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from modelselectai.utils import new_run_dir as utils_new_run_dir, save_json, log_exception
from modelselectai.loader import try_extract_archive
import main as main_module

# -------------------- Setup --------------------
UPLOAD_ROOT = "uploads"
RUNS_ROOT = "runs"
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(RUNS_ROOT, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "json", "xlsx", "xls", "parquet", "zip"}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "*"}})


# -------------------- Helpers --------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_job_stub(run_id: str, run_dir: str) -> None:
    """Create temporary JSON showing job started."""
    save_json({
        "run_id": run_id,
        "run_dir": run_dir,
        "out": {"status": "running"}
    }, os.path.join(run_dir, "summary.json"))


def pick_file(run_dir: str, filename: str):
    for folder in ["figs", "graphs", "artifacts", ""]:
        path = os.path.join(run_dir, folder, filename)
        if os.path.exists(path):
            return path
    return None


def _precreate_run() -> Tuple[str, str]:
    run_id, run_dir = utils_new_run_dir(base=RUNS_ROOT)
    return run_id, run_dir


def _monkeypatch_run_dir(run_id: str, run_dir: str):
    """Force run_pipeline to use same run_dir."""
    try:
        main_module.new_run_dir = lambda: (run_id, run_dir)
    except Exception:
        # If main_module cannot be monkeypatched, we'll still try to proceed.
        pass


# -------------------- Core Training Thread --------------------
def _train_job(run_id: str, run_dir: str, input_path: str, application: str):
    summary_path = os.path.join(run_dir, "summary.json")
    try:
        # IMPORTANT: ensure main.run_pipeline writes into the same run_dir we created
        _monkeypatch_run_dir(run_id, run_dir)

        app_domain = (application or "generic").strip().lower()

        # Debug: log where we expect main to write
        print(f"[DEBUG] _train_job starting. run_id={run_id}, run_dir={run_dir}, input_path={input_path}, application={app_domain}", flush=True)

        # Call pipeline (this is your existing ML logic)
        result = main_module.run_pipeline(
            input_path=input_path,
            application=app_domain,
            task_hint="auto"
        )

        # When run_pipeline returns, write the standard summary to the run_dir we control
        if isinstance(result, dict):
            result.setdefault("status", "done")
            save_json({
                "run_id": run_id,
                "run_dir": run_dir,
                "out": result
            }, summary_path)
        else:
            save_json({
                "run_id": run_id,
                "run_dir": run_dir,
                "out": {"status": "done", "result": str(result)}
            }, summary_path)

        print(f"[INFO] Run {run_id} finished. Results written to {summary_path}", flush=True)

    except Exception as e:
        tb = traceback.format_exc()
        log_exception(e)
        save_json({
            "run_id": run_id,
            "run_dir": run_dir,
            "out": {
                "status": "error",
                "error": str(e),
                "traceback": tb
            }
        }, summary_path)
        print(f"[ERROR] Run {run_id} failed: {e}", flush=True)


# -------------------- API --------------------
@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"ok": True})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part found"}), 400

        f = request.files["file"]

        if f.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # ✅ Save file to uploads directory
        os.makedirs(UPLOAD_ROOT, exist_ok=True)
        filename = secure_filename(f.filename)
        saved_path = os.path.join(UPLOAD_ROOT, filename)
        f.save(saved_path)

        print(f"✅ File received and saved at {saved_path}", flush=True)

        # Detect if it’s a ZIP or dataset
        input_path = saved_path
        if filename.lower().endswith(".zip"):
            folder = try_extract_archive(saved_path, out_dir=os.path.join(UPLOAD_ROOT, filename + "_extracted"))
            if folder and os.path.isdir(folder):
                input_path = folder

        # Get application domain
        application = request.form.get("application") or "generic"

        # Create run id / dir (we'll tell main to write to this exact folder)
        run_id, run_dir = _precreate_run()
        ensure_job_stub(run_id, run_dir)

        # Start training asynchronously (make non-daemon so system won't kill thread prematurely)
        t = threading.Thread(target=_train_job, args=(run_id, run_dir, input_path, application))
        t.daemon = False
        t.start()

        return jsonify({"run_id": run_id, "status": "started", "application": application})

    except Exception as e:
        import traceback
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<run_id>", methods=["GET"])
def api_results(run_id):
    run_dir = os.path.join(RUNS_ROOT, run_id)
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        return jsonify({"status": "running"}), 202

    with open(summary_path, "r") as f:
        summary = json.load(f)
    out = summary.get("out", {})
    status = out.get("status", "running")

    if status == "running":
        return jsonify({"status": "running"}), 202
    return jsonify(out), 200


@app.route("/api/fig/<run_id>/<filename>", methods=["GET"])
def api_fig(run_id: str, filename: str):
    run_dir = os.path.join(RUNS_ROOT, run_id)
    path = pick_file(run_dir, filename)
    if path:
        return send_file(path)
    return jsonify({"error": "Figure not found"}), 404


@app.route("/api/graph/<run_id>/<filename>", methods=["GET"])
def api_graph(run_id: str, filename: str):
    run_dir = os.path.join(RUNS_ROOT, run_id)
    path = pick_file(run_dir, filename)
    if path:
        return send_file(path)
    return jsonify({"error": "Graph not found"}), 404


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)