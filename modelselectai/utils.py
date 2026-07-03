import os, json, time, uuid, traceback

def new_run_dir(base="outputs"):
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    path = os.path.join(base, run_id)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "figs"), exist_ok=True)
    os.makedirs(os.path.join(path, "artifacts"), exist_ok=True)
    return run_id, path

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def safe_msg(msg: str):
    print(msg, flush=True)

def format_seconds(s: float) -> str:
    try:
        return f"{s:.6f}"
    except Exception:
        return str(s)

def print_block(text: str):
    print("\n" + "="*72)
    print(text)
    print("="*72 + "\n")

def log_exception(e: Exception) -> str:
    traceback.print_exc()
    return f"{type(e).__name__}: {e}"