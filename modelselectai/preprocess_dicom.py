import os, pydicom, cv2, numpy as np

def load_dicom_folder(folder, size=(224,224), label_from_subfolder=True, max_files=None):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith(".dcm"):
                files.append(os.path.join(root, f))
    if max_files: files = files[:max_files]
    X, y = [], []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array.astype("float32")
            arr = arr - arr.min()
            if arr.max() != 0:
                arr = arr / (arr.max()+1e-9)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            arr = cv2.resize(arr, size)
            X.append(arr)
            label = os.path.basename(os.path.dirname(f)) if label_from_subfolder else 0
            y.append(label)
        except Exception:
            continue
    if not X:
        return np.zeros((0, *size, 3), dtype="float32"), []
    return np.stack(X), y