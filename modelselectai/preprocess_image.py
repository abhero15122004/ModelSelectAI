import os, numpy as np, cv2

def load_images_from_folder(folder, size=(224,224), max_files=None, label_from_subfolder=True):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                files.append(os.path.join(root, f))
    if max_files: files = files[:max_files]
    X, y = [], []
    for f in files:
        im = cv2.imread(f)
        if im is None: continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, size)
        X.append(im.astype("float32")/255.0)
        label = os.path.basename(os.path.dirname(f)) if label_from_subfolder else os.path.splitext(os.path.basename(f))[0]
        y.append(label)
    if not X:
        return np.zeros((0, *size, 3), dtype="float32"), []
    return np.stack(X), y