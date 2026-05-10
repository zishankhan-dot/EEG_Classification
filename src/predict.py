import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.model_pipeline.data import dataPrep
from src.model_pipeline.preprocessing import featureExtraction

LABEL_MAP = {2: "Left (T1)", 3: "Right (T2)"}


def load_model(model_name, save_dir="models"):
    path = os.path.join(save_dir, f"{model_name}.pkl")
    return joblib.load(path)


def run_pipeline(raw, model_name, save_dir="models"):
    X, Y, _ = dataPrep(raw=raw, from_file=False)
    if len(X) == 0:
        return None, None, None, None

    X_feat = featureExtraction(X)

    bundle = load_model(model_name, save_dir)
    y_pred = bundle['model'].predict(X_feat)

    acc = accuracy_score(Y, y_pred)
    f1  = f1_score(Y, y_pred, average='macro')
    cm  = confusion_matrix(Y, y_pred, labels=[2, 3])

    sfreq       = 160
    event_times = np.arange(len(Y)) * (4 + 4)
    table = pd.DataFrame({
        'Epoch':        range(1, len(Y) + 1),
        'Ground Truth': [LABEL_MAP.get(y, str(y)) for y in Y],
        'Prediction':   [LABEL_MAP.get(p, str(p)) for p in y_pred],
        'Correct':      ['Yes' if y == p else 'NO' for y, p in zip(Y, y_pred)]
    })

    return table, acc, f1, cm
