import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.model_pipeline.model import models


def withinSubjectEval(X, Y, Groups, n_splits=5):
    subjects = np.unique(Groups)
    results  = []

    for s in subjects:
        idx   = Groups == s
        X_sub = X[idx]
        Y_sub = Y[idx]

        # skip subjects with too few epochs or only one class
        if len(X_sub) < n_splits or len(np.unique(Y_sub)) < 2:
            print(f"Subject {s:02d} | skipped ({len(X_sub)} epochs)")
            continue

        # use fewer splits if a class has fewer members than n_splits
        min_class_count = min(np.bincount(Y_sub - Y_sub.min()))
        actual_splits   = min(n_splits, min_class_count)
        skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)

        for name, pipeline in models.items():
            accs, f1s = [], []

            for train_idx, test_idx in skf.split(X_sub, Y_sub):
                X_train, X_test = X_sub[train_idx], X_sub[test_idx]
                y_train, y_test = Y_sub[train_idx], Y_sub[test_idx]

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                accs.append(accuracy_score(y_test, y_pred))
                f1s.append(f1_score(y_test, y_pred, average='macro'))

            results.append({
                'Subject':  s,
                'Model':    name,
                'Accuracy': np.mean(accs),
                'F1':       np.mean(f1s)
            })

        # print best model result for this subject
        subj_results = [r for r in results if r['Subject'] == s]
        best = max(subj_results, key=lambda r: r['Accuracy'])
        print(f"Subject {s:02d} | Best: {best['Model']} | Acc: {best['Accuracy']:.3f} | F1: {best['F1']:.3f}")

    df = pd.DataFrame(results)
    print("\nFinal Model Comparison:")
    print(df.groupby('Model')[['Accuracy', 'F1']].mean().round(3))
    return df
