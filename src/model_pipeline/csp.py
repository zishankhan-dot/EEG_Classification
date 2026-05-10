from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def runCSP_withinSubject(X, Y, Groups, n_components=2):
    subjects = np.unique(Groups)
    results  = []

    for s in subjects:
        idx     = Groups == s
        X_sub   = X[idx]
        Y_sub   = Y[idx]

        # skip subjects with too few epochs after artifact rejection
        if len(X_sub) < 10 or len(np.unique(Y_sub)) < 2:
            print(f"Subject {s:02d} | skipped (only {len(X_sub)} epochs)")
            continue

        skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs    = []

        for train_idx, test_idx in skf.split(X_sub, Y_sub):
            X_train, X_test = X_sub[train_idx], X_sub[test_idx]
            y_train, y_test = Y_sub[train_idx], Y_sub[test_idx]

            csp = CSP(n_components=n_components, reg=None, log=True)
            X_train_csp = csp.fit_transform(X_train, y_train)
            X_test_csp  = csp.transform(X_test)

            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train_csp, y_train)
            accs.append(accuracy_score(y_test, lda.predict(X_test_csp)))

        mean_acc = np.mean(accs)
        results.append({'Subject': s, 'Accuracy': mean_acc})
        print(f"Subject {s:02d} | Acc: {mean_acc:.3f}")

    df = pd.DataFrame(results)
    print(f"\nMean Accuracy: {df['Accuracy'].mean():.3f}")
    return df

