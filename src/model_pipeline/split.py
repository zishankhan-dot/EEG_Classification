import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score


def SplitData(x,y,groups):
    logo=LeaveOneGroupOut()
    foldScore=[]
    for fold,(train_idx,test_idx) in enumerate(logo.split(x,y,groups)):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        test_subject = np.unique(groups[test_idx])[0]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    
        foldScore.append(acc)
        print(f"Fold {fold+1:2d} | held-out subject: {test_subject:2d} | "
            f"train: {len(train_idx):4d} trials | test: {len(test_idx):3d} trials | "
            f"accuracy: {acc:.3f}")

