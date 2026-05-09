import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from src.model_pipeline.model import models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def SplitData(x,y,groups):
    logo=LeaveOneGroupOut()
    results=[]
    for fold,(train_idx,test_idx) in enumerate(logo.split(x,y,groups)):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        #storing test_subject for eval
        test_subject = np.unique(groups[test_idx])[0]

        for name,pipeline in models.items():
            #training on models -> pipeline in model.py
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            #cal metric
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            cm = confusion_matrix(y_test, y_pred)

            results.append({
                "Subject": test_subject,
                "Model":name,
                "Accuracy": acc,
                "F1": f1,
                "CM": cm,
                "y_true": y_test, 
                "y_pred": y_pred  
            })

    
        print(f"Subject {test_subject} | Acc: {acc:.3f} | F1: {f1:.3f}")
    
    return pd.DataFrame(results)

