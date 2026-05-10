import os
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from src.model_pipeline.model import models
from src.model_pipeline.data import dataPrep
from src.model_pipeline.preprocessing import featureExtraction

def trainModel(data_path,save_dir="../../models/"):
    X,Y,Groups=dataPrep(data_path)

    X_feature=featureExtraction(X)
    skf=StratifiedKFold(shuffle=True)

    for name,pipeline in models.items():
        accs,f1 =[],[]

        for train_idx,test_idx in skf.split(X_feature,Y):
            X_train,X_test=X_feature[train_idx],X_feature[test_idx]
            Y_train,Y_test=Y[train_idx],Y[test_idx]

            pipeline.fit(X_train,Y_train)
            y_pred=pipeline.predict(X_test)
            accs.append(accuracy_score(Y_test,y_pred))
            f1.append(f1_score(Y_test,y_pred,average='macro'))
        
        #fitting all data for saving model 
        pipeline.fit(X_feature,Y)

        #create bundle to save 
        bundle={
            'model': pipeline,
            'accuracy': np.mean(accs),
            'f1': np.mean(f1)
        }

        #saveing the trained model
        joblib.dump(bundle,os.path.join(save_dir,f"{name}.pkl"))
        print("acc:", np.mean(accs), "f1:", np.mean(f1))



