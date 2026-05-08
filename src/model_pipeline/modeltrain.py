from model_pipeline.data import dataPrep
import numpy as np 
from sklearn.model_selection import GroupKFold


X,Y,Groups=dataPrep("../../data/raw")  #where i have raw(1-20 each 3runs) stored
print(X.shape,Y.shape,Groups.shape)

# splitting data for train and test train will be (1-19 subjects )