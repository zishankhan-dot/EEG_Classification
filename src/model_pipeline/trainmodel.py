from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

models = {
    'LDA': Pipeline([('scaler', StandardScaler()), ('clf', LinearDiscriminantAnalysis())]),
    'RF':  Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))]),
    'MLP': Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42))]),
}