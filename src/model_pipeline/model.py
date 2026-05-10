from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

models = {
    # shrinkage='auto' regularises LDA for small datasets — better than plain LDA
    'LDA': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ]),

    # SVM with RBF kernel — best general performer for small EEG feature sets
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
    ]),

    # larger forest + balanced class weight handles unequal T1/T2 trial counts
    'RF': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=500, max_depth=10,
                                       class_weight='balanced', random_state=42))
    ]),

    # deeper MLP + more iterations to actually converge
    'MLP': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(64, 32, 16),
                              max_iter=1000, learning_rate='adaptive',
                              random_state=42))
    ]),
}