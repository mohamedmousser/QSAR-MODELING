from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from matplotlib import pyplot
import pandas as pd
import numpy as np
import math
import xgboost as xgb
import os
from sklearn import preprocessing
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

target = 'AKT1'

separator = ','

def preapre_matrix(X, min=None, max=None):
    if (min is None):
        min = X.min(axis=0)
    if (max is None):
        max = X.max(axis=0)
    res = (X.T - min[:, None]).T
    diff = (max[:, None] - min[:, None])
    res = np.divide(res.T, diff).T
    return X, min, max
    
    
train_filename = 'train_'+target+'.csv'
test_filename = 'test_'+target+'.csv'

train_df = pd.read_csv(train_filename, sep=separator)
test_df = pd.read_csv(test_filename, sep=separator)
print(train_df.shape, test_df.shape)


train_molecules_names = train_df['CHEMBL_ID']
test_molecules_names = test_df['CHEMBL_ID']

train_df = train_df.drop(columns=['CHEMBL_ID'])
test_df = test_df.drop(columns=['CHEMBL_ID'])
descriptors_names = list(train_df.columns)

print(train_df.columns[0])

y_train = train_df['Activity'].values
X_train = train_df.drop(columns='Activity')

train_descriptors = list(X_train.columns)

X_train = X_train.values
X_train, min, max = preapre_matrix(X_train)

y_test = test_df['Activity'].values
X_test = test_df.drop(columns='Activity').values
X_test, min, max = preapre_matrix(X_test, min, max)

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)

# encode classes
le = preprocessing.LabelEncoder()

le.fit(y_train)
le.classes_ = np.flip(le.classes_, axis=None)

le.classes_

y_train_transformed = le.transform(y_train)
y_test_transformed = le.transform(y_test)


from sklearn.svm import SVC

# 2. Define SVM classifier and hyperparameter grid
svc = SVC(probability=False)

# 3. Set up GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(svc, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train_transformed)

# Best estimator
best_svm = grid_search.best_estimator_

# 4. Evaluate on test data
y_pred = best_svm.predict(X_test)
y_proba = best_svm.predict_proba(X_test)[:, 1]

# Accuracy
accuracy = accuracy_score(y_test_transformed, y_pred)

# Sensitivity (Recall for positive class)
sensitivity = recall_score(y_test_transformed, y_pred, pos_label=1)

# Specificity (Recall for negative class)
conf_matrix = confusion_matrix(y_test_transformed, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# AUC
auc_score = roc_auc_score(y_test_transformed, y_proba)

# 5. 10-Fold Cross-validation AUC score
cv_scores = cross_val_score(best_svm, X_train, y_train, cv=10, scoring='accuracy')
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# 6. Print results
print("===== Model Evaluation Report =====")
print("Best Hyperparameters:", grid_search.best_params_)
print(f"Accuracy:     {accuracy:.3f}")
print(f"Sensitivity:  {sensitivity:.3f}")
print(f"Specificity:  {specificity:.3f}")
print(f"AUC:          {auc_score:.3f}")
print(f"10-Fold CV AUC:       {cv_mean:.3f} ± {cv_std:.3f}")
print("===================================")









