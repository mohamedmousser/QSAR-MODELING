import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import pandas as pd
import numpy as np

import os, sys

# hide TF C++ logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# silence stderr during TF load
stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

import tensorflow as tf

sys.stderr = stderr_backup


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout


os.chdir(os.getcwd())


# -------------------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------------------
def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"F1: {f1:.4f}")


# -------------------------------------------------------------------------------------
# MODEL: Neural Network
# -------------------------------------------------------------------------------------
def create_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# -------------------------------------------------------------------------------------
# USER INPUT
# -------------------------------------------------------------------------------------
Target = input("Type the kinase name please : ")
n_estim = int(input("Enter the optimal number of XGBoost decision trees please : "))


# -------------------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------------------
train_df = pd.read_csv(f"training_data/train_{Target}.csv")
test_df = pd.read_csv(f"training_data/test_{Target}.csv")

# Keep original copies
train_raw = train_df.copy()
test_raw = test_df.copy()


# -------------------------------------------------------------------------------------
# XGBOOST PHASE
# -------------------------------------------------------------------------------------
X_train_xgb = train_df.iloc[:, 2:].values
Y_train_xgb = train_df.iloc[:, 1].values

# Scale
scaler_xgb = MinMaxScaler()
X_train_xgb = scaler_xgb.fit_transform(X_train_xgb)

# Encode target
encoder = LabelEncoder()
Y_train_xgb = encoder.fit_transform(Y_train_xgb)

# Reverse labels (active = 1 after your inversion)
Y_train_xgb = 1 - Y_train_xgb

X_test_xgb = test_df.iloc[:, 2:].values
Y_test_xgb = test_df.iloc[:, 1].values

X_test_xgb = scaler_xgb.transform(X_test_xgb)
Y_test_xgb = 1 - encoder.transform(Y_test_xgb)

# XGBoost
xgb_model = xgb.XGBClassifier(
    colsample_bytree=0.4,
    learning_rate=0.15,
    gamma=0.3,
    max_depth=12,
    min_child_weight=1,
    n_estimators=n_estim,
)

xgb_model.fit(X_train_xgb, Y_train_xgb)

# Generate probabilities
train_df["xgb_pred"] = xgb_model.predict_proba(X_train_xgb)[:, 1]
test_df["xgb_pred"] = xgb_model.predict_proba(X_test_xgb)[:, 1]

# Store XGBoost test labels for later evaluation
y_test_saved = Y_test_xgb


# -------------------------------------------------------------------------------------
# BUILD FINAL DATASET (train + test merged)
# -------------------------------------------------------------------------------------
all_data = pd.concat([train_df, test_df], ignore_index=True)
X = all_data.iloc[:, 2:].values
Y = all_data.iloc[:, 1].values


# -------------------------------------------------------------------------------------
# TRAIN/TEST SPLIT FOR DNN
# -------------------------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)

# Scale
scaler_dnn = StandardScaler()
X_train = scaler_dnn.fit_transform(X_train)
X_test = scaler_dnn.transform(X_test)

# Encode
encoder = LabelEncoder()
Y_train = 1 - encoder.fit_transform(Y_train)
Y_test = 1 - encoder.transform(Y_test)


# -------------------------------------------------------------------------------------
# TRAIN DNN
# -------------------------------------------------------------------------------------
model = create_model(input_dim=X_train.shape[1])

# Replace NaNs safely
X_train = np.nan_to_num(X_train, nan=0)

model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)




# -------------------------------------------------------------------------------------
# METRICS ON DNN TEST
# -------------------------------------------------------------------------------------
y_prob = model.predict(X_test)

message = f"Congratulation, your {Target} QSAR model has been built, below is the model report"
border = "*" * (len(message) + 4)

print(border)
print(f"* {message} *")
print(border)

calculate_metrics(Y_test, y_prob)

#print(100*'*','\nPlease, if you use our programs in your research, do not forget to cite the following paper : \nMousser, M. O., Matougui, B., Chafaa, F., & Dems, M. A. (2025). Enhancing predictive modeling \nwith XGBoost-engineered probabilities and deep neural networks: \nA hybrid approach for building reliable kinase inhibition QSAR models.\nJournal of Molecular Graphics and Modelling, 109216.')
#print(100*'*')
message = (
    "Please, if you use our programs in your research, do not forget to cite the following paper :\n"
    "Mousser, M. O., Matougui, B., Chafaa, F., & Dems, M. A. (2025). Enhancing predictive modeling\n"
    "with XGBoost-engineered probabilities and deep neural networks:\n"
    "A hybrid approach for building reliable kinase inhibition QSAR models.\n"
    "Journal of Molecular Graphics and Modelling, 109216."
)

# Split lines
lines = message.split("\n")

# Determine maximum line length
max_len = max(len(line) for line in lines)

# Build border
border = "*" * (max_len + 4)

# Print box
print(border)
for line in lines:
    print(f"* {line.ljust(max_len)} *")
print(border)
