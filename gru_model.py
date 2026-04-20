# =========================================
# GRU-Based Plant Stress Prediction
# =========================================
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# =========================================
# LOAD DATA
# =========================================

data = pd.read_excel("dataset.xlsx")

data = data[['soil_moisture_%', 'temperature_C', 'humidity_%', 'NDVI_index', 'crop_disease_status']]

X = data.drop('crop_disease_status', axis=1)
y = data['crop_disease_status']

# =========================================
# PREPROCESSING
# =========================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

def create_sequences(X, y, time_steps=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 5
X_seq, y_seq = create_sequences(X_scaled, y.values, time_steps)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# =========================================
# MODEL FUNCTION
# =========================================

def build_gru(units):
    model = Sequential([
        GRU(units, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =========================================
# EVALUATION FUNCTION
# =========================================

def evaluate(name, y_true, y_pred, inference_time):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Inference Time: {inference_time:.4f} sec")

    return acc, prec, rec, f1, inference_time

# =========================================
# RANDOM FOREST (BASELINE)
# =========================================

X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

rf = RandomForestClassifier()

start = time.time()
rf.fit(X_train_rf, y_train)
y_pred_rf = rf.predict(X_test_rf)
rf_time = time.time() - start

rf_metrics = evaluate("Random Forest", y_test, y_pred_rf, rf_time)

# =========================================
# ORIGINAL GRU
# =========================================

gru_model = build_gru(32)

gru_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

start = time.time()
y_pred_gru = (gru_model.predict(X_test) > 0.5).astype(int)
gru_time = time.time() - start

gru_metrics = evaluate("GRU (Original)", y_test, y_pred_gru, gru_time)

# =========================================
# TUNED GRU
# =========================================

tuned_model = build_gru(64)

tuned_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

start = time.time()
y_pred_tuned = (tuned_model.predict(X_test) > 0.5).astype(int)
tuned_time = time.time() - start

tuned_metrics = evaluate("GRU (Tuned)", y_test, y_pred_tuned, tuned_time)

# =========================================
# CONFUSION MATRIX (OPTIONAL)
# =========================================

cm = confusion_matrix(y_test, y_pred_gru)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("GRU Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# FINAL SUMMARY

print("\n=== FINAL COMPARISON ===")
print("Random Forest:", rf_metrics)
print("GRU:", gru_metrics)
print("Tuned GRU:", tuned_metrics)
