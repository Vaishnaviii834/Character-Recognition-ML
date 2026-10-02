import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier


# Load dataset
X = pd.read_csv("dataset/X.csv")
y = pd.read_csv("dataset/y.csv").values.ravel()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.values)
X_test = scaler.transform(X_test.values)


# =========================
# KNN
# =========================

print("\nTraining KNN model...")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("KNN training completed")


# =========================
# SVM
# =========================

print("\nTraining SVM model...")

# Use a smaller subset for faster SVM training
svm_train_size = 15000

X_svm = X_train[:svm_train_size]
y_svm = y_train[:svm_train_size]

svm = CalibratedClassifierCV(
    SVC(kernel="rbf"),
    ensemble=False
)

svm.fit(X_svm, y_svm)

joblib.dump(svm, "models/svm_model.pkl")
joblib.dump(scaler, "models/svm_scaler.pkl")

print(f"SVM training completed using {svm_train_size} samples")


# =========================
# Neural Network
# =========================

print("\nTraining Neural Network model...")

nn = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=20,
    random_state=42,
    verbose=True
)

nn.fit(X_train, y_train)

joblib.dump(nn, "models/nn_model.pkl")
joblib.dump(scaler, "models/nn_scaler.pkl")

print("Neural Network training completed")


print("\nAll models trained and saved successfully.")