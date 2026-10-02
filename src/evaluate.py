import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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


# Convert test data to NumPy
X_test = X_test.values


# Load scalers
knn_scaler = joblib.load("models/scaler.pkl")
svm_scaler = joblib.load("models/svm_scaler.pkl")
nn_scaler = joblib.load("models/nn_scaler.pkl")


# Scale test data
X_test_knn = knn_scaler.transform(X_test)
X_test_svm = svm_scaler.transform(X_test)
X_test_nn = nn_scaler.transform(X_test)


# Load models
knn = joblib.load("models/knn_model.pkl")
svm = joblib.load("models/svm_model.pkl")
nn = joblib.load("models/nn_model.pkl")


# Models and test data
models = {
    "KNN": (knn, X_test_knn),
    "SVM": (svm, X_test_svm),
    "Neural Network": (nn, X_test_nn)
}


print("\nModel Evaluation")
print("================")


for name, (model, test_data) in models.items():

    print(f"\nEvaluating {name}...")

    y_pred = model.predict(test_data)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")


print("\nEvaluation completed successfully.")