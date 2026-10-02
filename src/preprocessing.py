import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load dataset
X = pd.read_csv("dataset/X.csv")
y = pd.read_csv("dataset/y.csv").values.ravel()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale pixel values
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.values)
X_test = scaler.transform(X_test.values)

print("Preprocessing completed successfully")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)