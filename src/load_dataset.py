import pandas as pd

X = pd.read_csv("dataset/X.csv")
y = pd.read_csv("dataset/y.csv")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 labels:")
print(y.head())

print("\nFirst 5 rows of X:")
print(X.head())