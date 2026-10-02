from sklearn.datasets import fetch_openml
import pandas as pd

print("Downloading dataset...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target

print("Dataset downloaded")
print("Features:", X.shape)
print("Labels:", y.shape)

pd.DataFrame(X).to_csv("dataset/X.csv", index=False)
pd.DataFrame(y, columns=["label"]).to_csv("dataset/y.csv", index=False)

print("Dataset saved successfully")