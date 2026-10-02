import pandas as pd
import numpy as np
from PIL import Image


X = pd.read_csv("dataset/X.csv")
y = pd.read_csv("dataset/y.csv").values.ravel()

# Find the first image labeled 5
index = np.where(y == 5)[0][0]

image = X.iloc[index].values.astype(np.uint8)
image = image.reshape(28, 28)

Image.fromarray(image).save("test_images/mnist_5.png")

print("Image saved: test_images/mnist_5.png")
print("Actual label:", y[index])