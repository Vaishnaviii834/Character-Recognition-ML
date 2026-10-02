import joblib
import numpy as np
from PIL import Image


# Load trained model and scaler
model = joblib.load("models/nn_model.pkl")
scaler = joblib.load("models/nn_scaler.pkl")


# Enter image path
image_path = input("Enter image path: ")


# Load image
image = Image.open(image_path).convert("L")

# Resize image
image = image.resize((28, 28))

# Convert image to numpy array
image_array = np.array(image)

# Invert image if background is white
if image_array.mean() > 127:
    image_array = 255 - image_array

# Flatten image
image_array = image_array.reshape(1, -1)

# Scale image
image_array = scaler.transform(image_array)

# Predict
prediction = model.predict(image_array)

print("\nPredicted Digit:", prediction[0])