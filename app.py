import streamlit as st
import joblib
import numpy as np
from PIL import Image


# Page configuration
st.set_page_config(
    page_title="Character Recognition",
    page_icon="🔢",
    layout="centered"
)


# Title
st.title("Character Recognition System")
st.write("Upload a handwritten digit image and select a machine learning model.")


# Model selection
model_name = st.selectbox(
    "Select Model",
    ["Neural Network", "SVM", "KNN"]
)


# Load selected model
if model_name == "Neural Network":
    model = joblib.load("models/nn_model.pkl")
    scaler = joblib.load("models/nn_scaler.pkl")

elif model_name == "SVM":
    model = joblib.load("models/svm_model.pkl")
    scaler = joblib.load("models/svm_scaler.pkl")

else:
    model = joblib.load("models/knn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")


# Upload image
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("L")

    # Display uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        width=200
    )

    # Resize image
    image = image.resize((28, 28))

    # Convert to NumPy array
    image_array = np.array(image)

    # Invert image if background is white
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # Flatten image
    image_array = image_array.reshape(1, -1)

    # Scale image
    image_array = scaler.transform(image_array)

    # Prediction
    prediction = model.predict(image_array)

    # Display prediction
    st.success(f"Predicted Digit: {prediction[0]}")

    # Confidence
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(image_array)

        confidence = np.max(probabilities) * 100

        st.info(f"Confidence: {confidence:.2f}%")