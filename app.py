import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("waste_classification_model.h5")  # Ensure the correct model path

# Streamlit App Title
st.markdown("""
    <h1 style='text-align: center; color: green;'>♻️ Waste Classification using CNN</h1>
    <h4 style='text-align: center; color: grey;'>Upload an image to classify it as Recyclable or Organic Waste</h4>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Resize image to match model input
    img_resized = cv2.resize(img_array, (224, 224))
    img_resized = img_resized / 255.0  # Normalize
    img_resized = np.expand_dims(img_resized, axis=0)  # Expand dimensions
    
    # Display the uploaded image
    st.image(image, caption='📷 Uploaded Image', use_column_width=True)
    
    # Make a prediction
    prediction = model.predict(img_resized)[0][0]
    
    # Display the result
    if prediction >= 0.5:
        st.write("### 🟢 This image is classified as **Organic Waste**")
    else:
        st.write("### 🔵 This image is classified as **Recyclable Waste**")
