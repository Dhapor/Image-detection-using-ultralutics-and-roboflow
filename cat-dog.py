import streamlit as st
import cv2  # Import OpenCV
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO("best.pt")

# Streamlit title
st.title("YOLO Model Deployment")

# Image upload option
uploaded_file = st.file_uploader("Upload an image...", type=["jpeg", "jpg", "png"])
if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert the image to OpenCV format (BGR)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Run prediction using the YOLO model
    results = model(img_cv)  # Use the OpenCV image for YOLO
    result_img = results[0].plot()  # Get the plot with detected objects
    
    # Convert the result image to PIL Image for Streamlit
    result_img = Image.fromarray(result_img)
    st.image(result_img, caption="Detected Objects", use_column_width=True)
