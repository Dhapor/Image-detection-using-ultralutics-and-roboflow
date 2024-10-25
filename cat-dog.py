# import streamlit as st
# # import torch
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np

# # Load YOLO model
# model = YOLO("best.pt")

# # Streamlit title
# st.title("YOLO Model Deployment")

# # Image upload option
# uploaded_file = st.file_uploader("Upload an image...", type=["jpeg", "jpg", "png"])
# if uploaded_file is not None:
#     # Load and display the uploaded image
#     image = Image.open(uploaded_file)
#     img_array = np.array(image)
#     st.image(image, caption="Uploaded Image", use_column_width=True)
    
#     # Run prediction
#     results = model(img_array)
#     result_img = results[0].plot()
    
#     # Convert result image to PIL Image for Streamlit
#     result_img = Image.fromarray(result_img)
#     st.image(result_img, caption="Detected Objects", use_column_width=True)



import streamlit as st
from ultralytics import YOLO

st.title("Test Ultralytics in Streamlit")

try:
    # Load YOLO model
    model = YOLO("path/to/your/best.pt")  # Adjust the path
    st.write("Ultralytics YOLO model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

