import streamlit as st
# import torch
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
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Run prediction
    results = model(img_array)
    result_img = results[0].plot()
    
    # Convert result image to PIL Image for Streamlit
    result_img = Image.fromarray(result_img)
    st.image(result_img, caption="Detected Objects", use_column_width=True)


# import torch
# import cv2
# import numpy as np
# from PIL import Image
# import streamlit as st

# # Load YOLO model
# model = torch.load("best.pt", map_location=torch.device('cpu'))  # Adjust the path as needed
# model.eval()  # Set the model to evaluation mode

# # Streamlit interface
# st.title("YOLO Model Deployment")

# # Image upload option
# uploaded_file = st.file_uploader("Upload an image...", type=["jpeg", "jpg", "png"])
# if uploaded_file is not None:
#     # Load and display the uploaded image
#     image = Image.open(uploaded_file)
#     img_array = np.array(image)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess the image for the model
#     img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # Convert to tensor and change dimensions
#     img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
#     img_tensor = img_tensor.float() / 255.0  # Normalize to [0, 1]

#     # Run prediction
#     with torch.no_grad():  # Disable gradient calculation
#         results = model(img_tensor)  # Get predictions

#     # Process results (you might need to modify this based on your model's output)
#     # Assuming the model outputs boxes, scores, and class ids
#     boxes = results[0]['boxes']  # Example access; modify based on your model's output structure
#     scores = results[0]['scores']
#     class_ids = results[0]['class_ids']

#     # Draw boxes on the image (modify this to fit your output format)
#     for box, score, class_id in zip(boxes, scores, class_ids):
#         if score > 0.5:  # Confidence threshold
#             x1, y1, x2, y2 = box.int()  # Convert to integer if needed
#             cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle

#     # Convert result image back to PIL for Streamlit
#     result_img = Image.fromarray(img_array)
#     st.image(result_img, caption="Detected Objects", use_column_width=True)
