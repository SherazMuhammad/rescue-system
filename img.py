import cv2
import streamlit as st
from ultralytics import YOLO
import os

# Load the YOLOv10 model (with your custom-trained weights)
model = YOLO("best.pt")  # Ensure that 'best.pt' is compatible with YOLOv10

# Function to perform inference on an image
def process_image(input_image_path, output_image_path):
    # Read the image using OpenCV
    image = cv2.imread(input_image_path)
    
    # Perform inference on the image
    results = model(image)
    
    # Visualize the results on the image (bounding boxes, labels, scores)
    annotated_image = results[0].plot()
    
    # Save the annotated image to the output path
    cv2.imwrite(output_image_path, annotated_image)

# Streamlit app
st.title("YOLOv10 Image Inference")
st.write("Upload an image to perform object detection.")

# File uploader for image input
input_image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if input_image_file is not None:
    # Save the uploaded image to a temporary location
    input_image_path = os.path.join("temp_input.jpg")
    with open(input_image_path, "wb") as f:
        f.write(input_image_file.read())
    
    # Specify the output image path
    output_image_path = "output_image.jpg"
    
    # Button to start processing the image
    if st.button("Process Image"):
        with st.spinner("Processing..."):
            process_image(input_image_path, output_image_path)
        
        # Provide a download link for the output image
        st.success("Inference complete!")
        st.write("Output image:")
        st.image(output_image_path, use_column_width=True)

        # Create a download link
        with open(output_image_path, "rb") as f:
            st.download_button("Download Output Image", f, "output_image.jpg", "image/jpeg")

    # Clean up the temporary files
    os.remove(input_image_path)
