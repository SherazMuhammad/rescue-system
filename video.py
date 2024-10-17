import cv2
import streamlit as st
from ultralytics import YOLO
import os

# Load the YOLOv10 model (with your custom-trained weights)
model = YOLO("best.pt")  # Ensure that 'best.pt' is compatible with YOLOv10

# Function to perform inference on a video
def process_video(input_video_path, output_video_path):
    # Open the input video using OpenCV
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame (bounding boxes, labels, scores)
        annotated_frame = results[0].plot()  # This should remain unchanged, assuming 'plot' method is similar in YOLOv10
        
        # Write the annotated frame to the output video
        out.write(annotated_frame)
        
    # Release the video objects
    cap.release()
    out.release()

# Streamlit app
st.title("YOLOv10m Video Inference")
st.write("Upload a video to perform object detection.")

# File uploader for video input
input_video_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

if input_video_file is not None:
    # Save the uploaded video to a temporary location
    input_video_path = os.path.join("temp_input.mp4")
    with open(input_video_path, "wb") as f:
        f.write(input_video_file.read())
    
    # Specify the output video path
    output_video_path = "output_video.mp4"
    
    # Button to start processing the video
    if st.button("Process Video"):
        with st.spinner("Processing..."):
            process_video(input_video_path, output_video_path)
        
        # Provide a download link for the output video
        st.success("Inference complete!")
        st.write("Output video:")
        st.video(output_video_path)

        # Create a download link
        with open(output_video_path, "rb") as f:
            st.download_button("Download Output Video", f, "output_video.mp4", "video/mp4")

    # Clean up the temporary files
    os.remove(input_video_path)
