import cv2
import streamlit as st
from ultralytics import YOLO
import time

# Load the YOLOv10 model (with your custom-trained weights)
model = YOLO("best.pt")  # Ensure that 'best.pt' is compatible with YOLOv10

# Initialize session state to manage webcam feed start/stop
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

# Function to perform inference on webcam video stream
def process_webcam(fps_limit=10):
    # Open the phone camera stream using OpenCV (replace with your IP Webcam app's IP address)
    cap = cv2.VideoCapture("http://192.168.192.182:8080/video")
    
    # Check if the phone camera opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open mobile phone camera stream.")
        return
    
    st.write("Mobile phone camera feed loaded successfully. Press 'Stop' to end.")
    
    # Create a placeholder for the webcam feed to dynamically update it
    image_placeholder = st.empty()

    # Start the video stream
    prev_time = 0
    while st.session_state.webcam_running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break
        
        # Limit the FPS by adding a delay
        curr_time = time.time()
        if (curr_time - prev_time) > 1.0 / fps_limit:
            prev_time = curr_time
            
            # Perform inference on the frame
            results = model(frame)
            
            # Visualize the results on the frame (bounding boxes, labels, scores)
            annotated_frame = results[0].plot()  # Plotting annotations on the frame
            
            # Convert the frame from BGR (OpenCV default) to RGB for Streamlit display
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Dynamically update the placeholder with the new frame
            image_placeholder.image(rgb_frame, channels="RGB", caption="Mobile Camera feed")

        # Sleep for a short time to prevent high CPU usage
        time.sleep(0.01)
    
    # Release the video stream when done
    cap.release()

# Streamlit app
st.title("YOLOv10m Mobile Camera Inference")

# Start Webcam Button
if not st.session_state.webcam_running:
    if st.button("Start Mobile Camera"):
        st.session_state.webcam_running = True

# Stop Webcam Button
if st.session_state.webcam_running:
    if st.button("Stop Mobile Camera"):
        st.session_state.webcam_running = False

# Run the mobile camera only if it is running
if st.session_state.webcam_running:
    process_webcam(fps_limit=20)
