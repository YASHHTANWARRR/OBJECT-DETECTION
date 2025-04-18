import numpy as np
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from matplotlib import pyplot as plt
import time
import os
import yaml

# Step 1: Configure the PiCamera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())

# Step 2: Initialize the YOLO model (YOLOv8 pre-trained model or custom model)
model = YOLO('yolov8m.pt')  # Using a pre-trained YOLOv8 model, can be replaced with a custom-trained model.

# Step 3: Prepare for training

# Define the path to the dataset's YAML configuration file
train_data_path = "/path/to/your/data.yaml"  # Adjust this path to your dataset's data.yaml file.
epochs = 50  # Number of epochs for training
imgsz = 640  # Image size for training
batch_size = 8  # Batch size for training

# Step 4: Train the model (This part trains the model on the dataset provided in the YAML file)
# Make sure the data.yaml file contains correct paths and labels for training
model.train(data=train_data_path, epochs=epochs, imgsz=imgsz, batch=batch_size)

# Step 5: Load the trained model (this is the trained model after the training phase)
# After training, YOLOv8 saves the best model weights by default.
trained_model = YOLO('runs/detect/train/weights/best.pt')  # Adjust the path if necessary based on your output

# Step 6: Start the camera preview and wait for the camera to warm up
picam2.start()

# Step 7: Process the camera stream for real-time object detection
while True:
    start_time = time.time()

    # Capture a frame from the Pi Camera
    frame = picam2.capture_array()

    # Resize the frame for YOLOv8 input size (640x640)
    img = cv2.resize(frame, (640, 640))

    # Step 8: Run YOLO object detection on the captured frame using the trained model
    results = trained_model(img)

    # Step 9: Extract bounding boxes and labels from the results
    boxes = results[0].boxes
    im_result = results[0].plot()  # Plotting the results on the frame

    # Step 10: Display the resulting image with detections
    plt.imshow(im_result)
    plt.axis('off')  # Turn off axis
    plt.show()

    # Step 11: Print the FPS (Frames Per Second)
    fps = 1 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")

    # Step 12: Optionally: Break the loop after a certain condition (e.g., a key press)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 13: Release the camera and clean up
picam2.stop()
cv2.destroyAllWindows()
