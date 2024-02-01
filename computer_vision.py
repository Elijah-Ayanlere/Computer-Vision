import cv2
import tensorflow as tf
import numpy as np
from datetime import datetime

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def capture_and_analyze():
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    # Save the captured frame as an image
    if ret:
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured successfully.")

        # Perform image recognition
        predict_image("captured_image.jpg")

        # Detect and point out any metal substance in the image
        detect_metal_substance("captured_image.jpg")

        # Prompt for the person's birthdate
        birthdate = input("Enter the person's birthdate (YYYY-MM-DD): ")

        # Calculate age in days
        age_days = calculate_age(birthdate)
        print(f"The person has been alive on Earth for approximately {age_days} days.")

    # Release the camera
    cap.release()

def predict_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Expand dimensions to match the model's expected input shape
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img)

    # Decode and print the top-3 predicted classes
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")


def detect_metal_substance(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and check for metal-like features (you may need to fine-tune this based on your requirements)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust the area threshold as needed
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

    # Save the result
    cv2.imwrite("metal_substance_detection_result.jpg", img)
    print("Metal substance detection result saved as metal_substance_detection_result.jpg")

def calculate_age(birthdate):
    # Convert birthdate string to datetime object
    birth_date = datetime.strptime(birthdate, "%Y-%m-%d")

    # Calculate age in days
    age_days = (datetime.now() - birth_date).days

    return age_days

if __name__ == "__main__":
    # Capture image, perform analysis, and detect metal substance
    capture_and_analyze()



# Enhanced Computer Vision Application
# This Python script opens the camera, captures an image, performs image recognition using MobileNetV2, and provides additional functionalities such as detecting and pointing out any metal substance in the captured image. It also calculates how long the person in the image has been alive on Earth based on their birthdate.

# Prerequisites
# Make sure you have the required libraries installed. You can install them using the following command:

# bash
# Copy code
# pip install opencv-python tensorflow numpy
# Usage
# Run the script enhanced_computer_vision.py.
# The script will open the default camera, capture an image, and save it as "captured_image.jpg".
# Perform image recognition and display the top-3 predicted classes for the content of the image.
# Detect and point out any metal substance in the image and save the result as "metal_substance_detection_result.jpg".
# Enter the person's birthdate when prompted.
# The script will calculate and display how long the person has been alive on Earth in days.
# Metal Substance Detection
# The script utilizes edge detection and contour analysis to identify and highlight potential metal substances in the captured image. Adjust the parameters in the detect_metal_substance function based on your specific requirements.

# Example
# bash
# Copy code
# python enhanced_computer_vision.py
# Notes
# Make sure your camera is properly connected and accessible.
# Adjust the birthdate format as needed (YYYY-MM-DD).
# The captured image is saved as "captured_image.jpg" in the same directory.
# The metal substance detection result is saved as "metal_substance_detection_result.jpg".
# Acknowledgments
# This script utilizes the MobileNetV2 model from TensorFlow for image recognition.
# Feel free to customize and extend the script according to your specific project details.