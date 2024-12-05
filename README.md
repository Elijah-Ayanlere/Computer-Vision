**Continuous Analysis Computer Vision**

This Python script opens the default camera and continuously analyzes the surroundings. It performs image recognition using the MobileNetV2 model and detects metal substances in real-time. The camera preview, along with image recognition results and metal substance detection, is displayed on the screen.

**Prerequisites**

Make sure you have the required libraries installed. You can install them using the following command:
"pip install opencv-python tensorflow numpy"

**Usage**

Run the script computer_vision.py.
The script will open the default camera and start analyzing the surroundings.
Image recognition results and metal substance detection will be displayed in real-time.
Press the 'q' key to exit the program.
Image Recognition
The script utilizes the MobileNetV2 model from TensorFlow for image recognition. It continuously analyzes the frames from the camera and prints the top-3 predicted classes for the content of the image.

**Metal Substance Detection**

Metal substance detection is performed using edge detection and contour analysis. The script highlights potential metal substances in the live camera preview.

**Notes**

Make sure your camera is properly connected and accessible.
Press the 'q' key to exit the program.
Feel free to customize and extend the script according to your specific project details.
