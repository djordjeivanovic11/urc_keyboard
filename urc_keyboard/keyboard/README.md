# Keyboard Detection and 3D Mapping for URC Missions

This project is designed to detect letters on a keyboard and map their positions to 3D coordinates for use in the **University Rover Challenge (URC)**. The rover's camera captures a top-down view of a keyboard, detects specific letters, and translates their positions into 3D coordinates for precise navigation and control.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Description](#description)
4. [Configuration](#configuration)
5. [How It Works](#how-it-works)
6. [Troubleshooting](#troubleshooting)
7. [Acknowledgements](#acknowledgements)

## Installation
1. **Clone this repository:**
   ```bash
   git clone <>
   cd <repository_name>
   ```

2. **Install the required Python packages:**
   ```bash
   pip install opencv-python-headless pytesseract numpy
   ```

3. **Install Tesseract OCR:**
   - **macOS (using Homebrew):**
     ```bash
     brew install tesseract
     ```
   - **Windows:** Download from [Tesseract's official site](https://github.com/tesseract-ocr/tesseract) and update the Tesseract path in the code.

## Usage
1. **Run the program:**
   ```bash
   python main.py or python3 main.py
   ```

2. The program will:
   - Open the default webcam (or the specified camera index). Put the keyboard in front, the camera will seem frozen.
   - Continuously capture frames, detect the keyboard, and identify specific passcode letters.
   - Calculate the 3D coordinates of the detected letters and display the annotated results. This however is only a simulation

3. **Press 'q'** to exit the program manually.

## Description
This program captures real-time video input from a webcam, identifies the keyboard, and detects specified letters on it. The detected letters are mapped to 3D coordinates, allowing the rover to navigate and interact based on these coordinates. The workflow involves:
1. **Capturing frames** from the camera.
2. **Detecting the keyboard** using contour detection.
3. **Applying a perspective transform** for a top-down view.
4. **Using OCR (Tesseract)** to identify specific passcode letters.
5. **Mapping the detected 2D coordinates** to 3D coordinates for real-world positioning.

## Configuration
### Camera Settings
- **Camera Index:** The camera index can be set in the `connect_to_webcam()` function (default is 0).
- **Resolution and Frame Rate:** Adjust the `resolution` and `fps` parameters to match your camera’s capabilities.

### Passcode Configuration
- **Passcode Letters:** Update the `passcode` variable in the `main()` function to specify which letters the program should detect (e.g., `'URCSOFTWARE'`).

### Tesseract OCR Settings
- **Update Tesseract Path:** Make sure the Tesseract installation path is correctly set in the code based on your operating system:
  - For **macOS**, update the path to `/opt/homebrew/bin/tesseract`.
  - For **Windows**, set it to the installed path, e.g., `'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'`.

### 3D Mapping Settings
- **Field of View (FOV):** Adjust the `fov_horizontal` and `fov_vertical` parameters in the `map_to_3d()` function based on your camera specifications.
- **Reference Distance:** Set the `reference_distance` parameter to specify the expected distance from the camera to the keyboard.

## How It Works
### Main Functions
1. **`connect_to_webcam()`:** Connects to the webcam and sets resolution and frame rate.
2. **`capture_frame()`:** Captures frames from the webcam for processing.
3. **`detect_keyboard()`:** Detects the keyboard in the captured frame using contour detection.
4. **`four_point_transform()`:** Applies a perspective transform to get a top-down view of the keyboard.
5. **`build_coordinate_map()`:** Uses Tesseract OCR to identify specified letters and map their coordinates.
6. **`map_to_3d()`:** Converts 2D coordinates to 3D coordinates for navigation and interaction.

### Overall Workflow
- The program starts by capturing frames from the webcam.
- It detects the keyboard using contours and performs a perspective transform to obtain a clear, flat image.
- OCR is applied to the image to detect specified letters, mapping their coordinates in the 2D image.
- These 2D coordinates are then converted to 3D coordinates, helping the rover understand the real-world position of the letters.

## Troubleshooting
- **Camera not detected:**
  - Ensure the camera index is correct in the `connect_to_webcam()` function.
  - Check if the camera is connected and accessible.

- **No letters detected:**
  - Adjust the lighting and camera angle for better visibility.
  - Try modifying the threshold value or the morphological kernel size in the `build_coordinate_map()` function.
  - Make sure the specified passcode letters are correct.

- **Perspective transform failed:**
  - Ensure that the keyboard is fully visible in the frame.
  - Adjust the camera angle or distance to capture a clearer image of the keyboard.

- **Low accuracy in 3D mapping:**
  - Verify the camera’s FOV values and reference distance.
  - Adjust the scaling factors in the `map_to_3d()` function for better accuracy.

## What is used
- This project makes use of the [OpenCV](https://opencv.org/) library for computer vision tasks and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for optical character recognition.
- Inspired by real-world applications for autonomous robotics in the University Rover Challenge (URC).
