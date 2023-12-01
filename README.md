# Dartboard Detector

## Overview
The Dartboard Detector is a Python-based application for detecting dartboards in images. It uses a combination of Sobel edge detection, Hough Circle Transform, and Viola-Jones method for accurate detection. The program takes an image as input, processes it to detect dartboards, and outputs an image with highlighted dartboards using bounding boxes.

## Requirements
- Python 3.8+
- OpenCV library (cv2)
- NumPy

## Installation
### For the Linux-Based Machine
1. Ensure that ```Python 3.8+``` is installed on your system.
2. Install ```OpenCV``` and ```NumPy``` libraries. You can install them using pip:
   ```bash
   pip install opencv-python numpy
   ```
   or
   ```bash
   pip3 install opencv-python numpy
   ```
### For the Windows and Mac (Intel & ARM) - Based Machine
You may want to use ```Conda```
1. Open a terminal and create virtual environment: 
    ```bash 
    conda create -n ipcv python=3.8
    ```
2. Activate your environment: 
    ```bash
    conda activate myproject
    ```
3. Might need to update pip:
    ```bash
    pip install --upgrade pip
    ```
   or
   ```bash
   pip3 install --upgrade pip
    ```
4. Install OpenCV packages:
    ```bash
    pip install numpy opencv-python
    ```
    or
    ```bash
    conda install numpy 
    ```
   then
    ```bash 
   conda install -c menpo opencv
    ```

## Usage
1. Place the image you want to process in the `````'Dartboard/'````` directory.
2. Navigate to the directory containing the Dartboard Detector script in the command line.
3. Run the script with the image filename as a parameter. For example:
   ```bash
   python DartboardDetector.py dart0.jpg
   ```
   or
    ```bash
    python3 DartboardDetector.py dart0.jpg
   ```
   
   Replace `````'dart0.jpg'````` with the name of your image file. Ensure the image file is located in the `````'Dartboard/'````` directory.

## Output
The program will:
- Process the specified image and output an image with detected dartboards highlighted by bounding boxes.
- Print the True Positive Rate (TPR) and F1 Score for the detection in the console.
- Save the processed image in the `````'output/'````` directory as `````'detected.jpg'`````.

## Ground Truth Data
The program utilizes ground truth data for evaluation purposes. The ground truth data should be
provided in a file named `````'groundtruth.txt'````` located in the same directory as the script.
The format for each line in this file is:
```image_name, x, y, width, height```
Example:
```dart0, 450.0000, 20.0000, 150.0000, 170.0000```

## Notes
- The Dartboard Detector uses a pre-trained model for the Viola-Jones method, located in `````'Dartboardcascade/cascade.xml'`````.
- The script is configured to work with images in the `````'Dartboard/'````` directory. Ensure your images are placed
  in this directory before running the script.
- For more information, please refer to `````'Instruction.md'`````