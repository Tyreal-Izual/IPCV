# Read me for Dartboard Detector

## Overview
The Dartboard Detector is a Python-based application for detecting dartboards in images. It uses a combination of Sobel edge detection, Hough Circle Transform, and Viola-Jones method for accurate detection. The program takes an image as input, processes it to detect dartboards, and outputs an image with highlighted dartboards using bounding boxes.

## Requirements
- Python 3.8+
- OpenCV library (cv2)
- NumPy

## Installation
### For MVB-2.11/1.15 Lab Machine in University of Bristol
1. Ensure "Anaconda" is ready to use, you may need to: "module load anaconda/3-2023".

2. Create a virtual environment using conda: "conda create -n ipcv python=3.8".
3. Activate the virtual environment: "conda activate ipcv".
4. Install OpenCV packages: "pip install numpy opencv-python numpy".

## Usage
1. Place the image you want to process in the ' Dartboard/ ' directory.
2. Navigate to the directory containing the Dartboard Detector script in the command line.
3. Run the script with the image filename as a parameter. For example:
   "python DartboardDetector.py dart0.jpg".

   Replace 'dart0.jpg' with the name of your image file. Ensure the image file is located in the 'Dartboard/' directory.

## Output
The program will:
- Process the specified image and output an image with detected dartboards highlighted by bounding boxes.
- Print the True Positive Rate (TPR) and F1 Score for the detection in the console.
- Save the processed image in the 'output/' directory as 'detected.jpg'.

## Ground Truth Data
The program utilizes ground truth data for evaluation purposes. The ground truth data should be
provided in a file named 'groundtruth.txt' located in the same directory as the script.
The format for each line in this file is: "image_name, x, y, width, height".
Example:
"dart0, 450.0000, 20.0000, 150.0000, 170.0000".

## Notes
- The Dartboard Detector uses a pre-trained model for the Viola-Jones method, located in 'Dartboardcascade/cascade.xml'.
- The script is configured to work with images in the 'Dartboard/' directory. Ensure your images are placed.
  in this directory before running the script.
- For more information, please refer to 'README.md' and 'Instruction.md'.

---
*This README is part of the Shape Detection, developed as part of the coursework in COMS30076 at the University of Bristol.*
