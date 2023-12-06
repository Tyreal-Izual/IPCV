import cv2
import os
import numpy as np
import sys

# The following part is using Lab3 code that I have implemented, to achieve hough transform

# Sobel's kernels
# for derivative in the x direction
derivativeInX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# for derivative in the y direction
derivativeInY = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])


def convolutionx(image, derivativeInX):
    i_height, i_width = image.shape
    k_height, k_width = derivativeInX.shape

    # Calculate half sizes for kernels with even dimensions
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Pad the image with zeros around the borders
    padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_REPLICATE)

    output = np.zeros_like(image, dtype=float)

    for y in range(i_height):
        for x in range(i_width):
            convolution_sum = (derivativeInX * padded_image[y:y+k_height, x:x+k_width]).sum()
            # output[y, x] = min(max(convolution_sum, 0), 255)
            output[y, x] = convolution_sum

    return output


def convolutiony(image, derivativeInY):
    i_height, i_width = image.shape
    k_height, k_width = derivativeInY.shape

    # Calculate half sizes for kernels with even dimensions
    pad_height = k_height // 2
    pad_width = k_width // 2

    # Pad the image with zeros around the borders
    padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_REPLICATE)

    output = np.zeros_like(image, dtype=float)

    for y in range(i_height):
        for x in range(i_width):
            convolution_sum = (derivativeInY * padded_image[y:y+k_height, x:x+k_width]).sum()
            # output[y, x] = min(max(convolution_sum, 0), 255)
            output[y, x] = convolution_sum

    return output


# gradient magnitude
def gradient_magnitude(dx, dy):
    return np.sqrt(dx**2 + dy**2)


# gradients direction
def gradient_direction(dx, dy):
    return np.arctan2(dy, dx)


def threshold_image(image, threshold):
    # Normalize the image to the range [0, 255]
    norm_image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)

    _, binary_image = cv2.threshold(norm_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def hough(binary_image, gradient_direction, min_radius, max_radius, threshold):
    # Initialize the Hough space
    rows, cols = binary_image.shape
    hough_space = np.zeros((rows, cols, max_radius - min_radius))

    edge_pixels = np.argwhere(binary_image)
    for y, x in edge_pixels:
        for r in range(min_radius, max_radius):
            # For positive gradient direction
            x0_pos = int(x - r * np.cos(gradient_direction[y, x]))
            y0_pos = int(y - r * np.sin(gradient_direction[y, x]))

            # For negative gradient direction
            x0_neg = int(x + r * np.cos(gradient_direction[y, x]))
            y0_neg = int(y + r * np.sin(gradient_direction[y, x]))

            # Accumulate votes in Hough space for positive direction
            if (0 <= x0_pos < cols) and (0 <= y0_pos < rows):
                hough_space[y0_pos, x0_pos, r - min_radius] += 1

            # Accumulate votes in Hough space for negative direction
            if (0 <= x0_neg < cols) and (0 <= y0_neg < rows):
                hough_space[y0_neg, x0_neg, r - min_radius] += 1

    return hough_space


def hough_transform_circle(image, derivativeInX, derivativeInY, threshold, min_radius, max_radius):
    # Perform Sobel edge detection
    convolvedx_image = convolutionx(image, derivativeInX)
    convolvedy_image = convolutiony(image, derivativeInY)

    magnitude_image = gradient_magnitude(convolvedx_image, convolvedy_image)
    direction_image = gradient_direction(convolvedx_image, convolvedy_image)

    # Thresholding the magnitude image
    binary_image = threshold_image(magnitude_image, threshold)

    # Perform Hough Circle Transform using the binary image and direction image
    hough_space = hough(binary_image, direction_image, min_radius, max_radius, threshold)

    # Extract circle centers from Hough space
    # Adjust the threshold here if needed
    circle_centers = np.argwhere(hough_space > threshold)

    # Convert the circle centers to the format (x, y, r)
    circle_centers = [(y, x, r + min_radius) for y, x, r in circle_centers]

    return circle_centers

# update for single parameter read in terminal
def read_ground_truth(file_path):
    ground_truth = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 5:
                image_name, x, y, w, h = parts
                # Remove the extension from the image name if it exists
                image_name = image_name.split('.')[0]
                if image_name not in ground_truth:
                    ground_truth[image_name] = []
                ground_truth[image_name].append((int(float(x)), int(float(y)), int(float(w)), int(float(h))))
    return ground_truth

# IoU == Intersection over Union
def intersection_over_union(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    intersection = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # area of union
    union = boxAArea + boxBArea - intersection

    # Compute the IoU
    iou = intersection / float(union)
    return iou


# detections and calculate TPR and F1-score
def evaluate_detections(detections, ground_truth):
    TP = 0  # True Positives
    FP = 0  # False Positives

    # Check each detection against ground truth data
    for det in detections:
        match_found = False
        for gt in ground_truth:
            if intersection_over_union(det, gt) > 0.2:  # Adjusted threshold to 0.2
                TP += 1  # Increment for each detection that matches
                match_found = True
                break  # Stop checking once a match is found

        if not match_found:
            FP += 1  # Increment false positives

    FN = len(ground_truth) - TP  # False Negatives
    if FN < 0:
        FN = 0  # Ensure FN is not negative

    # Calculate TPR and F1-score
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return TPR, F1



# Combines evidence from Viola-Jones and Hough Transform detections
def combine_evidence(vj_detections, hough_detections):
    combined_detections = []

    # Check each Hough detection for overlap with VJ detections
    for (y0, x0, r) in hough_detections:
        hough_circle = (x0 - r, y0 - r, 2 * r, 2 * r)  # Convert circle to bounding box
        overlap_with_vj = False

        for (x, y, w, h) in vj_detections:
            # Check if the center of the Hough circle falls within the VJ detection
            if x <= x0 < x + w and y <= y0 < y + h:
                overlap_with_vj = True
                break

        if overlap_with_vj:
            combined_detections.append((x, y, w, h))  # Add overlapping Hough detections as VJ detection
        else:
            combined_detections.append(hough_circle)  # Add non-overlapping Hough detections

    return combined_detections


# Perform detection and display the results
def detect_and_display(image, ground_truth, model, image_name, output_path):
    # Convert to grayscale and normalize lighting
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    # Detect dartboards Dartboard Detection
    dartboards_vj = model.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20), maxSize=(230,230))

    # Hough Circle Transform detection
    circle_centers = hough_transform_circle(gray_image, derivativeInX, derivativeInY, 15, 30, 150)

    # Combine evidence from both methods
    combined_detections = combine_evidence(dartboards_vj, circle_centers)

    # # Draw rectangles around the detected dartboards (in Yellow)
    # for (x, y, w, h) in dartboards_vj:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Yellow

    # Draw ground truth bounding boxes (in red)
    for (x, y, w, h) in ground_truth:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # # # Draw circles for Hough detections (in Blue)
    # for (y0, x0, r) in circle_centers:
    #     cv2.circle(image, (x0, y0), r, (255, 0, 0), 2)  # Blue

    # Draw rectangles around combined detections (in green)
    for (x, y, w, h) in combined_detections:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Evaluate detections and get performance metrics
    TPR, F1 = evaluate_detections(combined_detections, ground_truth)

    # Print the TPR and F1 score for the current image
    print(f'TPR: {TPR:.2f}, F1 Score: {F1:.2f}')

    # Display the image
    cv2.imshow('Dartboards Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save images
    save_path = os.path.join(output_path, image_name)
    cv2.imwrite(save_path, image)

    return TPR, F1


if __name__ == '__main__':
    # Check if the image filename is provided as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python DartboardDetector.py [image_filename]")
        sys.exit(1)

    # Load the trained model
    model = cv2.CascadeClassifier('Dartboardcascade/cascade.xml')

    # Get the image filename from the command line argument
    image_filename = sys.argv[1]
    image_path = os.path.join('Dartboard/', image_filename)

    # Output path for the detected image
    output_path = 'output/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image {image_filename} not found.")
        sys.exit(1)

    # Path to the ground truth file
    groundtruth_path = 'groundtruth.txt'
    ground_truth_data = read_ground_truth(groundtruth_path)

    # Extract the filename without extension for ground truth matching
    image_name_without_extension = os.path.splitext(image_filename)[0]

    # Extract the ground truth bounding boxes for the current image
    gt_bounding_boxes = ground_truth_data.get(image_name_without_extension, [])


    # Detect dartboards and display the results
    TPR, F1 = detect_and_display(image, gt_bounding_boxes, model, image_filename, output_path)

    # Save the detected image with bounding boxes
    detected_image_path = os.path.join('output/', 'detected.jpg')
    cv2.imwrite(detected_image_path, image)
    print(f"Detected dartboards saved to: {detected_image_path}")