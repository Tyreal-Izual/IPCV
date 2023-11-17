import cv2
import os


# read ground truth bounding boxes from file
def read_ground_truth(file_path):
    ground_truth = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 5:
                image_name, x, y, w, h = parts
                if image_name not in ground_truth:
                    ground_truth[image_name] = []
                ground_truth[image_name].append((int(float(x)), int(float(y)), int(float(w)), int(float(h))))
    return ground_truth


# perform detection and display the results
def detectAndDisplay(image, ground_truth, model):
    # Convert to grayscale and normalize lighting
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    # Detect dartboards
    dartboards = model.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected dartboards (in blue)
    for (x, y, w, h) in dartboards:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Draw ground truth bounding boxes (in red)
    for (x, y, w, h) in ground_truth:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Display the image
    cv2.imshow('Dartboards Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load the trained model
    model = cv2.CascadeClassifier('Dartboardcascade/cascade.xml')

    # Path of the image directory
    path = 'Dartboard/'

    # Path to the ground truth file
    groundtruth_path = 'groundtruth.txt'
    ground_truth_data = read_ground_truth(groundtruth_path)

    # Loop through the images
    for i in range(16):
        # Full path to the image
        image_path = os.path.join(path, f'dart{i}.jpg')

        # Read the image
        image = cv2.imread(image_path)

        # Extract the ground truth bounding boxes for the current image
        gt_bounding_boxes = ground_truth_data.get(f'dart{i}', [])

        # Use the function to detect and display dartboards
        detectAndDisplay(image, gt_bounding_boxes, model)
