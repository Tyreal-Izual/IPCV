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
    detected_indices = set()

    # Check each detection against ground truth data
    for i, det in enumerate(detections):
        match_found = False
        for j, gt in enumerate(ground_truth):
            if intersection_over_union(det, gt) > 0.5:  # threshold = 0.5
                if j not in detected_indices:
                    TP += 1
                    detected_indices.add(j)
                    match_found = True
                    break
        if not match_found:
            FP += 1

    FN = len(ground_truth) - TP  # False Negatives

    # Calculate TPR and F1-score
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return TPR, F1


# perform detection and display the results
def detect_and_display(image, ground_truth, model, image_name, output_path):
    # Convert to grayscale and normalize lighting
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)

    # Detect dartboards
    dartboards = model.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20), maxSize=(230,230))

    # Draw rectangles around the detected dartboards (in blue)
    for (x, y, w, h) in dartboards:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw ground truth bounding boxes (in red)
    for (x, y, w, h) in ground_truth:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Evaluate detections and get performance metrics
    TPR, F1 = evaluate_detections(dartboards, ground_truth)

    # Print the TPR and F1 score for the current image
    print(f'TPR: {TPR:.2f}, F1 Score: {F1:.2f}')

    # Display the image
    cv2.imshow('Dartboards Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save images
    save_path = os.path.join(output_path, image_name)
    cv2.imwrite(save_path, image)

    return evaluate_detections(dartboards, ground_truth)


if __name__ == '__main__':
    # Load the trained model
    model = cv2.CascadeClassifier('Dartboardcascade/cascade.xml')

    # Path of the image directory
    path = 'Dartboard/'
    output_path = 'output/'
    # Make sure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Path to the ground truth file
    groundtruth_path = 'groundtruth.txt'
    ground_truth_data = read_ground_truth(groundtruth_path)

    # Loop through the images and evaluate performance
    all_tpr = []
    all_f1 = []
    for i in range(16):
        # Full path to the image
        image_path = os.path.join(path, f'dart{i}.jpg')
        image_name = f'dart{i}.jpg'

        # Read the image
        image = cv2.imread(image_path)

        # Extract the ground truth bounding boxes for the current image
        gt_bounding_boxes = ground_truth_data.get(f'dart{i}', [])

        # # Use the function to detect and display dartboards
        # detect_and_display(image, gt_bounding_boxes, model)

        # Detect dartboards and display the results
        TPR, F1 = detect_and_display(image, gt_bounding_boxes, model, image_name, output_path)

        # Append the performance metrics
        all_tpr.append(TPR)
        all_f1.append(F1)

    # Output the average TPR and F1 score
    average_tpr = sum(all_tpr) / len(all_tpr)
    average_f1 = sum(all_f1) / len(all_f1)
    print(f'Average TPR: {average_tpr:.2f}')
    print(f'Average F1 Score: {average_f1:.2f}')