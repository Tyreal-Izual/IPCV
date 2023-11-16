import cv2
import os

# Load the trained model
model = cv2.CascadeClassifier('Dartboardcascade/cascade.xml')


# path of the image:
path = 'Dartboard/'

# Loop through the images
for i in range(16):
    # full path to the image
    image_path = os.path.join(path, f'dart{i}.jpg')

    # read the image
    image = cv2.imread(image_path)

    # convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect dartboards
    dartboards = model.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the dartboards
    for (x, y, w, h) in dartboards:
        start_point = (x, y)
        end_point = (x+w, y+h)
        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(image, start_point, end_point, color, thickness)

    # Display the image
    cv2.imshow(f'Dartboards Detected in dart{i}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
