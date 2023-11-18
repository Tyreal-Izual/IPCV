import cv2
import numpy as np

image = cv2.imread("Dartboard/dart0.jpg", cv2.IMREAD_GRAYSCALE)
color_image = cv2.imread("Dartboard/dart0.jpg", cv2.IMREAD_COLOR)

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


def display_hough_space(hough_space):
    hough_2d = np.sum(hough_space, axis=2)
    hough_2d = (255 * (hough_2d / np.max(hough_2d))).astype(np.uint8)  # Normalization to [0, 255]
    # hough_display = cv2.applyColorMap(hough_2d, cv2.COLORMAP_JET)  # Using a colormap for better visualization
    # cv2.imshow('Hough Space', hough_display)
    cv2.imshow('Hough Space', hough_2d)


def draw_circles(image, circle_centers, min_radius):
    output = image.copy()
    for y, x, r in circle_centers:
        cv2.circle(output, (x, y), r + min_radius, (0, 0, 255), 2)  # (0, 0, 255) represents red in BGR format
    cv2.imshow('Detected Circles', output)


def visualize_circle_centers(circle_centers):
    output = np.zeros_like(color_image)
    for y, x, r in circle_centers:
        cv2.circle(output, (x, y), 3, (0, 255, 0), -1)  # Drawing small green circles at each circle center
    return output


convolvedx_image = convolutionx(image, derivativeInX)
convolvedy_image = convolutiony(image, derivativeInY)

magnitude_image = gradient_magnitude(convolvedx_image, convolvedy_image)
direction_image = gradient_direction(convolvedx_image, convolvedy_image)


cv2.imshow('image', image)
cv2.imshow('convolvedx_image', convolvedx_image)
cv2.imshow('convolvedy_image', convolvedy_image)
# cv2.imshow('Magnitude', magnitude_image)

min_val, max_val = np.min(magnitude_image), np.max(magnitude_image)
normalized_magnitude = ((magnitude_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
cv2.imshow('Magnitude Image', normalized_magnitude)

min_val, max_val = np.min(direction_image), np.max(direction_image)
normalized_direction = ((direction_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
cv2.imshow('direction Image', normalized_direction)

# Thresholding
T = 50
binary_image = threshold_image(magnitude_image, T)

# Hough Transform
min_radius = 10
max_radius = 150
hough_threshold = 15
hough_space = hough(binary_image, direction_image, min_radius, max_radius, hough_threshold)
# Extract circle centers
circle_centers = np.argwhere(hough_space > hough_threshold)
print(circle_centers[:5])  # Print first 5 detected circle centers and radii

circle_centers_image = visualize_circle_centers(circle_centers)
cv2.imshow('Circle Centers', circle_centers_image)
# Display results
display_hough_space(hough_space)
draw_circles(color_image, circle_centers, min_radius)

cv2.waitKey(0)
cv2.destroyAllWindows()
