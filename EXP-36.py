
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
image = cv2.imread("input.jpg")

if image is None:
    print("Error: Image not found. Check the file path.")
    raise SystemExit

# Convert BGR image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define green color range in HSV
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Detect green background pixels
background_mask = cv2.inRange(
    hsv, lower_green, upper_green
)

# Invert mask to select foreground pixels
foreground_mask = cv2.bitwise_not(background_mask)

# Extract foreground from the original image
foreground = cv2.bitwise_and(
    image, image, mask=foreground_mask
)

# Convert images from BGR to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(background_mask, cmap="gray")
plt.title("Background Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(foreground_rgb)
plt.title("Foreground Extracted")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save the results
cv2.imwrite("background_mask.png", background_mask)
cv2.imwrite("foreground_result.png", foreground)

print("Background subtraction completed.")
