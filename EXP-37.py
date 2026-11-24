
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

# Define the green foreground color range
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create a mask for green foreground pixels
foreground_mask = cv2.inRange(
    hsv, lower_green, upper_green
)

# Extract the foreground from the original image
foreground = cv2.bitwise_and(
    image, image, mask=foreground_mask
)

# Convert images to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
foreground_rgb = cv2.cvtColor(
    foreground, cv2.COLOR_BGR2RGB
)

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(foreground_mask, cmap="gray")
plt.title("Foreground Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(foreground_rgb)
plt.title("Extracted Foreground")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save output images
cv2.imwrite("foreground_mask.png", foreground_mask)
cv2.imwrite("foreground_result.png", foreground)

print("Foreground extraction completed.")
