
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the input image
image = cv2.imread("input.jpg")

if image is None:
    raise FileNotFoundError("Please provide a valid image path.")

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Define the threshold value
threshold_value = 127

# Step 4: Apply threshold-based segmentation
ret, segmented = cv2.threshold(
    gray,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

# Step 5: Display the images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(segmented, cmap="gray")
plt.title("Segmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Step 6: Save the segmented image
cv2.imwrite("segmented_output.jpg", segmented)

print("Image segmentation completed successfully.")
print("Threshold value:", ret)
print("Output saved as segmented_output.jpg")
