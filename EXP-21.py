
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the input image
image = cv2.imread("input.jpg")

if image is None:
    raise FileNotFoundError("Please provide a valid image path.")

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Create a binary image
_, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

# Step 4: Create a structuring element
kernel = np.ones((5, 5), np.uint8)

# Step 5: Apply Opening operation
opening = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel
)

# Step 6: Display the images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(binary, cmap="gray")
plt.title("Original Binary Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(opening, cmap="gray")
plt.title("Opening Operation")
plt.axis("off")

plt.tight_layout()
plt.show()

# Step 7: Save the result
cv2.imwrite("opening_output.jpg", opening)

print("Opening morphological operation completed.")
print("Output saved as opening_output.jpg")
