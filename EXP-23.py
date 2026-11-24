
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the input image
image = cv2.imread("input.jpg")

if image is None:
    raise FileNotFoundError("Please provide a valid image path.")

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Create a structuring element
kernel = np.ones((5, 5), np.uint8)

# Step 4: Apply Top Hat operation
tophat = cv2.morphologyEx(
    gray, cv2.MORPH_TOPHAT, kernel
)

# Step 5: Display the images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(tophat, cmap="gray")
plt.title("Top Hat Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# Step 6: Save the output
cv2.imwrite("tophat_output.jpg", tophat)

print("Top Hat morphological operation completed.")
print("Output saved as tophat_output.jpg")
