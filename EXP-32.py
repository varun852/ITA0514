
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Get user-defined image dimensions
width = int(input("Enter image width: "))
height = int(input("Enter image height: "))

# Step 2: Validate dimensions
if width <= 0 or height <= 0:
    raise ValueError("Width and height must be positive.")

# Step 3: Create a white image
image = np.full((height, width, 3), 255, dtype=np.uint8)

# Step 4: Define box size
box_size = min(width, height) // 5

if box_size < 1:
    raise ValueError("Image dimensions are too small.")

# Step 5: Draw colored boxes at each corner
# Top-left: Black
cv2.rectangle(
    image, (0, 0), (box_size - 1, box_size - 1),
    (0, 0, 0), -1
)

# Top-right: Blue
cv2.rectangle(
    image, (width - box_size, 0),
    (width - 1, box_size - 1),
    (255, 0, 0), -1
)

# Bottom-left: Green
cv2.rectangle(
    image, (0, height - box_size),
    (box_size - 1, height - 1),
    (0, 255, 0), -1
)

# Bottom-right: Red
cv2.rectangle(
    image, (width - box_size, height - box_size),
    (width - 1, height - 1),
    (0, 0, 255), -1
)

# Step 6: Display the output
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title("White Image with Four Colored Corner Boxes")
plt.axis("off")
plt.show()

# Step 7: Save the output image
cv2.imwrite("colored_corner_boxes.png", image)

print("Image created successfully!")
print("Output saved as colored_corner_boxes.png")
