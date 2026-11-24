
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a blank white image
image = np.ones((400, 600, 3), dtype=np.uint8) * 255

# Step 2: Define circle properties
center = (300, 200)
radius = 100
color = (0, 0, 255)  # Red in BGR format
thickness = 3

# Step 3: Draw the circle
cv2.circle(
    image,
    center,
    radius,
    color,
    thickness
)

# Step 4: Display the image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Circle Using OpenCV")
plt.axis("off")
plt.show()

# Step 5: Save the output
cv2.imwrite("circle_output.png", image)

print("Circle created successfully!")
print("Output saved as circle_output.png")
