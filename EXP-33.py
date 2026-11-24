
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a blank white image
image = np.ones((400, 600, 3), dtype=np.uint8) * 255

# Step 2: Define rectangle coordinates
start_point = (150, 100)
end_point = (450, 300)

# Step 3: Define color and thickness
color = (255, 0, 0)  # Blue in BGR format
thickness = 3

# Step 4: Draw the rectangle
cv2.rectangle(
    image,
    start_point,
    end_point,
    color,
    thickness
)

# Step 5: Display the image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Rectangle Using OpenCV")
plt.axis("off")
plt.show()

# Step 6: Save the output
cv2.imwrite("rectangle_output.png", image)

print("Rectangle created successfully!")
print("Output saved as rectangle_output.png")
