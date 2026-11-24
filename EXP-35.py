
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a blank white image
image = np.ones((400, 700, 3), dtype=np.uint8) * 255

# Step 2: Get text from the user
text = input("Enter the text to display: ")

# Step 3: Define text properties
position = (50, 200)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
color = (255, 0, 0)  # Blue in BGR format
thickness = 2

# Step 4: Display text on the image
cv2.putText(
    image,
    text,
    position,
    font,
    font_scale,
    color,
    thickness,
    cv2.LINE_AA
)

# Step 5: Display the image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(9, 5))
plt.imshow(image_rgb)
plt.title("User-Entered Text Using OpenCV")
plt.axis("off")
plt.show()

# Step 6: Save the output image
cv2.imwrite("text_output.png", image)

print("Text displayed successfully!")
print("Output saved as text_output.png")
