
import cv2
import matplotlib.pyplot as plt

# Step 1: Load the face detection classifier
cascade_path = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade classifier.")

# Step 2: Read the input image
image = cv2.imread("input.jpg")

if image is None:
    raise FileNotFoundError("Please provide a valid image path.")

# Step 3: Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 4: Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Step 5: Draw rectangles around detected faces
output = image.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# Step 6: Display the output image
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(output_rgb)
plt.title(f"Face Detection - Faces Found: {len(faces)}")
plt.axis("off")
plt.show()

# Step 7: Save the output
cv2.imwrite("face_detection_output.jpg", output)

print("Number of faces detected:", len(faces))
print("Output saved as face_detection_output.jpg")
