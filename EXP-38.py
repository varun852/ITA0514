
import cv2
import matplotlib.pyplot as plt

# Load the input image
image = cv2.imread("input.jpg")

if image is None:
    print("Error: Image not found. Check the file path.")
    raise SystemExit

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Could not load face detection model.")
    raise SystemExit

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

# Count the faces
face_count = len(faces)

# Display the face count
cv2.putText(
    image,
    f"Faces: {face_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

print("Number of faces detected:", face_count)

# Display the output image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title(f"Number of Faces: {face_count}")
plt.axis("off")
plt.show()

# Save the output image
cv2.imwrite("face_count_output.jpg", image)
