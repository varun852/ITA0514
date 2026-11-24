
import cv2
import matplotlib.pyplot as plt

# Step 1: Load Haar Cascade classifiers
face_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

smile_path = (
    cv2.data.haarcascades
    + "haarcascade_smile.xml"
)

face_cascade = cv2.CascadeClassifier(face_path)
smile_cascade = cv2.CascadeClassifier(smile_path)

if face_cascade.empty() or smile_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade classifiers.")

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
    minSize=(50, 50)
)

output = image.copy()
smile_count = 0

# Step 5: Detect smiles inside each face
for (x, y, w, h) in faces:
    # Draw rectangle around the face
    cv2.rectangle(
        output, (x, y), (x + w, y + h),
        (255, 0, 0), 2
    )

    # Consider the lower half of the face
    roi_gray = gray[
        y + h // 2:y + h,
        x:x + w
    ]

    roi_color = output[
        y + h // 2:y + h,
        x:x + w
    ]

    smiles = smile_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.7,
        minNeighbors=22,
        minSize=(25, 15)
    )

    # Step 6: Draw rectangles around detected smiles
    if len(smiles) > 0:
        # Use the largest detected region to reduce duplicates
        sx, sy, sw, sh = max(
            smiles, key=lambda r: r[2] * r[3]
        )

        cv2.rectangle(
            roi_color,
            (sx, sy),
            (sx + sw, sy + sh),
            (0, 255, 0), 2
        )

        cv2.putText(
            output, "Smile detected",
            (x, max(y + h - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

        smile_count += 1

# Step 7: Display the output image
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(output_rgb)
plt.title(f"Smile Detection - Smiling Faces: {smile_count}")
plt.axis("off")
plt.show()

# Step 8: Save the output
cv2.imwrite("smile_detection_output.jpg", output)

print("Faces detected:", len(faces))
print("Faces with detected smiles:", smile_count)
print("Output saved as smile_detection_output.jpg")
