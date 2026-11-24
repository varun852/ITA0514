
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Step 1: Load a watch-trained YOLO model
# Replace this with the path to your trained model.
model = YOLO("watch_model.pt")

# Step 2: Read the input image
image = cv2.imread("watch.jpg")

if image is None:
    raise FileNotFoundError("Please provide a valid image path.")

# Step 3: Perform object detection
results = model.predict(image, conf=0.25)

# Step 4: Draw bounding boxes and labels
output = image.copy()
watch_detected = False

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])

        if "watch" in label.lower():
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            cv2.rectangle(
                output, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            text = f"{label}: {confidence:.2f}"
            cv2.putText(
                output, text, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )

            watch_detected = True

# Step 5: Display the output
if watch_detected:
    print("Watch recognized successfully.")
else:
    print("No watch detected.")

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Watch Object Recognition")
plt.axis("off")
plt.show()

# Step 6: Save the output
cv2.imwrite("watch_detection_output.jpg", output)
print("Output saved as watch_detection_output.jpg")
