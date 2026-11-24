
import cv2
from ultralytics import YOLO

# Step 1: Load the pretrained YOLO model
model = YOLO("yolov8n.pt")

# COCO class IDs for vehicles
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Step 2: Open the input video
cap = cv2.VideoCapture("traffic.mp4")

if not cap.isOpened():
    raise FileNotFoundError("Cannot open traffic.mp4")

# Step 3: Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

# Step 4: Create output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "vehicle_detection_output.mp4",
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video")

frame_count = 0

# Step 5: Process each video frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Step 6: Detect objects in the current frame
    results = model.predict(frame, conf=0.35, verbose=False)

    vehicle_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            # Step 7: Keep only vehicle classes
            if class_id in vehicle_classes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])
                label = vehicle_classes[class_id]

                vehicle_count += 1

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display label and confidence
                text = f"{label}: {confidence:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Step 8: Display vehicle count
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Step 9: Display and save frame
    cv2.imshow("Vehicle Detection", frame)
    out.write(frame)

    # Press q to stop early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Step 10: Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Vehicle detection completed.")
print("Total frames processed:", frame_count)
print("Output: vehicle_detection_output.mp4")
