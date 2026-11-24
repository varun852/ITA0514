
import cv2

# Open the input video
video = cv2.VideoCapture("input.mp4")

if not video.isOpened():
    print("Error: Cannot open video file.")
    raise SystemExit

# Store all video frames
frames = []

while True:
    ret, frame = video.read()

    if not ret:
        break

    frames.append(frame)

video.release()

if len(frames) == 0:
    print("Error: No frames found in the video.")
    raise SystemExit

print("Total frames:", len(frames))
print("Playing video in reverse slow motion...")
print("Press 'q' to stop playback.")

# Reverse playback with slow motion
# 100 ms delay gives approximately 10 frames per second
delay = 100

for frame in reversed(frames):
    cv2.imshow("Reverse Slow Motion Video", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()

print("Reverse slow-motion playback completed.")
