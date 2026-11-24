
import cv2
import pytesseract

# Windows users: uncomment and update this path if necessary
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

# Open the input video
video = cv2.VideoCapture("input.mp4")

if not video.isOpened():
    print("Error: Cannot open video file.")
    raise SystemExit

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

frame_interval = max(1, int(fps))  # Process one frame per second
frame_number = 0
extracted_text = []

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Process one frame per second
    if frame_number % frame_interval == 0:
        timestamp = frame_number / fps

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Improve text visibility using thresholding
        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Extract text using OCR
        text = pytesseract.image_to_string(
            thresh,
            config="--psm 6"
        ).strip()

        if text:
            result = f"Time {timestamp:.2f}s: {text}"
            extracted_text.append(result)
            print(result)

    frame_number += 1

video.release()

# Save extracted text to a file
with open("extracted_text.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(extracted_text))

print("\nText extraction completed.")
print("Results saved to extracted_text.txt")
