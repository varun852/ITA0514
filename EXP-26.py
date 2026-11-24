
import cv2

def reverse_video(input_path, output_path):
    # Step 1: Open the input video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open input video.")
        return

    # Step 2: Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        print("Error: Invalid video properties.")
        cap.release()
        return

    # Step 3: Read and store all frames
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if not frames:
        print("Error: No frames found.")
        return

    # Step 4: Create the output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        print("Error: Cannot create output video.")
        return

    # Step 5: Write frames in reverse order
    for frame in reversed(frames):
        writer.write(frame)

    # Step 6: Release the writer
    writer.release()

    print("Reverse video created successfully!")
    print("Output file:", output_path)
    print("Total frames:", len(frames))


# Function call
reverse_video("input.mp4", "reverse_output.mp4")
