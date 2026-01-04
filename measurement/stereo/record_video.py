import cv2
import os
from datetime import datetime

# Output directory
OUTPUT_DIR = "recorded_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_DIR, f"stereo_video_{timestamp}.mp4")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open camera index 0")

# Get camera properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

print(f"Camera opened: {frame_width}x{frame_height} @ {fps} FPS")
print(f"Recording to: {output_file}")
print("Press ESC to stop recording...")

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("ERROR: Could not create video writer")
    cap.release()
    exit(1)

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Write frame to video file
        out.write(frame)
        frame_count += 1

        # Display the frame
        cv2.imshow("Recording...", frame)

        # Press ESC to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\nRecording interrupted by user")

finally:
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n--- Recording Complete ---")
    print(f"Frames recorded: {frame_count}")
    print(f"Saved to: {output_file}")
