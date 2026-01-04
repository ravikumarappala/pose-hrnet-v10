import cv2

DEVICE_PATH = "/dev/v4l/by-path/platform-3610000.xhci-usb-0:4.3:1.0-video-index0"

# Use V4L2 backend explicitly for Jetson
cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)

# Optional: set a decent resolution to get full body
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"❌ Could not open camera at {DEVICE_PATH}")
    exit(1)

print("✅ Camera opened.")
print("👉 Stand ~6 ft away so your full body is visible.")
print("   Press 'c' to capture, 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Live - press 'c' to capture, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        save_path = "person.jpg"
        cv2.imwrite(save_path, frame)
        print(f"✅ Saved image as {save_path}")
        break
    elif key == ord('q'):
        print("❌ Quit without saving")
        break

cap.release()
cv2.destroyAllWindows()
