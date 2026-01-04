import cv2
import os
import time

# --- SETTINGS ---
VIDEO_PATH = "recorded_videos/stereo_video_20260104_095722.mp4"  # <-- set your file path here
SAVE_DIR_LEFT = "calib_left"
SAVE_DIR_RIGHT = "calib_right"
NUM_IMAGES = 40
# Your target display/save size (optional). If None, keep original size.
# TARGET_WIDTH = 2880
# TARGET_HEIGHT = 1620
TARGET_WIDTH = None
TARGET_HEIGHT = None


def open_video_file(video_path: str) -> cv2.VideoCapture:
    """Strictly open a video file and validate it produces frames."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    # Probe a few frames to ensure decode works
    ok = False
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            ok = True
            break
        time.sleep(0.05)

    if not ok:
        cap.release()
        raise RuntimeError(f"Video opened but no frames could be read: {video_path}")

    # Rewind so we start from the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def main():
    os.makedirs(SAVE_DIR_LEFT, exist_ok=True)
    os.makedirs(SAVE_DIR_RIGHT, exist_ok=True)

    cap = open_video_file(VIDEO_PATH)

    # Labels for window titles (purely cosmetic)
    dev_left = f"Video (Left): {os.path.basename(VIDEO_PATH)}"
    dev_right = f"Video (Right): {os.path.basename(VIDEO_PATH)}"

    # Honor file FPS for natural playback (recommended)
    file_fps = cap.get(cv2.CAP_PROP_FPS)
    if not file_fps or file_fps <= 1:
        file_fps = 30.0  # fallback
    delay_ms = max(int(1000 / file_fps), 1)

    # Read one frame to get source dimensions
    ret, frame0 = cap.read()
    if not ret or frame0 is None:
        cap.release()
        raise RuntimeError("Could not read first frame from video.")

    src_h, src_w = frame0.shape[:2]
    print(f"Opened video: {VIDEO_PATH}")
    print(f"Source resolution: {src_w}x{src_h} @ ~{file_fps:.2f} FPS")
    print("Will split each frame into left/right halves.")

    # Rewind again after probing first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    mode = input("Enter mode (m = manual, a = auto): ").strip().lower()
    if mode not in ["m", "a"]:
        print("Invalid mode! Use m or a.")
        cap.release()
        return

    interval = None
    if mode == "a":
        try:
            interval = float(input("Enter interval in seconds: "))
        except Exception:
            print("Invalid number.")
            cap.release()
            return
        print(f"Auto mode: capturing every {interval} sec")

    count = 0
    last_capture_time = time.time()

    print("\n--- Calibration Capture Started ---")
    if mode == "m":
        print("Press 's' to save pair, 'q' to quit.")
    else:
        print("Auto mode running... Press 'q' to stop early.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video (or read error).")
            break

        h, w = frame.shape[:2]
        mid = max(w // 2, 1)

        # Split single frame into two virtual views
        frame_left = frame[:, :mid]
        frame_right = frame[:, mid:] if mid < w else frame

        # # Optional: resize each half to a fixed target size
        # if TARGET_WIDTH and TARGET_HEIGHT:
        #     frame_left = cv2.resize(frame_left, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        #     frame_right = cv2.resize(frame_right, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Mirror (keep your original behavior)
        frame_left = cv2.flip(frame_left, 1)
        frame_right = cv2.flip(frame_right, 1)

        cv2.imshow(f"Left ({dev_left})", frame_left)
        cv2.imshow(f"Right ({dev_right})", frame_right)

        key = cv2.waitKey(delay_ms) & 0xFF

        if mode == "m":
            if key == ord("s"):
                if count < NUM_IMAGES:
                    left_path = os.path.join(SAVE_DIR_LEFT, f"left_{count}.png")
                    right_path = os.path.join(SAVE_DIR_RIGHT, f"right_{count}.png")
                    cv2.imwrite(left_path, frame_left)
                    cv2.imwrite(right_path, frame_right)

                    print(f"[{count}] Saved pair → {left_path}, {right_path}")
                    count += 1
                    time.sleep(0.25)  # debounce
                else:
                    print("Done saving all images.")
                    break
        else:
            now = time.time()
            if now - last_capture_time >= interval:
                if count < NUM_IMAGES:
                    left_path = os.path.join(SAVE_DIR_LEFT, f"left_{count}.png")
                    right_path = os.path.join(SAVE_DIR_RIGHT, f"right_{count}.png")
                    cv2.imwrite(left_path, frame_left)
                    cv2.imwrite(right_path, frame_right)

                    print(f"[{count}] Auto-saved pair → {left_path}, {right_path}")
                    count += 1
                    last_capture_time = now
                else:
                    print("Finished auto capture.")
                    break

        if key == ord("q"):
            print("Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
