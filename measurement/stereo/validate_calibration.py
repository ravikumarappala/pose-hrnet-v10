import cv2
import numpy as np
import glob
import os

# --- Checkerboard settings ---
CHECKERBOARD = (9, 6)   # internal corners (your board: 9x6 squares -> 8x5 corners)
SQUARE_SIZE = 27.0      # mm (adjust to your checkerboard square size)

# --- Prepare 3D object points ---
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world
imgpoints_left = []  # 2D points in left camera
imgpoints_right = []  # 2D points in right camera

left_images = sorted(glob.glob("calib_left/*.png"))
right_images = sorted(glob.glob("calib_right/*.png"))

print("Found left:", len(left_images), " right:", len(right_images))

if len(left_images) != len(right_images):
    print("Error: left and right image count mismatch!")
    print("Left files:", left_images)
    print("Right files:", right_images)
    exit(1)

# prepare debug folders
os.makedirs("debug_left", exist_ok=True)
os.makedirs("debug_right", exist_ok=True)

good_pairs = 0
for idx, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images), start=0):
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)
    if imgL is None or imgR is None:
        print(f"Cannot read pair: {left_img_path}, {right_img_path}")
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL and retR:
        # refine corners
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), term)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), term)

        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
        good_pairs += 1

        # draw and save debug images
        visL = imgL.copy()
        visR = imgR.copy()
        cv2.drawChessboardCorners(visL, CHECKERBOARD, cornersL, retL)
        cv2.drawChessboardCorners(visR, CHECKERBOARD, cornersR, retR)
        cv2.imwrite(f"debug_left/debug_left_{idx}.png", visL)
        cv2.imwrite(f"debug_right/debug_right_{idx}.png", visR)
        print(f"[OK] Pair {idx} - corners found and saved debug images.")
    else:
        print(f"[BAD] Checkerboard not found in pair: {left_img_path}  {right_img_path}")

if good_pairs == 0:
    print("No good pairs found. Re-capture images ensuring the checkerboard is visible & not blurry.")
    exit(1)

print(f"\nUsing {good_pairs} good pairs for calibration.\n")

# --- Calibration ---
print("Calibrating LEFT camera...")
retL, K1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
print("Left RMS:", retL)

print("\nCalibrating RIGHT camera...")
retR, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)
print("Right RMS:", retR)

# --- Stereo calibration ---
print("\nRunning stereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

retS, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K1,
    dist1,
    K2,
    dist2,
    grayL.shape[::-1],
    criteria=criteria,
    flags=flags
)
print("Stereo RMS:", retS)

# --- Projection matrices ---
R1 = np.eye(3)
T1 = np.zeros((3, 1))
P1 = K1 @ np.hstack((R1, T1))  # Left camera
P2 = K2 @ np.hstack((R, T))    # Right camera

baseline_m = np.linalg.norm(T) / 1000.0 if np.linalg.norm(T) > 0 else np.linalg.norm(T)  # if T in mm else in units of SQUARE_SIZE
print("\nRotation R:\n", R)
print("\nTranslation T:\n", T)
print(f"\nBaseline (norm of T): {np.linalg.norm(T):.3f} (units same as SQUARE_SIZE; if SQUARE_SIZE=27mm, divide by 1000 to get meters)")

print("\nSaving params to stereo_params.npz...")
np.savez("stereo_params.npz",
         K1=K1, dist1=dist1,
         K2=K2, dist2=dist2,
         R=R, T=T,
         P1=P1, P2=P2)

print("\nDONE! Stereo parameters saved as stereo_params.npz and debug images in debug_left/debug_right.")
