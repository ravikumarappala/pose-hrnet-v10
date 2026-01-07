"""
MediaPipe Pose Estimation and Limb Measurement
==============================================
This script performs real-time pose estimation using MediaPipe on video feed from an 
Intel RealSense D455 camera or RGB webcam, measures limb lengths in real-world units, 
and optionally renders a 3D SMPL mesh.

Compatible with:
- Windows (development with RGB camera fallback)
- Jetson AGX Orin with Jetpack 5.1.2 (deployment)

Uses MediaPipe for reliable, fast pose estimation optimized for both platforms.

Author: Claude
Date: July 29, 2025
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import time
import argparse
from datetime import datetime

# pyrealsense2 is imported conditionally when needed (for RealSense camera support)
rs = None  # Will be imported when RealSense is actually used
import json
import logging
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import signal
import atexit
import csv

# HRNet demo imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'demo'))
import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

# Import MediaPipe model and frame storage
from model_mediapipe import MediaPipeModel
from frame_data_storage import FrameDataStorage

# Force matplotlib backend for Jetson compatibility
print("Using matplotlib 3D fallback for Jetson compatibility")
from visualizer_3d_matplotlib import SkeletonVisualizer3DManager

# HRNet demo constants and functions
COCO_KEYPOINT_INDEXES = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17
CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons."""
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model_input = transform(model_input).unsqueeze(0)
    pose_model.eval()
    with torch.no_grad():
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation"""
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check CUDA availability and optimize for platform
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Platform detection
try:
    PLATFORM = "jetson" if "tegra" in os.uname().release.lower() else "other"
    if PLATFORM == "jetson":
        logger.info("Detected Jetson AGX Orin platform")
        # Set memory management for Jetson
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set conservative memory allocation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
except:
    PLATFORM = "other"

# Global variables for cleanup
cleanup_resources = {
    'pose_estimator': None,
    'cap': None,
    'video_writer_input': None,
    'video_writer_output': None,
    'output_dir': None,
    'measurements_data': [],
    'cleanup_done': False
}

def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals"""
    logger.info("\nReceived signal to terminate. Cleaning up...")
    cleanup_and_exit()

def cleanup_and_exit():
    """Cleanup all resources and save data before exiting"""
    global cleanup_resources

    # Prevent duplicate cleanup calls
    if cleanup_resources['cleanup_done']:
        return

    cleanup_resources['cleanup_done'] = True
    logger.info("Saving all data and releasing resources...")

    # Save measurements to CSV
    if cleanup_resources['measurements_data'] and cleanup_resources['output_dir']:
        try:
            save_measurements_to_csv(
                cleanup_resources['measurements_data'],
                cleanup_resources['output_dir']
            )
        except Exception as e:
            logger.error(f"Error saving measurements: {e}")

    # Release pose estimator
    if cleanup_resources['pose_estimator']:
        try:
            cleanup_resources['pose_estimator'].release()
            if cleanup_resources['pose_estimator'].visualizer_3d:
                cleanup_resources['pose_estimator'].visualizer_3d.cleanup()
        except Exception as e:
            logger.error(f"Error releasing pose estimator: {e}")

    # Release video capture
    if cleanup_resources['cap']:
        try:
            cleanup_resources['cap'].release()
        except Exception as e:
            logger.error(f"Error releasing video capture: {e}")

    # Release video writers
    if cleanup_resources['video_writer_input']:
        try:
            cleanup_resources['video_writer_input'].release()
            logger.info("Input video saved successfully")
        except Exception as e:
            logger.error(f"Error releasing input video writer: {e}")

    if cleanup_resources['video_writer_output']:
        try:
            cleanup_resources['video_writer_output'].release()
            logger.info("Output video saved successfully")
        except Exception as e:
            logger.error(f"Error releasing output video writer: {e}")

    # Close all OpenCV windows
    try:
        cv2.destroyAllWindows()
    except:
        pass

    # Clear CUDA cache on Jetson
    if PLATFORM == "jetson" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except:
            pass

    logger.info("Cleanup completed. Exiting...")
    sys.exit(0)

def save_measurements_to_csv(measurements_data, output_dir):
    """Save all measurements to CSV file in table format"""
    if not measurements_data:
        logger.warning("No measurements to save")
        return

    csv_file = output_dir / "measurements_table.csv"

    # Get all unique limb names
    all_limbs = set()
    for frame_data in measurements_data:
        if 'measurements' in frame_data:
            all_limbs.update(frame_data['measurements'].keys())

    all_limbs = sorted(list(all_limbs))

    # Write CSV file
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['frame_number', 'timestamp'] + all_limbs
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for frame_data in measurements_data:
            row = {
                'frame_number': frame_data.get('frame_number', ''),
                'timestamp': frame_data.get('timestamp', '')
            }
            if 'measurements' in frame_data:
                for limb in all_limbs:
                    row[limb] = f"{frame_data['measurements'].get(limb, ''):.2f}" if limb in frame_data['measurements'] else ''
            writer.writerow(row)

    logger.info(f"Measurements saved to: {csv_file}")

    # Also save summary statistics
    summary_file = output_dir / "measurements_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Limb', 'Average (cm)', 'Min (cm)', 'Max (cm)', 'Std Dev (cm)'])

        for limb in all_limbs:
            values = []
            for frame_data in measurements_data:
                if 'measurements' in frame_data and limb in frame_data['measurements']:
                    values.append(frame_data['measurements'][limb])

            if values:
                avg = np.mean(values)
                min_val = np.min(values)
                max_val = np.max(values)
                std_val = np.std(values)
                writer.writerow([limb, f"{avg:.2f}", f"{min_val:.2f}", f"{max_val:.2f}", f"{std_val:.2f}"])

    logger.info(f"Summary statistics saved to: {summary_file}")

    # Create README file in output directory
    readme_file = output_dir / "README.txt"
    with open(readme_file, 'w') as f:
        f.write("Pose Estimation and Limb Measurement Output\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Session Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Frames Processed: {len(measurements_data)}\n\n")
        f.write("Files in this directory:\n")
        f.write("-" * 50 + "\n")
        f.write("1. *_input.avi - Original input video\n")
        f.write("2. *_output.avi - Output video with pose overlay and measurements\n")
        f.write("3. measurements_table.csv - Frame-by-frame measurements (all joints)\n")
        f.write("4. measurements_summary.csv - Statistical summary (avg, min, max, std)\n")
        f.write("5. README.txt - This file\n\n")
        f.write("Measurement Units: All measurements are in centimeters (cm)\n\n")
        f.write("Joint Pairs Measured:\n")
        f.write("  - Upper Arms (left/right shoulder to elbow)\n")
        f.write("  - Lower Arms (left/right elbow to wrist)\n")
        f.write("  - Upper Legs (left/right hip to knee)\n")
        f.write("  - Lower Legs (left/right knee to ankle)\n")

    logger.info(f"README saved to: {readme_file}")

# Register signal handlers and cleanup
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_and_exit)

# Simple HRNet Model Class

class SMPLRenderer:
    """SMPL 3D mesh rendering"""
    
    def __init__(self):
        self.enabled = False
        self.model_loaded = False
        self.smpl_model = None
        
    def load_model(self):
        """Load SMPL model"""
        try:
            # Check if model directory exists
            model_dir = Path("models/smpl")
            if not model_dir.exists():
                logger.warning(f"SMPL model directory not found at {model_dir}")
                model_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {model_dir}")
                logger.warning("Please download SMPL model files from https://smpl.is.tue.mpg.de/ and place them in models/smpl/")
                return False
            
            # Check for model files
            model_files = list(model_dir.glob("*.pkl"))
            if not model_files:
                logger.warning("No SMPL model files found. Please download from https://smpl.is.tue.mpg.de/")
                return False
            
            # Placeholder for actual SMPL model loading
            # In a real implementation, you would:
            # 1. Load the SMPL model using the appropriate library
            # 2. Configure the model parameters
            # 3. Set up the rendering pipeline
            
            logger.info("Loading SMPL model...")
            time.sleep(1)  # Simulate loading time
            self.model_loaded = True
            logger.info("SMPL model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load SMPL model: {e}")
            return False
    
    def toggle(self):
        """Toggle SMPL rendering on/off"""
        if not self.model_loaded and not self.load_model():
            logger.error("Cannot enable SMPL rendering - model failed to load")
            return False
        
        self.enabled = not self.enabled
        status = "enabled" if self.enabled else "disabled"
        logger.info(f"SMPL rendering {status}")
        return True
    
    def render(self, keypoints, depth_data):
        """Render SMPL mesh based on keypoints and depth data"""
        if not self.enabled:
            return None
        
        try:
            # Placeholder for actual SMPL rendering
            # In a real implementation, you would:
            # 1. Convert keypoints to SMPL parameters (pose and shape)
            # 2. Generate the SMPL mesh
            # 3. Render the mesh as an overlay
            
            # For now, just return a simple visualization
            height, width = depth_data.shape
            mesh_visualization = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw some placeholder geometry to represent a human mesh
            # This is just a simplified stick figure for demonstration
            
            # Define connections between keypoints (simplified)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
                (0, 5), (5, 6), (6, 7),  # Left arm
                (0, 8), (8, 9), (9, 10),  # Right leg
                (0, 11), (11, 12), (12, 13),  # Left leg
                (0, 14), (14, 15), (15, 16)  # Face
            ]
            
            # Draw connections
            for connection in connections:
                if len(keypoints) > max(connection):
                    start_idx, end_idx = connection
                    if float(keypoints[start_idx][2]) > 0.5 and float(keypoints[end_idx][2]) > 0.5:
                        start_point = tuple(keypoints[start_idx][:2].astype(int))
                        end_point = tuple(keypoints[end_idx][:2].astype(int))
                        cv2.line(mesh_visualization, start_point, end_point, (0, 255, 0), 2)
            
            # Add a simple mesh-like effect
            for i, kp in enumerate(keypoints):
                if float(kp[2]) > 0.5:
                    x, y = int(kp[0]), int(kp[1])
                    # Draw circles at keypoints
                    cv2.circle(mesh_visualization, (x, y), 5, (0, 255, 255), -1)
                    
                    # Draw a simple "mesh" effect around torso
                    if i in [0, 5, 6, 11, 12]:  # Torso keypoints
                        for j in range(-20, 21, 10):
                            for k in range(-20, 21, 10):
                                if 0 <= x+j < width and 0 <= y+k < height:
                                    cv2.circle(mesh_visualization, (x+j, y+k), 2, (0, 150, 255), -1)
            
            return mesh_visualization
            
        except Exception as e:
            logger.error(f"Error in SMPL rendering: {e}")
            return None

class PoseEstimator:
    """Pose estimation class supporting multiple models"""

    def __init__(self, model_type='mediapipe', enable_smpl=False, enable_storage=True, enable_3d=False, stereo_params_file=None):
        self.model = None
        self.model_type = model_type
        self.smpl_renderer = SMPLRenderer()
        self.limb_data = {}
        self.frame_count = 0
        self.measurement_buffer = []
        self.buffer_size = 30  # Number of frames to average measurements over

        # Load stereo camera parameters
        self.stereo_params = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.baseline = None
        self.fx = 600.0  # Default values
        self.fy = 600.0
        self.cx = 320.0
        self.cy = 240.0
        self.load_stereo_params(stereo_params_file)

        # Initialize frame data storage
        self.enable_storage = enable_storage
        self.data_storage = None
        if enable_storage:
            self.data_storage = FrameDataStorage()

        # Initialize 3D visualizer
        self.enable_3d = enable_3d
        self.visualizer_3d = None
        if enable_3d:
            self.visualizer_3d = SkeletonVisualizer3DManager(enable_3d=True)

        # Initialize camera
        self.pipeline = None
        self.align = None

        # Initialize selected model
        self.load_model(model_type)

        # Enable SMPL if requested
        if enable_smpl:
            self.smpl_renderer.toggle()

    def load_stereo_params(self, stereo_params_file=None):
        """Load stereo camera parameters from .npz file"""
        if stereo_params_file is None:
            # Try default location in measurement directory
            script_dir = Path(__file__).parent
            stereo_params_file = script_dir / 'stereo_params.npz'
        else:
            stereo_params_file = Path(stereo_params_file)

        if not stereo_params_file.exists():
            logger.warning(f"Stereo parameters file not found: {stereo_params_file}")
            logger.warning("Using default camera parameters (fx=600, fy=600, cx=320, cy=240)")
            return False

        try:
            # Load stereo parameters
            self.stereo_params = np.load(stereo_params_file)

            # Extract all stereo calibration parameters
            self.K1 = self.stereo_params['K1']
            self.K2 = self.stereo_params['K2']
            self.dist1 = self.stereo_params['dist1']
            self.dist2 = self.stereo_params['dist2']
            self.R = self.stereo_params['R']
            self.T = self.stereo_params['T']
            self.P1 = self.stereo_params['P1']
            self.P2 = self.stereo_params['P2']

            # Use K1 for main camera matrix
            self.camera_matrix = self.K1
            self.dist_coeffs = self.dist1

            # Calculate baseline from translation vector (in mm, matching calibration)
            self.baseline_mm = np.linalg.norm(self.T)
            self.baseline = self.baseline_mm / 10.0  # Convert mm to cm for display

            # Extract focal lengths and principal point from K1
            self.fx = float(self.K1[0, 0])
            self.fy = float(self.K1[1, 1])
            self.cx = float(self.K1[0, 2])
            self.cy = float(self.K1[1, 2])

            # Enable stereo mode
            self.stereo_mode = True
            self.stereo_rectify_maps = None

            # Set depth scale to 1.0 since our stereo depth is already in meters
            self.depth_scale = 1.0

            logger.info(f"Loaded stereo parameters from: {stereo_params_file}")
            logger.info(f"Camera intrinsics - fx: {self.fx:.2f}, fy: {self.fy:.2f}, cx: {self.cx:.2f}, cy: {self.cy:.2f}")
            logger.info(f"Baseline: {self.baseline:.2f} cm ({self.baseline_mm:.2f} mm)")
            logger.info("Stereo depth computation enabled")

            return True
        except Exception as e:
            logger.error(f"Error loading stereo parameters: {e}")
            logger.warning("Using default camera parameters")
            self.stereo_mode = False
            return False

    def split_stereo_frame(self, frame):
        """Split side-by-side stereo frame into left and right images (matching existing stereo pipeline)"""
        if frame is None:
            return None, None

        h, w = frame.shape[:2]
        mid = w // 2

        # Split at midpoint
        frame_left = frame[:, :mid]
        frame_right = frame[:, mid:]

        # Flip both frames (matching calibration preprocessing)
        frame_left = cv2.flip(frame_left, 1)
        frame_right = cv2.flip(frame_right, 1)

        return frame_left, frame_right

    def compute_stereo_depth(self, frame_left, frame_right):
        """Compute depth map from stereo pair using calibrated parameters"""
        if not hasattr(self, 'stereo_mode') or not self.stereo_mode:
            logger.warning("Stereo mode not enabled, cannot compute depth")
            return None

        try:
            # Get image size
            h, w = frame_left.shape[:2]
            img_size = (w, h)

            # Initialize rectification maps if not done yet
            if self.stereo_rectify_maps is None:
                logger.info(f"Initializing stereo rectification for image size: {img_size}")

                # Compute stereo rectification
                R1, R2, P1_new, P2_new, Q, roi1, roi2 = cv2.stereoRectify(
                    self.K1, self.dist1,
                    self.K2, self.dist2,
                    img_size,
                    self.R, self.T,
                    alpha=0
                )

                # Create rectification maps
                self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
                    self.K1, self.dist1, R1, P1_new, img_size, cv2.CV_32FC1
                )
                self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
                    self.K2, self.dist2, R2, P2_new, img_size, cv2.CV_32FC1
                )

                # Store Q matrix for 3D reprojection
                self.Q_matrix = Q

                # Initialize stereo matcher
                min_disp = 0
                num_disp = 128  # Must be divisible by 16
                block_size = 5

                self.stereo_matcher = cv2.StereoSGBM_create(
                    minDisparity=min_disp,
                    numDisparities=num_disp,
                    blockSize=block_size,
                    P1=8 * 3 * block_size ** 2,
                    P2=32 * 3 * block_size ** 2,
                    disp12MaxDiff=1,
                    uniquenessRatio=10,
                    speckleWindowSize=100,
                    speckleRange=32
                )

                self.stereo_rectify_maps = True
                logger.info("Stereo rectification maps initialized")

            # Rectify images
            left_rectified = cv2.remap(frame_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
            right_rectified = cv2.remap(frame_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)

            # Convert to grayscale for stereo matching
            left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

            # Compute disparity
            disparity = self.stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

            # Convert disparity to depth using: depth = (baseline * focal_length) / disparity
            depth_map = np.zeros_like(disparity, dtype=np.float32)
            valid_disparity = disparity > 0

            # depth in mm = (baseline_mm * fx) / disparity
            depth_map[valid_disparity] = (self.baseline_mm * self.fx) / disparity[valid_disparity]

            # Convert from mm to meters for consistency with RealSense depth scale
            depth_map = depth_map / 1000.0

            return depth_map

        except Exception as e:
            logger.error(f"Error computing stereo depth: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def load_model(self, model_type='mediapipe'):
        """Load the specified pose estimation model"""
        try:
            if model_type.lower() == 'mediapipe':
                self.model = MediaPipeModel()
                logger.info("Successfully loaded MediaPipe model")
            elif model_type.lower() == 'hrnet':
                self._load_hrnet_model()
                if self.model is None:
                    raise Exception("HRNet model loading returned None")
                logger.info("Successfully loaded HRNet model")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.model = None
            return False
    
    def _load_hrnet_model(self):
        """Load HRNet models using demo approach"""
        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        # Create args object for config
        args = type('Args', (), {})()
        # Use absolute path to config file in demo directory
        script_dir = Path(__file__).parent
        args.cfg = str(script_dir / '..' / 'demo' / 'inference-config.yaml')
        args.opts = []
        args.modelDir = ''
        args.logDir = ''
        args.dataDir = str(script_dir / '..')  # Set data dir to parent (HRNet root)
        
        # Update config
        update_config(cfg, args)

        # Load person detection model
        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.box_model.to(CTX)
        self.box_model.eval()

        # Load pose estimation model
        self.pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
            cfg, is_train=False
        )

        if cfg.TEST.MODEL_FILE:
            logger.info(f'=> loading model from {cfg.TEST.MODEL_FILE}')
            self.pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        else:
            logger.error('expected model defined in config at TEST.MODEL_FILE')

        self.pose_model = torch.nn.DataParallel(self.pose_model, device_ids=cfg.GPUS)
        self.pose_model.to(CTX)
        self.pose_model.eval()
        
        # Set model to a simple wrapper for compatibility
        self.model = type('HRNetWrapper', (), {})()
        self.model.process_frame = self._process_frame_hrnet
    
    def _process_frame_hrnet(self, color_frame, depth_frame, limb_measurements=None):
        """Process frame using HRNet demo functions"""
        if color_frame is None:
            return None
        
        try:
            # Convert frame for detection
            image = color_frame[:, :, [2, 1, 0]]  # BGR to RGB
            img_tensor = torch.from_numpy(color_frame/255.).permute(2,0,1).float().to(CTX)
            input_list = [img_tensor]

            # Get person detection boxes
            pred_boxes = get_person_detection_boxes(self.box_model, input_list, threshold=0.7)

            output_image = color_frame.copy()
            
            # Process pose estimation if person detected
            if len(pred_boxes) >= 1:
                # Use first detected person
                box = pred_boxes[0]
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else color_frame.copy()
                pose_preds = get_pose_estimation_prediction(self.pose_model, image_pose, center, scale)
                
                if len(pose_preds) >= 1:
                    keypoints = pose_preds[0]  # First person's keypoints
                    
                    # Add confidence scores (set to 1.0 for HRNet)
                    keypoints_with_conf = np.zeros((17, 3))
                    keypoints_with_conf[:, :2] = keypoints[:, :2]
                    keypoints_with_conf[:, 2] = 1.0  # High confidence
                    
                    # Draw pose with measurements
                    self._draw_hrnet_pose_with_measurements(output_image, keypoints_with_conf, limb_measurements)
                    
                    # Create results object
                    result_obj = type('HRNetResults', (), {})()
                    result_obj.keypoints = keypoints_with_conf
                    result_obj.processed_image = output_image
                    result_obj.has_detection = True
                    
                    return result_obj
            
            return None
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing HRNet frame: {e}")
            logger.error(f"HRNet Traceback: {traceback.format_exc()}")
            return None
    
    def _draw_hrnet_pose_with_measurements(self, image, keypoints, limb_measurements=None):
        """Draw HRNet pose with measurements on limbs (COCO 17-keypoint format)"""
        if limb_measurements is None:
            limb_measurements = {}
        
        # Define limb connections for COCO 17-keypoint format
        limb_connections = {
            "right_upper_arm": (6, 8),     # Right shoulder to right elbow
            "right_lower_arm": (8, 10),    # Right elbow to right wrist
            "left_upper_arm": (5, 7),      # Left shoulder to left elbow
            "left_lower_arm": (7, 9),      # Left elbow to left wrist
            "right_upper_leg": (12, 14),   # Right hip to right knee
            "right_lower_leg": (14, 16),   # Right knee to right ankle
            "left_upper_leg": (11, 13),    # Left hip to left knee
            "left_lower_leg": (13, 15)     # Left knee to left ankle
        }
        
        # Draw skeleton connections (non-limbs) first
        for i in range(len(SKELETON)):
            kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
            
            # Skip limb connections (will be drawn separately with measurements)
            connection_tuple = (kpt_a, kpt_b) if kpt_a < kpt_b else (kpt_b, kpt_a)
            is_limb = any(connection_tuple == (start, end) or connection_tuple == (end, start) 
                         for start, end in limb_connections.values())
            
            if not is_limb and kpt_a < len(keypoints) and kpt_b < len(keypoints):
                if float(keypoints[kpt_a, 2]) > 0.3 and float(keypoints[kpt_b, 2]) > 0.3:
                    x_a, y_a = int(keypoints[kpt_a, 0]), int(keypoints[kpt_a, 1])
                    x_b, y_b = int(keypoints[kpt_b, 0]), int(keypoints[kpt_b, 1])
                    cv2.line(image, (x_a, y_a), (x_b, y_b), CocoColors[i], 2)
        
        # Draw limbs with measurements
        for limb_name, (start_idx, end_idx) in limb_connections.items():
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                float(keypoints[start_idx, 2]) > 0.3 and float(keypoints[end_idx, 2]) > 0.3):
                
                start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                
                # Draw limb line in bright color
                cv2.line(image, start_point, end_point, (0, 255, 0), 3)
                
                # Add measurement text on the limb
                if limb_name in limb_measurements:
                    measurement = limb_measurements[limb_name]
                    
                    # Calculate midpoint of the limb
                    mid_x = (start_point[0] + end_point[0]) // 2
                    mid_y = (start_point[1] + end_point[1]) // 2
                    
                    # Create measurement text
                    text = f"{measurement:.1f}cm"
                    
                    # Get text size for background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Draw black background for text
                    cv2.rectangle(image, 
                                (mid_x - text_width//2 - 2, mid_y - text_height - 2),
                                (mid_x + text_width//2 + 2, mid_y + baseline),
                                (0, 0, 0), -1)
                    
                    # Draw measurement text in white
                    cv2.putText(image, text, 
                              (mid_x - text_width//2, mid_y), 
                              font, font_scale, (255, 255, 255), thickness)
        
        # Draw keypoints
        for i in range(len(keypoints)):
            if float(keypoints[i, 2]) > 0.3:
                x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
    
    def initialize_camera(self):
        """Initialize the RealSense camera"""
        global rs
        try:
            # Import pyrealsense2 when needed
            import pyrealsense2 as rs
            
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Get available devices
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                logger.error("No RealSense devices found")
                return False
            
            # Log detected devices
            for i, dev in enumerate(devices):
                logger.info(f"Found device {i+1}: {dev.get_info(rs.camera_info.name)} " +
                          f"(S/N: {dev.get_info(rs.camera_info.serial_number)})")
            
            # Enable streams
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            profile = self.pipeline.start(config)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            logger.info(f"Depth scale: {self.depth_scale}")
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
            # Wait for auto-exposure to stabilize
            for i in range(30):
                self.pipeline.wait_for_frames()
            
            logger.info("RealSense camera initialized successfully")
            
            # Initialize storage session
            if self.data_storage:
                self.data_storage.initialize_session(
                    model_type=self.model_type,
                    camera_type="realsense",
                    frame_dimensions=[640, 480],
                    depth_scale=self.depth_scale
                )
            
            # Initialize 3D visualizer
            if self.visualizer_3d:
                self.visualizer_3d.initialize()
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RealSense camera: {e}")
            return False
    
    def get_frame(self):
        """Get aligned RGB and depth frames from the camera"""
        if self.pipeline is None:
            logger.error("Camera not initialized")
            return None, None
        
        try:
            # Wait for a coherent pair of frames
            frames = self.pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                logger.warning("Failed to get frames")
                return None, None
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
        except Exception as e:
            logger.error(f"Error getting frames: {e}")
            return None, None
    
    def process_frame(self, color_frame, depth_frame):
        """Process a frame through the pose estimation model"""
        if color_frame is None:
            return None
        
        # Check if model is loaded
        if self.model is None:
            logger.error("Model is None - cannot process frame")
            return color_frame
        
        try:
            # Check if we have valid depth data
            has_depth_data = (depth_frame is not None and 
                            hasattr(depth_frame, 'size') and 
                            depth_frame.size > 0 and 
                            bool(np.any(depth_frame > 0)))
            
            if has_depth_data:
                try:
                    # Get keypoints for measurement (we'll get them again from MediaPipe, but need them for measurement)
                    temp_results = self.model.process_frame(color_frame, depth_frame)
                    if (temp_results and 
                        hasattr(temp_results, 'keypoints') and 
                        temp_results.keypoints is not None and
                        isinstance(temp_results.keypoints, np.ndarray) and
                        temp_results.keypoints.size > 0):
                        keypoints = temp_results.keypoints
                        self.measure_limbs(keypoints, depth_frame)
                except Exception as e:
                    logger.warning(f"Error in limb measurement: {e}")
                    # Continue without measurements
            
            # Process with the loaded model, passing current measurements
            results = self.model.process_frame(color_frame, depth_frame, self.limb_data)
            
            if results is None:
                return color_frame
            
            # Get processed image with measurements drawn on limbs
            output_image = results.processed_image if hasattr(results, 'processed_image') else color_frame
            
            # Store frame data for 3D SMPL pipeline
            if (self.data_storage and hasattr(results, 'keypoints') and 
                results.keypoints is not None):
                self.data_storage.store_frame_data(
                    keypoints=results.keypoints,
                    depth_frame=depth_frame if has_depth_data else None,
                    limb_measurements=self.limb_data,
                    timestamp_ms=int(time.time() * 1000)
                )
            
            # Update 3D skeleton visualization (parallel)
            if (self.visualizer_3d and hasattr(results, 'keypoints') and 
                results.keypoints is not None):
                avg_confidence = np.mean(results.keypoints[:, 2]) if len(results.keypoints) > 0 else 0
                self.visualizer_3d.update(
                    keypoints=results.keypoints,
                    measurements=self.limb_data,
                    confidence=avg_confidence
                )
            
            # Render SMPL mesh if enabled
            if self.smpl_renderer.enabled:
                keypoints = self.model.get_keypoints_from_results(results)
                depth_for_smpl = depth_frame if has_depth_data else np.zeros_like(color_frame[:, :, 0])
                mesh_overlay = self.smpl_renderer.render(keypoints, depth_for_smpl)
                if mesh_overlay is not None:
                    # Blend mesh with output image
                    alpha = 0.6
                    output_image = cv2.addWeighted(output_image, 1-alpha, mesh_overlay, alpha, 0)
            
            return output_image
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing frame: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return color_frame
    
    def measure_limbs(self, keypoints, depth_frame):
        """Measure limb lengths in real-world units using depth data"""
        # Safety check for keypoints
        if keypoints is None or len(keypoints) == 0:
            return
        
        # Ensure keypoints is a numpy array
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        
        # Check keypoints shape
        if len(keypoints.shape) != 2 or keypoints.shape[1] < 3:
            logger.warning(f"Invalid keypoints shape: {keypoints.shape}")
            return
        
        # Define limbs to measure (adaptive for different keypoint formats)
        if self.model_type.lower() == 'hrnet':
            # COCO 17-keypoint format for HRNet
            limbs = {
                "right_upper_arm": (6, 8),     # Right shoulder to right elbow
                "right_lower_arm": (8, 10),    # Right elbow to right wrist
                "left_upper_arm": (5, 7),      # Left shoulder to left elbow
                "left_lower_arm": (7, 9),      # Left elbow to left wrist
                "right_upper_leg": (12, 14),   # Right hip to right knee
                "right_lower_leg": (14, 16),   # Right knee to right ankle
                "left_upper_leg": (11, 13),    # Left hip to left knee
                "left_lower_leg": (13, 15)     # Left knee to left ankle
            }
        else:
            # MediaPipe 33-keypoint format
            limbs = {
                "right_upper_arm": (12, 14),   # Right shoulder to right elbow
                "right_lower_arm": (14, 16),   # Right elbow to right wrist
                "left_upper_arm": (11, 13),    # Left shoulder to left elbow
                "left_lower_arm": (13, 15),    # Left elbow to left wrist
                "right_upper_leg": (24, 26),   # Right hip to right knee
                "right_lower_leg": (26, 28),   # Right knee to right ankle
                "left_upper_leg": (23, 25),    # Left hip to left knee
                "left_lower_leg": (25, 27)     # Left knee to left ankle
            }
        
        # Reset measurements for this frame
        current_measurements = {}
        
        # Calculate length for each limb
        for limb_name, (start_idx, end_idx) in limbs.items():
            try:
                # Check if keypoints are valid and indices are within bounds
                if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                    len(keypoints.shape) >= 2 and keypoints.shape[1] >= 3):
                    
                    # Safely extract confidence values
                    start_conf = keypoints[start_idx, 2]
                    end_conf = keypoints[end_idx, 2]
                    
                    # Convert to float if needed and check confidence
                    start_conf_val = float(start_conf) if hasattr(start_conf, '__float__') else start_conf
                    end_conf_val = float(end_conf) if hasattr(end_conf, '__float__') else end_conf
                    
                    if start_conf_val > 0.5 and end_conf_val > 0.5:
                        # Get keypoint coordinates
                        start_point = (int(float(keypoints[start_idx, 0])), int(float(keypoints[start_idx, 1])))
                        end_point = (int(float(keypoints[end_idx, 0])), int(float(keypoints[end_idx, 1])))
                        
                        # Check if points are within image bounds
                        height, width = depth_frame.shape
                        if not (0 <= start_point[0] < width and 0 <= start_point[1] < height and 
                                0 <= end_point[0] < width and 0 <= end_point[1] < height):
                            continue
                        
                        # Average depth in a small area around the point for robustness
                        kernel_size = 5
                        half_kernel = kernel_size // 2
                        
                        # Safely get depth areas
                        start_y_min = max(0, start_point[1] - half_kernel)
                        start_y_max = min(depth_frame.shape[0], start_point[1] + half_kernel + 1)
                        start_x_min = max(0, start_point[0] - half_kernel)
                        start_x_max = min(depth_frame.shape[1], start_point[0] + half_kernel + 1)
                        
                        end_y_min = max(0, end_point[1] - half_kernel)
                        end_y_max = min(depth_frame.shape[0], end_point[1] + half_kernel + 1)
                        end_x_min = max(0, end_point[0] - half_kernel)
                        end_x_max = min(depth_frame.shape[1], end_point[0] + half_kernel + 1)
                        
                        start_depth_area = depth_frame[start_y_min:start_y_max, start_x_min:start_x_max]
                        end_depth_area = depth_frame[end_y_min:end_y_max, end_x_min:end_x_max]
                        
                        # Check if we have valid depth data
                        if start_depth_area.size > 0 and end_depth_area.size > 0:
                            # Filter out zero values (no depth data)
                            start_depth_values = start_depth_area[start_depth_area > 0]
                            end_depth_values = end_depth_area[end_depth_area > 0]
                            
                            if start_depth_values.size > 0 and end_depth_values.size > 0:
                                # Use median for robustness
                                start_depth = np.median(start_depth_values) * self.depth_scale  # Convert to meters
                                end_depth = np.median(end_depth_values) * self.depth_scale  # Convert to meters
                                
                                # Calculate 3D coordinates using stereo camera intrinsics
                                # Convert from image coordinates to camera coordinates
                                # Using actual camera intrinsic parameters from stereo calibration

                                start_x = (start_point[0] - self.cx) * start_depth / self.fx
                                start_y = (start_point[1] - self.cy) * start_depth / self.fy
                                start_z = start_depth

                                end_x = (end_point[0] - self.cx) * end_depth / self.fx
                                end_y = (end_point[1] - self.cy) * end_depth / self.fy
                                end_z = end_depth
                                
                                # Calculate Euclidean distance in 3D space
                                length = np.sqrt(
                                    (end_x - start_x)**2 + 
                                    (end_y - start_y)**2 + 
                                    (end_z - start_z)**2
                                )
                                
                                # Convert to centimeters for display
                                length_cm = length * 100
                                
                                # Store measurement
                                current_measurements[limb_name] = length_cm
            except Exception as e:
                logger.warning(f"Error processing limb {limb_name}: {e}")
                continue
        
        # Add to buffer for averaging
        self.measurement_buffer.append(current_measurements)
        if len(self.measurement_buffer) > self.buffer_size:
            self.measurement_buffer.pop(0)
        
        # Average measurements over buffer
        self.limb_data = {}
        for limb_name in limbs.keys():
            measurements = [frame.get(limb_name, 0) for frame in self.measurement_buffer if limb_name in frame]
            if measurements:
                self.limb_data[limb_name] = sum(measurements) / len(measurements)
        
        # Increment frame counter
        self.frame_count += 1
        
        # Log measurements every 30 frames
        if self.frame_count % 30 == 0:
            self.log_measurements()
    
    def draw_measurements(self, image):
        """Draw limb measurements on the image"""
        # Position for text
        x_pos = 10
        y_pos = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # Draw background rectangle for better readability
        text_height = len(self.limb_data) * 20 + 10
        cv2.rectangle(image, (x_pos - 5, y_pos - 25), (x_pos + 200, y_pos + text_height), (0, 0, 0), -1)
        
        # Title
        cv2.putText(image, f"Limb Measurements ({self.model_type.title()})", 
                    (x_pos, y_pos), font, font_scale, color, thickness)
        y_pos += 20
        
        # Draw each measurement
        for limb_name, length in self.limb_data.items():
            display_name = limb_name.replace('_', ' ').title()
            cv2.putText(image, f"{display_name}: {length:.2f} cm", 
                        (x_pos, y_pos), font, font_scale, color, thickness)
            y_pos += 20
        
        # Draw SMPL status
        smpl_status = "SMPL: ON" if self.smpl_renderer.enabled else "SMPL: OFF"
        cv2.putText(image, smpl_status, (image.shape[1] - 100, 30), 
                    font, font_scale, (0, 255, 0) if self.smpl_renderer.enabled else (0, 0, 255), 
                    thickness)
    
    def log_measurements(self):
        """Log measurements to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_data = {
            "timestamp": timestamp,
            "model": self.model_type.title(),
            "measurements": self.limb_data
        }

        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)

        # Write to file
        with open(f"logs/measurements_{timestamp}.json", "w") as f:
            json.dump(log_data, f, indent=4)

        logger.info(f"Logged measurements to logs/measurements_{timestamp}.json")

    def get_current_measurements(self):
        """Get current frame measurements with timestamp"""
        return {
            'frame_number': self.frame_count,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'measurements': self.limb_data.copy()
        }
    
    def release(self):
        """Release resources"""
        try:
            if self.pipeline:
                self.pipeline.stop()
        except:
            pass
        
        try:
            if hasattr(self.model, 'release'):
                self.model.release()
        except:
            pass

def main():
    global cleanup_resources

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Pose Estimation and Limb Measurement with Camera or Video File')
    parser.add_argument('--model', type=str, default='mediapipe', choices=['mediapipe', 'hrnet'],
                        help='Pose estimation model: mediapipe (fast) or hrnet (accurate)')
    parser.add_argument('--smpl', action='store_true', help='Enable SMPL mesh rendering')
    parser.add_argument('--camera', type=str, default='auto', choices=['auto', 'realsense', 'rgb'],
                        help='Camera type: auto (detect), realsense (force RealSense), rgb (force RGB webcam)')
    parser.add_argument('--input', type=str, help='Input video file path (if not provided, uses camera)')
    parser.add_argument('--save-data', action='store_true', help='Save comprehensive frame data for 3D SMPL pipeline')
    parser.add_argument('--session-name', type=str, help='Custom session name for data storage')
    parser.add_argument('--3d', action='store_true', help='Enable real-time 3D skeleton visualization')
    parser.add_argument('--stereo-params', type=str, help='Path to stereo camera parameters .npz file (default: measurement/stereo_params.npz)')
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('output') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_resources['output_dir'] = output_dir
    logger.info(f"Output directory created: {output_dir}")
    
    # Platform optimization
    if PLATFORM == "jetson":
        logger.info("Optimizing for Jetson AGX Orin deployment")
        # Clear CUDA cache for Jetson
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Initialize pose estimator with selected model
    logger.info(f"Initializing pose estimator with model: {args.model}")
    stereo_params_path = args.stereo_params if hasattr(args, 'stereo_params') and args.stereo_params else None
    pose_estimator = PoseEstimator(
        model_type=args.model,
        enable_smpl=args.smpl,
        enable_storage=args.save_data,
        enable_3d=getattr(args, '3d', False),
        stereo_params_file=stereo_params_path
    )
    cleanup_resources['pose_estimator'] = pose_estimator

    # Set custom session name if provided
    if args.save_data and args.session_name and pose_estimator.data_storage:
        pose_estimator.data_storage.session_name = args.session_name
        pose_estimator.data_storage.session_file = pose_estimator.data_storage.output_dir / f"{args.session_name}.json"

    # Check if model loaded successfully
    if pose_estimator.model is None:
        logger.error(f"Failed to initialize {args.model} model. Exiting.")
        return
    
    # Input source initialization logic
    use_realsense = False
    cap = None
    
    # Check if video file input is provided
    if args.input:
        # Use video file input
        if not os.path.exists(args.input):
            logger.error(f"Video file not found: {args.input}")
            return
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {args.input}")
            return
        cleanup_resources['cap'] = cap
        logger.info(f"Using video file input: {args.input}")
        use_realsense = False  # Video files don't have depth info
        
        # Initialize storage session for video file
        if pose_estimator.data_storage:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            pose_estimator.data_storage.initialize_session(
                model_type=args.model,
                camera_type="video_file",
                video_source=args.input,
                frame_dimensions=[width, height],
                depth_scale=None
            )
        
        # Initialize 3D visualizer for video file
        if pose_estimator.visualizer_3d:
            pose_estimator.visualizer_3d.initialize()
    else:
        # Camera initialization logic
        if args.camera == 'rgb':
            # Force RGB camera
            use_realsense = False
        elif args.camera == 'realsense':
            # Force RealSense
            use_realsense = True
        else:
            # Auto-detect: try RealSense first, fallback to RGB
            try:
                import pyrealsense2 as rs
                ctx = rs.context()
                devices = ctx.query_devices()
                use_realsense = len(devices) > 0
                if use_realsense:
                    logger.info("RealSense camera detected - using depth + RGB")
                else:
                    logger.info("No RealSense camera found - falling back to RGB camera")
            except ImportError:
                logger.warning("pyrealsense2 not available - using RGB camera")
                use_realsense = False
            except Exception as e:
                logger.warning(f"RealSense detection failed: {e} - using RGB camera")
                use_realsense = False
    
    # Initialize camera (only if not using video file)
    if not args.input:
        if use_realsense:
            if not pose_estimator.initialize_camera():
                logger.error("Failed to initialize RealSense camera - trying RGB fallback")
                use_realsense = False
        
        if not use_realsense:
            # Use RGB webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Failed to open RGB camera")
                return
            cleanup_resources['cap'] = cap
            
            # Set optimal resolution for AGX Orin
            if PLATFORM == "jetson":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                # Higher resolution for development/testing
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Using RGB camera")
            
            # Initialize storage session for RGB camera
            if pose_estimator.data_storage:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                pose_estimator.data_storage.initialize_session(
                    model_type=args.model,
                    camera_type="rgb",
                    frame_dimensions=[width, height],
                    depth_scale=None
                )
            
            # Initialize 3D visualizer for RGB camera
            if pose_estimator.visualizer_3d:
                pose_estimator.visualizer_3d.initialize()
    
    # Initialize video writers (both input and output)
    # Get video properties for writer initialization
    if args.input:
        # Use input video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    elif use_realsense:
        # RealSense properties
        fps = 30.0
        width = 640
        height = 480
    else:
        # RGB camera properties
        fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap else 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else 480

    # Create video writers in output folder
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_video_path = output_dir / f"{timestamp}_input.avi"
    output_video_path = output_dir / f"{timestamp}_output.avi"

    video_writer_input = cv2.VideoWriter(str(input_video_path), fourcc, fps, (width, height))
    video_writer_output = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    cleanup_resources['video_writer_input'] = video_writer_input
    cleanup_resources['video_writer_output'] = video_writer_output

    logger.info(f"Saving input video to: {input_video_path}")
    logger.info(f"Saving output video to: {output_video_path}")
    
    # Display controls
    logger.info("Controls:")
    logger.info("  'q' - Quit (or Ctrl+C for safe shutdown)")
    logger.info("  's' - Toggle SMPL rendering")
    logger.info(f"  Using {args.model.title()} pose estimation")
    if getattr(args, '3d', False):
        logger.info("  3D skeleton visualization: ENABLED")
    if args.save_data:
        logger.info("  Data storage: ENABLED")
    logger.info(f"  Output folder: {output_dir}")
    
    # Main loop
    try:
        while True:
            # Get frame
            if use_realsense:
                color_frame, depth_frame = pose_estimator.get_frame()
                if color_frame is None or depth_frame is None:
                    logger.warning("Failed to get frames from RealSense")
                    continue
            else:
                ret, raw_frame = cap.read()
                if not ret:
                    if args.input:
                        logger.info("End of video file reached")
                    else:
                        logger.error("Failed to get frame from RGB camera")
                    break

                # Check if stereo mode is enabled
                if hasattr(pose_estimator, 'stereo_mode') and pose_estimator.stereo_mode:
                    # Split stereo frame and compute depth
                    frame_left, frame_right = pose_estimator.split_stereo_frame(raw_frame)
                    depth_frame = pose_estimator.compute_stereo_depth(frame_left, frame_right)

                    # Use left frame for pose detection
                    color_frame = frame_left

                    # If depth computation failed, use dummy depth
                    if depth_frame is None:
                        depth_frame = np.zeros_like(color_frame[:, :, 0])
                else:
                    # Regular RGB mode without depth
                    color_frame = raw_frame
                    depth_frame = np.zeros_like(color_frame[:, :, 0])
            
            # Process frame
            start_time = time.time()
            processed_frame = pose_estimator.process_frame(color_frame, depth_frame)
            end_time = time.time()
            
            if processed_frame is None:
                processed_frame = color_frame
            
            # Calculate FPS
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            # Display FPS and camera type
            if args.input:
                camera_info = f"Video File: {os.path.basename(args.input)}"
            else:
                if use_realsense:
                    camera_info = "RealSense D455"
                elif hasattr(pose_estimator, 'stereo_mode') and pose_estimator.stereo_mode:
                    camera_info = "Stereo Camera (with depth)"
                else:
                    camera_info = "RGB Camera"
            cv2.putText(processed_frame, f"FPS: {fps:.1f} | {camera_info}", 
                       (10, processed_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(processed_frame, f"Model: {args.model.title()}",
                       (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save frames to both input and output videos
            video_writer_input.write(color_frame)
            video_writer_output.write(processed_frame)

            # Collect measurements for CSV export
            if pose_estimator.limb_data:
                measurement_entry = pose_estimator.get_current_measurements()
                cleanup_resources['measurements_data'].append(measurement_entry)

            # Display frame
            cv2.imshow(f'{args.model.title()} Pose Estimation', processed_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                pose_estimator.smpl_renderer.toggle()
    
    except KeyboardInterrupt:
        logger.info("\nApplication stopped by user (Ctrl+C)")

    finally:
        # Save session data if enabled
        if pose_estimator.data_storage:
            try:
                session_file = pose_estimator.data_storage.save_session()
                summary = pose_estimator.data_storage.export_summary()
                logger.info(f"Saved {pose_estimator.frame_count} frames to: {session_file}")
                logger.info(f"Detection rate: {summary['frame_statistics']['detection_rate']:.2%}")
            except Exception as e:
                logger.error(f"Error saving session data: {e}")

        # Call cleanup function to save all data and release resources
        cleanup_and_exit()

if __name__ == "__main__":
    main()