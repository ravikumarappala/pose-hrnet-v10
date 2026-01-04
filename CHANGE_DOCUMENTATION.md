# HRNet Pose Estimation System - Change Documentation

**Document Version:** 1.0
**Date:** January 4, 2026
**Repository:** HRNet_0.1 Enhanced Pose Estimation System
**Platform:** Jetson AGX Orin with JetPack 5.1.2

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Modified Files](#modified-files)
3. [New Files Added](#new-files-added)
4. [New Directories](#new-directories)
5. [Architecture Overview](#architecture-overview)
6. [Feature Enhancements](#feature-enhancements)
7. [Dependencies](#dependencies)
8. [Usage Guide](#usage-guide)
9. [API Documentation](#api-documentation)
10. [Deployment Notes](#deployment-notes)
11. [File Reference](#file-reference)

---

## Executive Summary

This document captures all modifications and additions made to the HRNet pose estimation repository by the previous development team. The system has been significantly enhanced from a basic HRNet demo to a comprehensive pose estimation platform with the following capabilities:

### Key Achievements
- **Multi-model support**: MediaPipe (fast) and HRNet (accurate)
- **Real-world measurements**: 3D limb length calculation in centimeters
- **API services**: FastAPI-based REST and WebSocket endpoints
- **3D visualization**: Real-time skeleton rendering
- **Data pipeline**: Comprehensive frame data storage for SMPL reconstruction
- **Platform optimization**: Jetson AGX Orin specific enhancements

### Statistics
- **Modified files**: 2
- **New files in demo/**: 15+
- **New directories**: 3
- **Total new code**: ~150KB
- **New dependencies**: 4 major packages

---

## Modified Files

### 1. demo/demo.py

**File Path:** `demo/demo.py`
**Size:** Not specified
**Purpose:** Main HRNet demo script

#### Changes Made

**Change 1: Added Camera Device Selection**
- **Line 203**: Added new command-line argument
  ```python
  parser.add_argument('--cam_id', type=int, default=0,
                      help='Camera device ID (e.g., 0 for /dev/video0, 1 for /dev/video1)')
  ```

- **Line 249**: Updated camera initialization
  ```python
  # Before:
  vidcap = cv2.VideoCapture(0)

  # After:
  vidcap = cv2.VideoCapture(args.cam_id)
  ```

**Rationale:** Allows users to select different camera devices when multiple cameras are connected to the system.

**Impact:** Non-breaking change, backwards compatible (defaults to camera 0)

---

### 2. requirements.txt

**File Path:** `requirements.txt`
**Original Dependencies:** 9 packages
**Updated Dependencies:** 13 packages

#### Changes Made

**Change 1: Updated OpenCV**
```diff
- opencv-python==3.4.1.15
+ opencv-python
```
**Reason:** Remove version constraint for better compatibility with newer systems

**Change 2: Fixed NumPy Version**
```diff
+ numpy==1.24.4
```
**Reason:** Compatibility with Jetson and other dependencies

**Change 3: Added YACS (Conda Build)**
```diff
+ yacs @ file:///home/conda/feedstock_root/build_artifacts/yacs_1645705974477/work
```
**Reason:** Configuration management from conda build

**Change 4: Added API Dependencies**
```diff
+ uvicorn==0.33.0
+ fastapi==0.122.0
```
**Reason:** Support FastAPI-based pose estimation service

#### Complete New requirements.txt
```
EasyDict==1.7
opencv-python
shapely==1.6.4
Cython
scipy
opencv-contrib-python
matplotlib
json_tricks
scikit-image
yacs>=0.1.5
tensorboardX==1.6
numpy==1.24.4
yacs @ file:///home/conda/feedstock_root/build_artifacts/yacs_1645705974477/work
uvicorn==0.33.0
fastapi==0.122.0
```

---

## New Files Added

### Demo Directory Files

#### 1. demo/capture_body.py

**File Path:** `demo/capture_body.py`
**Size:** 40 lines
**Purpose:** Camera utility for capturing full-body images

**Key Features:**
- Uses V4L2 backend for Jetson compatibility
- Specific device path: `/dev/v4l/by-path/platform-3610000.xhci-usb-0:4.3:1.0-video-index0`
- Resolution: 1280x720 for full body capture
- Interactive controls: 'c' to capture, 'q' to quit
- Output: Saves as `person.jpg`

**Code Structure:**
```python
DEVICE_PATH = "/dev/v4l/by-path/platform-3610000.xhci-usb-0:4.3:1.0-video-index0"
cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

**Usage:**
```bash
python demo/capture_body.py
```

---

#### 2. demo/demo-api.py

**File Path:** `demo/demo-api.py`
**Size:** 354 lines
**Purpose:** FastAPI-based REST API and WebSocket server for pose estimation

**Key Features:**

1. **HTTP Endpoint** - `/process-frame` (POST)
   - Input: JPEG image (binary)
   - Output: Processed JPEG with skeleton overlay
   - Use case: Single frame processing

2. **WebSocket Endpoint** - `/ws-stream`
   - Input: Binary JPEG stream
   - Output: Processed JPEG stream
   - Use case: Real-time video processing

3. **Models Loaded on Startup:**
   - Faster R-CNN ResNet50 FPN (person detection)
   - HRNet W32 384x288 (pose estimation)

4. **Core Processing:**
   - Person detection (threshold: 0.9)
   - Pose estimation per detected person
   - Skeleton drawing with color-coded joints

**API Structure:**
```python
@app.on_event("startup")
def on_startup():
    init_models()  # Load models once

@app.post("/process-frame")
async def process_frame_api(request: Request):
    # Process single frame

@app.websocket("/ws-stream")
async def websocket_stream(ws: WebSocket):
    # Process streaming frames
```

**Server Configuration:**
- Host: 0.0.0.0
- Port: 8000
- ASGI Server: Uvicorn

**Usage:**
```bash
# Start server
python demo/demo-api.py

# HTTP endpoint
curl -X POST http://localhost:8000/process-frame \
  --data-binary @image.jpg \
  --output processed.jpg

# WebSocket (use WebSocket client)
ws://localhost:8000/ws-stream
```

---

#### 3. demo/limb_measurement.py

**File Path:** `demo/limb_measurement.py`
**Size:** 51,873 bytes (1,193 lines)
**Purpose:** Comprehensive pose estimation system with limb measurement
**Author:** Claude
**Date:** July 29, 2025

**Key Features:**

1. **Dual Model Support:**
   - **MediaPipe**: Fast (~30 FPS), 33 keypoints, CPU/GPU compatible
   - **HRNet**: Accurate (~15 FPS), 17 COCO keypoints, GPU recommended

2. **Camera Support:**
   - Intel RealSense D455 (depth + RGB)
   - Standard RGB webcam (auto-detect)
   - Video file input
   - Auto-detection with intelligent fallback

3. **3D Capabilities:**
   - Real-time 3D skeleton visualization (matplotlib backend)
   - 3D world coordinates calculation from depth data
   - SMPL mesh rendering (placeholder implementation ready)

4. **Limb Measurements:**
   - 8 limbs measured: upper/lower arms and legs (left/right)
   - Real-world units: Centimeters
   - Method: Depth-based 3D Euclidean distance
   - Smoothing: Moving average over 30-frame buffer
   - Display: On-limb measurement annotations

5. **Data Storage:**
   - Frame-by-frame comprehensive logging
   - JSON export with full session metadata
   - Auto-save every 30 frames
   - Statistics and summary generation

6. **Platform Optimization:**
   - Jetson AGX Orin specific optimizations
   - Conservative CUDA memory management
   - Platform detection and adaptive settings

**Command-Line Arguments:**
```bash
--model {mediapipe,hrnet}     # Pose model selection
--smpl                         # Enable SMPL mesh rendering
--camera {auto,realsense,rgb}  # Camera type
--input PATH                   # Video file input
--output PATH                  # Video output path
--save-data                    # Enable frame data storage
--session-name NAME            # Custom session name
--3d                          # Enable 3D visualization
```

**Code Organization:**

1. **Classes:**
   - `SMPLRenderer`: SMPL 3D mesh rendering
   - `PoseEstimator`: Main pose estimation orchestrator
   - Helper functions for HRNet integration

2. **Key Methods:**
   - `initialize_camera()`: RealSense camera setup
   - `get_frame()`: Aligned RGB + depth frame capture
   - `process_frame()`: Full processing pipeline
   - `measure_limbs()`: 3D limb length calculation
   - `draw_measurements()`: Overlay measurements on image

**Limb Measurement Details:**

**MediaPipe Format (33 keypoints):**
| Limb | Start | End | Keypoint Indices |
|------|-------|-----|------------------|
| Right Upper Arm | Right Shoulder | Right Elbow | (12, 14) |
| Right Lower Arm | Right Elbow | Right Wrist | (14, 16) |
| Left Upper Arm | Left Shoulder | Left Elbow | (11, 13) |
| Left Lower Arm | Left Elbow | Left Wrist | (13, 15) |
| Right Upper Leg | Right Hip | Right Knee | (24, 26) |
| Right Lower Leg | Right Knee | Right Ankle | (26, 28) |
| Left Upper Leg | Left Hip | Left Knee | (23, 25) |
| Left Lower Leg | Left Knee | Left Ankle | (25, 27) |

**HRNet Format (17 COCO keypoints):**
| Limb | Start | End | Keypoint Indices |
|------|-------|-----|------------------|
| Right Upper Arm | Right Shoulder | Right Elbow | (6, 8) |
| Right Lower Arm | Right Elbow | Right Wrist | (8, 10) |
| Left Upper Arm | Left Shoulder | Left Elbow | (5, 7) |
| Left Lower Arm | Left Elbow | Left Wrist | (7, 9) |
| Right Upper Leg | Right Hip | Right Knee | (12, 14) |
| Right Lower Leg | Right Knee | Right Ankle | (14, 16) |
| Left Upper Leg | Left Hip | Left Knee | (11, 13) |
| Left Lower Leg | Left Knee | Left Ankle | (13, 15) |

**3D Coordinate Calculation:**
```python
# Camera intrinsics (approximate)
fx = 600.0  # Focal length X (pixels)
fy = 600.0  # Focal length Y (pixels)
cx = width / 2   # Principal point X
cy = height / 2  # Principal point Y

# 2D to 3D conversion
world_x = (pixel_x - cx) * depth / fx
world_y = (pixel_y - cy) * depth / fy
world_z = depth

# 3D Euclidean distance
length = sqrt((x2-x1)² + (y2-y1)² + (z2-z1)²)
length_cm = length * 100
```

**Usage Examples:**
```bash
# MediaPipe with RealSense
python demo/limb_measurement.py --model mediapipe --camera realsense

# HRNet with 3D visualization
python demo/limb_measurement.py --model hrnet --3d

# Video file processing with data export
python demo/limb_measurement.py --input video.mp4 --output result.mp4 --save-data

# Full featured
python demo/limb_measurement.py --model hrnet --camera realsense \
  --3d --save-data --session-name session01
```

**Output Files:**
- `logs/measurements_TIMESTAMP.json` - Periodic measurement logs
- `measurement/data/SESSION_NAME.json` - Complete session data
- `measurement/data/SESSION_NAME_summary.json` - Session statistics

---

#### 4. demo/model_mediapipe.py

**File Path:** `demo/model_mediapipe.py`
**Size:** 222 lines
**Purpose:** MediaPipe model wrapper for pose estimation
**Author:** Claude
**Date:** July 29, 2025

**Key Features:**

1. **Platform-Aware Optimization:**
   - **Jetson Platform:**
     - Model complexity: 1 (lighter)
     - Detection confidence: 0.7 (higher)
     - Tracking confidence: 0.7
   - **Development Platform:**
     - Model complexity: 2 (full)
     - Detection confidence: 0.5
     - Tracking confidence: 0.5

2. **MediaPipe Configuration:**
   ```python
   self.model = self.mp_pose.Pose(
       static_image_mode=False,
       model_complexity=model_complexity,
       enable_segmentation=False,
       min_detection_confidence=min_detection_confidence,
       min_tracking_confidence=min_tracking_confidence
   )
   ```

3. **Custom Visualization:**
   - Limb-specific drawing with measurements
   - Color-coded skeleton (gray for body, green for limbs)
   - Measurement text overlay on limb midpoints
   - 33 keypoints in MediaPipe format

4. **Output Format:**
   - Returns structured object with:
     - `keypoints`: numpy array (33, 3) - [x, y, visibility]
     - `processed_image`: BGR image with skeleton overlay
     - `has_detection`: boolean flag

**Class Structure:**
```python
class MediaPipeModel:
    def __init__(self):
        self._load_model()

    def process_frame(self, color_frame, depth_frame, limb_measurements=None):
        # Main processing pipeline

    def _draw_pose_with_measurements(self, image, keypoints, limb_measurements):
        # Custom skeleton drawing

    def release(self):
        # Resource cleanup
```

---

#### 5. demo/frame_data_storage.py

**File Path:** `demo/frame_data_storage.py`
**Size:** 296 lines
**Purpose:** Frame data storage for 3D SMPL pipeline
**Author:** Claude
**Date:** August 28, 2025

**Key Features:**

1. **Session Management:**
   - Unique session naming with timestamps
   - Metadata tracking (model type, camera, resolution, depth scale)
   - Auto-save every 30 frames

2. **Frame Data Structure:**
   ```json
   {
     "frame_id": 0,
     "timestamp_ms": 1234567890,
     "keypoints": {
       "raw_2d": [[x, y], ...],
       "confidence": [0.9, 0.8, ...],
       "world_3d": [[x, y, z], ...],
       "format": "mediapipe_33"
     },
     "depth_data": {
       "keypoint_depths": [1.2, 1.3, ...],
       "depth_confidence": [0.95, 0.89, ...],
       "has_valid_depth": true
     },
     "limb_measurements": {
       "right_upper_arm": 28.5,
       "left_upper_arm": 27.9
     },
     "detection_quality": {
       "pose_detected": true,
       "avg_confidence": 0.87,
       "valid_keypoints": 31,
       "detection_score": 0.82
     }
   }
   ```

3. **3D Coordinate Calculation:**
   - Camera intrinsics application
   - Depth averaging over 5x5 kernel
   - Median filtering for robustness
   - Full 3D world coordinate output

4. **Export Capabilities:**
   - SMPL-ready data formatting
   - Session summary statistics
   - Limb measurement statistics (avg, std, min, max)
   - Detection rate calculations

**Class Structure:**
```python
class FrameDataStorage:
    def __init__(self, output_dir="measurement/data", session_name=None)
    def initialize_session(self, model_type, camera_type, ...)
    def store_frame_data(self, keypoints, depth_frame, limb_measurements, ...)
    def save_session(self)
    def load_session(self, session_file)
    def get_frames_for_smpl(self, frame_range=None)
    def export_summary(self)
```

**Output Location:**
- Session data: `measurement/data/SESSION_NAME.json`
- Summary: `measurement/data/SESSION_NAME_summary.json`

---

#### 6. demo/visualizer_3d_matplotlib.py

**File Path:** `demo/visualizer_3d_matplotlib.py`
**Size:** 399 lines
**Purpose:** 3D skeleton visualization using matplotlib
**Author:** Claude
**Date:** August 28, 2025

**Key Features:**

1. **Dual Format Support:**
   - Auto-detects COCO-17 or MediaPipe-33 format
   - Automatic bone connection mapping
   - Format-specific skeleton structure

2. **Visualization Features:**
   - Real-time 3D skeleton rendering
   - Coordinate normalization and auto-centering
   - Confidence-based fade effects
   - Measurement annotations on 3D bones
   - FPS monitoring
   - Interactive matplotlib 3D view

3. **Bone Connections:**
   ```python
   # MediaPipe-33 bones
   [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8),
    (11,12), (11,13), (13,15), (12,14), (14,16),
    (11,23), (12,24), (23,24),
    (23,25), (25,27), (24,26), (26,28)]

   # COCO-17 bones
   [(11,5), (12,6), (5,6), (11,12),
    (5,7), (7,9), (6,8), (8,10),
    (11,13), (13,15), (12,14), (14,16),
    (0,1), (1,2), (2,3), (3,4)]
   ```

4. **Coordinate System:**
   - X axis: Normalized horizontal (-1.5 to 1.5)
   - Y axis: Normalized vertical (-1.5 to 1.5)
   - Z axis: Confidence/depth (0 to 1)

5. **Color Coding:**
   - 12 distinct colors for bone segments
   - Color cycling for visibility
   - Alpha blending based on detection confidence

**Class Structure:**
```python
class SkeletonVisualizer3D:
    def initialize(self)
    def update_skeleton(self, keypoints, measurements, confidence)
    def _update_skeleton_geometry(self)
    def _normalize_coordinates(self, keypoints)
    def render_frame(self)
    def stop(self)

class SkeletonVisualizer3DManager:
    def initialize(self)
    def update(self, keypoints, measurements, confidence)
    def get_stats(self)
    def cleanup(self)
```

**Usage:**
```python
# Initialize
visualizer = SkeletonVisualizer3DManager(enable_3d=True)
visualizer.initialize()

# Update with pose data
visualizer.update(
    keypoints=keypoints_array,
    measurements=limb_dict,
    confidence=0.85
)

# Cleanup
visualizer.cleanup()
```

---

#### 7. demo/limb_usage.md

**File Path:** `demo/limb_usage.md`
**Size:** 158 lines
**Purpose:** User documentation and usage guide

**Content Sections:**

1. **Setup Instructions** - Prerequisites and environment setup
2. **Usage Examples** - Command-line examples for various scenarios
3. **Model Comparison** - MediaPipe vs HRNet feature comparison
4. **File Structure** - Directory layout and file locations
5. **Limb Measurements** - Keypoint mapping tables
6. **Requirements** - Dependencies for each model
7. **Troubleshooting** - Common issues and solutions
8. **Performance Notes** - Jetson AGX Orin benchmarks

**Key Usage Examples from Doc:**
```bash
# MediaPipe (Default)
python demo/limb_measurement.py

# HRNet (Accurate)
python demo/limb_measurement.py --model hrnet

# RealSense Camera
python demo/limb_measurement.py --model hrnet --camera realsense

# RGB Webcam with specific ID
python demo/limb_measurement.py --model mediapipe --camera rgb --camera-id 0

# SMPL Rendering
python demo/limb_measurement.py --model hrnet --smpl
```

---

#### 8. Additional Demo Files

**Test Images:**
- `demo/output_processed.jpg` - Processed output sample
- `demo/person.jpg` - Captured person image
- `demo/person_processed.jpg` - Processed person image

**Test Videos:**
- `demo/test2.mp4` - Test video 2
- `demo/test_1.mp4` - Test video 1
- `demo/test_processed.mp4` - Processed test video

**Other Files:**
- `demo/ravinotes` - Developer scratch notes
- `demo/temp_reqs.txt` - Temporary requirements dump

---

## New Directories

### 1. measurement/ Directory

**Location:** `measurement/`
**Purpose:** Standalone measurement utilities
**Total Size:** ~120KB

**Contents:**

| File | Size | Purpose |
|------|------|---------|
| frame_data_storage.py | 11,806 bytes | Data storage module |
| limb_measurement.py | 51,873 bytes | Main measurement script |
| model_mediapipe.py | 9,138 bytes | MediaPipe wrapper |
| smpl_data_loader.py | 8,011 bytes | SMPL data utilities |
| visualizer_3d_matplotlib.py | 15,783 bytes | 3D visualization |
| visualizer_3d.py | 19,979 bytes | Open3D-based visualizer |
| __pycache__/ | - | Python cache |

**Note:** This appears to be an earlier standalone version before integration into demo/. Some files are duplicates with slight differences.

**Key Differences from demo/ version:**
- May have different import paths
- Possibly earlier development versions
- Alternative visualization approach (Open3D in visualizer_3d.py)

---

### 2. vision/ Directory

**Location:** `vision/`
**Type:** Complete torchvision repository clone
**Purpose:** Custom torchvision build for Jetson

**Structure:**
```
vision/
├── android/          # Android support
├── build/            # Build artifacts
├── cmake/            # CMake build scripts
├── dist/             # Built distributions
├── docs/             # Documentation
├── examples/         # Usage examples
├── gallery/          # Model gallery
├── ios/              # iOS support
├── packaging/        # Package configuration
├── scripts/          # Build/test scripts
├── test/             # Test suite
├── torchvision/      # Main library
├── CMakeLists.txt
├── setup.py
└── ...
```

**Built Packages in dist/:**
- Likely contains wheel files for Jetson installation

**Reason for Inclusion:**
- Custom build for Jetson compatibility
- Version-specific requirements
- Local development and testing

---

### 3. Other New Files

#### lib/nms/gpu_nms.cpp
**Purpose:** GPU-accelerated Non-Maximum Suppression
**Type:** C++ source file
**Use:** Person detection post-processing

#### poc-usecase
**Purpose:** Developer notes and command history
**Type:** Text file
**Contents:**
- Conda environment info (frfd01)
- Test command examples
- Development workflow notes
- Build instructions

#### output.avi
**Purpose:** Video output sample
**Type:** Video file
**Size:** Not specified

---

## Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    HRNet Pose Estimation System              │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │  Input  │          │ Models  │          │ Output  │
   │ Sources │          │         │          │         │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                     │                     │
        │                     │                     │
   ┌────▼────────────┐   ┌───▼──────────┐    ┌────▼─────────┐
   │ • RealSense D455│   │ • HRNet      │    │ • 2D Display │
   │ • RGB Webcam    │   │ • MediaPipe  │    │ • 3D Viewer  │
   │ • Video File    │   │ • Faster RCNN│    │ • JSON Data  │
   │ • API Stream    │   │              │    │ • API Stream │
   └─────────────────┘   └──────────────┘    └──────────────┘
                              │
                    ┌─────────┼─────────┐
                    │                   │
              ┌─────▼──────┐     ┌─────▼──────┐
              │ Processing │     │   Storage  │
              └─────┬──────┘     └─────┬──────┘
                    │                   │
         ┌──────────┼──────────┐       │
         │          │          │       │
    ┌────▼───┐ ┌───▼────┐ ┌──▼───┐ ┌─▼────────┐
    │Person  │ │ Pose   │ │ Limb │ │  Frame   │
    │Detect  │ │ Estimate│ │Measure│ │ Storage │
    └────────┘ └────────┘ └──────┘ └──────────┘
```

### Data Flow

```
Camera/Video Input
       │
       ▼
Frame Capture (RGB + Depth)
       │
       ▼
Person Detection (Faster R-CNN)
       │
       ▼
Pose Estimation (HRNet/MediaPipe)
       │
       ├──► 2D Keypoints (x, y, confidence)
       │
       ▼
Depth Processing (if available)
       │
       ├──► 3D World Coordinates (x, y, z)
       │
       ▼
Limb Measurement
       │
       ├──► 3D Euclidean Distance → cm
       │
       ▼
Visualization Pipeline
       │
       ├──► 2D Skeleton Overlay
       ├──► 3D Skeleton Viewer
       └──► Measurement Annotations
       │
       ▼
Data Export
       │
       ├──► JSON Frame Data
       ├──► Session Summary
       └──► SMPL-ready Format
```

### Component Integration

```
┌─────────────────────────────────────────────────────┐
│                  PoseEstimator                      │
│                                                     │
│  ┌──────────────┐         ┌──────────────┐        │
│  │ Model Loader │         │Camera Manager│        │
│  │  • HRNet     │         │ • RealSense  │        │
│  │  • MediaPipe │         │ • RGB        │        │
│  └──────┬───────┘         └──────┬───────┘        │
│         │                         │                │
│  ┌──────▼─────────────────────────▼──────┐        │
│  │      Frame Processing Pipeline         │        │
│  │  1. Capture                            │        │
│  │  2. Detect                             │        │
│  │  3. Estimate                           │        │
│  │  4. Measure                            │        │
│  └──────┬─────────────────────────────────┘        │
│         │                                           │
│  ┌──────▼───────┐  ┌──────────┐  ┌──────────┐    │
│  │ Visualizer3D │  │  Storage │  │SMPL Render│    │
│  └──────────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## Feature Enhancements

### 1. Multi-Model Architecture

**Before:** Only HRNet support
**After:** MediaPipe + HRNet with unified interface

**Benefits:**
- Speed vs accuracy tradeoff
- Platform flexibility (CPU vs GPU)
- Application-specific optimization

**Implementation:**
```python
pose_estimator = PoseEstimator(model_type='mediapipe')  # or 'hrnet'
```

**Model Comparison:**

| Feature | MediaPipe | HRNet |
|---------|-----------|-------|
| Speed | ~30 FPS | ~15 FPS |
| Accuracy | Good | Excellent |
| Keypoints | 33 (full body) | 17 (COCO) |
| Hardware | CPU/GPU | GPU recommended |
| Latency | Low (~33ms) | Medium (~67ms) |
| Use Case | Real-time apps | High accuracy needs |

---

### 2. Real-World Limb Measurements

**New Capability:** 3D limb length calculation in centimeters

**Method:**
1. Capture aligned RGB + Depth frames
2. Extract 2D keypoints from pose model
3. Query depth values at keypoint locations
4. Convert to 3D world coordinates using camera intrinsics
5. Calculate 3D Euclidean distance
6. Smooth with 30-frame moving average

**Accuracy Factors:**
- Depth camera quality (RealSense D455: ±2mm @ 2m)
- Camera calibration (intrinsics)
- Keypoint detection confidence
- Subject distance (optimal: 1-3 meters)

**Measured Limbs:**
1. Right Upper Arm (shoulder → elbow)
2. Right Lower Arm (elbow → wrist)
3. Left Upper Arm
4. Left Lower Arm
5. Right Upper Leg (hip → knee)
6. Right Lower Leg (knee → ankle)
7. Left Upper Leg
8. Left Lower Leg

**Measurement Smoothing:**
```python
# 30-frame buffer
self.buffer_size = 30
self.measurement_buffer = []

# Average calculation
measurements = [frame.get(limb_name, 0) for frame in buffer]
smoothed_value = sum(measurements) / len(measurements)
```

---

### 3. API Service Layer

**New Capability:** FastAPI-based REST and WebSocket endpoints

**HTTP API:**
```
POST /process-frame
Content-Type: image/jpeg

Request: Raw JPEG bytes
Response: Processed JPEG with skeleton
```

**WebSocket API:**
```
WS /ws-stream

Send: JPEG frame (binary)
Receive: Processed JPEG frame (binary)
Continuous streaming
```

**Use Cases:**
- Remote pose estimation service
- Web application integration
- Mobile app backend
- Edge device deployment
- Multi-client streaming

**Performance:**
- Model loaded once on startup
- GPU memory reused across requests
- Concurrent WebSocket connections supported

---

### 4. 3D Visualization

**New Capability:** Real-time 3D skeleton rendering

**Backend:** Matplotlib (Jetson-compatible, no OpenGL/GLX issues)

**Features:**
1. **Auto-Format Detection**
   - COCO-17 or MediaPipe-33
   - Automatic bone mapping

2. **Coordinate System**
   - Normalized coordinates
   - Auto-centering on skeleton
   - Z-axis: confidence/depth

3. **Visual Effects**
   - Color-coded bones (12 colors)
   - Confidence-based fading
   - Measurement annotations
   - FPS display

4. **Interactive View**
   - Rotate, zoom, pan
   - Real-time updates
   - Measurement labels

**Technical Implementation:**
```python
# Initialize
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Update in real-time
for each frame:
    update_skeleton_geometry(keypoints)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
```

---

### 5. Data Pipeline for SMPL

**New Capability:** Comprehensive frame data storage

**Purpose:** Feed 3D human body reconstruction pipelines (SMPL, SMPL-X)

**Stored Data Per Frame:**
- 2D keypoints (pixel coordinates)
- Confidence scores
- 3D world coordinates
- Depth values per keypoint
- Limb measurements
- Detection quality metrics
- Timestamp

**Session Metadata:**
- Model type used
- Camera type and parameters
- Depth scale
- Frame dimensions
- Total frame count

**Export Formats:**

1. **Full Session JSON:**
```json
{
  "session_info": {...},
  "frames": [
    {
      "frame_id": 0,
      "timestamp_ms": 1234567890,
      "keypoints": {...},
      "depth_data": {...},
      "limb_measurements": {...},
      "detection_quality": {...}
    },
    ...
  ]
}
```

2. **Summary Statistics:**
```json
{
  "session_info": {...},
  "frame_statistics": {
    "total_frames": 1000,
    "valid_frames": 987,
    "detection_rate": 0.987,
    "avg_confidence": 0.85
  },
  "limb_statistics": {
    "right_upper_arm": {
      "avg": 28.5,
      "std": 0.8,
      "min": 26.2,
      "max": 30.1,
      "count": 950
    },
    ...
  }
}
```

3. **SMPL-Ready Format:**
```python
smpl_data = storage.get_frames_for_smpl(frame_range=(0, 100))
# Returns list of frames with only SMPL-relevant data
```

---

### 6. Platform Optimization

**Target Platform:** Jetson AGX Orin with JetPack 5.1.2

**Optimizations Applied:**

1. **CUDA Memory Management**
```python
if PLATFORM == "jetson":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    torch.cuda.empty_cache()
```

2. **Model Complexity Adjustment**
```python
# MediaPipe on Jetson
model_complexity = 1  # vs 2 on desktop
min_detection_confidence = 0.7  # vs 0.5
```

3. **Resolution Optimization**
```python
# Jetson
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Desktop
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

4. **V4L2 Backend**
```python
cap = cv2.VideoCapture(DEVICE_PATH, cv2.CAP_V4L2)
```

**Performance Results:**
- MediaPipe: ~30 FPS (Jetson)
- HRNet: ~15 FPS (Jetson)
- Memory usage: <4GB GPU
- CPU usage: 40-60%

---

## Dependencies

### New Dependencies Added

#### 1. FastAPI (0.122.0)
**Purpose:** Web framework for API services
**Features:**
- Automatic API documentation
- WebSocket support
- Async request handling
- Type validation

**Installation:**
```bash
pip install fastapi==0.122.0
```

**Usage in Project:**
- `demo/demo-api.py`: REST and WebSocket endpoints

---

#### 2. Uvicorn (0.33.0)
**Purpose:** ASGI server for FastAPI
**Features:**
- High performance
- Async I/O
- WebSocket support
- Auto-reload (development)

**Installation:**
```bash
pip install uvicorn==0.33.0
```

**Usage:**
```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

#### 3. MediaPipe
**Purpose:** Pose estimation model
**Features:**
- 33 keypoint detection
- CPU/GPU support
- High speed (~30 FPS)
- Cross-platform

**Installation:**
```bash
pip install mediapipe
```

**Usage in Project:**
- `demo/model_mediapipe.py`: Model wrapper
- `demo/limb_measurement.py`: Integration

---

#### 4. PyRealSense2
**Purpose:** Intel RealSense camera SDK
**Features:**
- Depth stream access
- RGB stream access
- Stream alignment
- Camera calibration data

**Installation:**
```bash
pip install pyrealsense2
```

**Usage in Project:**
- `demo/limb_measurement.py`: Camera initialization and frame capture

---

### Complete Dependency List

```txt
# Original dependencies
EasyDict==1.7
shapely==1.6.4
Cython
scipy
opencv-contrib-python
matplotlib
json_tricks
scikit-image
yacs>=0.1.5
tensorboardX==1.6

# Updated/Added
opencv-python              # Updated: removed version pin
numpy==1.24.4             # Added: version fix
yacs @ file://...         # Added: conda build
uvicorn==0.33.0          # Added: ASGI server
fastapi==0.122.0         # Added: web framework

# Implicit (not in requirements.txt)
mediapipe                 # Required for MediaPipe model
pyrealsense2             # Required for RealSense camera
torch                    # Existing (HRNet dependency)
torchvision              # Existing (custom build in vision/)
```

---

## Usage Guide

### Basic Usage Scenarios

#### Scenario 1: Quick Pose Estimation (MediaPipe)

**Use Case:** Fast real-time pose detection for demos

```bash
cd /path/to/HRnet_0.1
python demo/limb_measurement.py
```

**Expected Output:**
- Window with pose skeleton overlay
- FPS: ~30
- Model: MediaPipe
- Camera: Auto-detected (RealSense or RGB)

---

#### Scenario 2: High Accuracy Pose (HRNet)

**Use Case:** Research or high-quality applications

```bash
cd /path/to/HRnet_0.1
python demo/limb_measurement.py --model hrnet
```

**Expected Output:**
- Window with pose skeleton overlay
- FPS: ~15
- Model: HRNet
- Keypoints: 17 COCO format

---

#### Scenario 3: Limb Measurement with RealSense

**Use Case:** Physiotherapy, fitness tracking

```bash
python demo/limb_measurement.py --model mediapipe --camera realsense
```

**Expected Output:**
- Pose skeleton with measurements on each limb
- Real-world measurements in centimeters
- Depth-based 3D calculations
- Smoothed measurements (30-frame average)

---

#### Scenario 4: Video File Processing

**Use Case:** Offline analysis, batch processing

```bash
python demo/limb_measurement.py \
  --input test_video.mp4 \
  --output processed_video.mp4 \
  --model hrnet
```

**Expected Output:**
- Processed video saved to file
- Frame-by-frame pose detection
- No real-time display required

---

#### Scenario 5: Data Collection for SMPL

**Use Case:** 3D reconstruction, dataset creation

```bash
python demo/limb_measurement.py \
  --model hrnet \
  --camera realsense \
  --save-data \
  --session-name training_session_01 \
  --3d
```

**Expected Output:**
- Real-time pose estimation
- 3D skeleton visualization window
- Data saved to: `measurement/data/training_session_01.json`
- Summary: `measurement/data/training_session_01_summary.json`
- Frame count: All frames with metadata

---

#### Scenario 6: API Server for Remote Processing

**Use Case:** Web apps, mobile backends, distributed systems

```bash
# Start server
python demo/demo-api.py

# Server runs on: http://0.0.0.0:8000
```

**Client Usage:**

1. **HTTP Single Frame:**
```bash
curl -X POST http://localhost:8000/process-frame \
  --data-binary @image.jpg \
  --output processed.jpg \
  -H "Content-Type: image/jpeg"
```

2. **WebSocket Stream (Python client):**
```python
import asyncio
import websockets
import cv2

async def stream_frames():
    uri = "ws://localhost:8000/ws-stream"
    async with websockets.connect(uri) as ws:
        # Capture from camera
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            _, jpeg = cv2.imencode('.jpg', frame)

            # Send frame
            await ws.send(jpeg.tobytes())

            # Receive processed frame
            processed_jpeg = await ws.recv()
            # Display or save processed frame
```

---

#### Scenario 7: Multi-Camera Setup

**Use Case:** Multiple camera angles

```bash
# Terminal 1: Camera 0
python demo/limb_measurement.py --cam_id 0

# Terminal 2: Camera 1
python demo/limb_measurement.py --cam_id 1
```

---

#### Scenario 8: Quick Body Capture

**Use Case:** Capture reference image for pose estimation

```bash
python demo/capture_body.py
```

**Instructions:**
1. Stand 6 feet away
2. Press 'c' to capture
3. Image saved as `person.jpg`
4. Press 'q' to quit

---

### Advanced Usage

#### Custom Session with All Features

```bash
python demo/limb_measurement.py \
  --model hrnet \
  --camera realsense \
  --3d \
  --save-data \
  --session-name production_test_01 \
  --output output_video.mp4
```

**Features Enabled:**
- HRNet pose model (accurate)
- RealSense depth camera
- 3D skeleton visualization
- Comprehensive data logging
- Video output recording

**Expected Outputs:**
1. Real-time display window (2D pose)
2. 3D skeleton window (matplotlib)
3. Video file: `output_video.mp4`
4. Session data: `measurement/data/production_test_01.json`
5. Summary: `measurement/data/production_test_01_summary.json`
6. Periodic logs: `logs/measurements_*.json`

---

## API Documentation

### REST API

#### Endpoint: /process-frame

**Method:** POST
**URL:** `http://host:8000/process-frame`
**Content-Type:** `image/jpeg`

**Request:**
- **Body:** Raw JPEG image bytes
- **Headers:** `Content-Type: image/jpeg`

**Response:**
- **Status:** 200 OK
- **Content-Type:** `image/jpeg`
- **Body:** Processed JPEG with skeleton overlay

**Error Responses:**
- 400 Bad Request: No image data or invalid image
- 500 Internal Server Error: Processing failed

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/process-frame \
  --data-binary @input.jpg \
  --output output.jpg \
  -H "Content-Type: image/jpeg"
```

**Example (Python):**
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process-frame',
        data=f.read(),
        headers={'Content-Type': 'image/jpeg'}
    )

with open('processed.jpg', 'wb') as f:
    f.write(response.content)
```

---

### WebSocket API

#### Endpoint: /ws-stream

**Method:** WebSocket
**URL:** `ws://host:8000/ws-stream`

**Protocol:**
1. Client connects to WebSocket
2. Server accepts connection
3. Client sends JPEG frame (binary)
4. Server processes and sends back processed JPEG (binary)
5. Repeat steps 3-4 for continuous streaming
6. Client disconnects when done

**Message Format:**
- **Client → Server:** Binary JPEG bytes
- **Server → Client:** Binary JPEG bytes

**Error Handling:**
- Invalid frames are skipped (no response)
- Connection closes on error

**Example (Python):**
```python
import asyncio
import websockets
import cv2
import numpy as np

async def process_video_stream():
    uri = "ws://localhost:8000/ws-stream"

    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Encode frame to JPEG
                _, jpeg_frame = cv2.imencode('.jpg', frame)

                # Send to server
                await websocket.send(jpeg_frame.tobytes())

                # Receive processed frame
                processed_jpeg = await websocket.recv()

                # Decode and display
                nparr = np.frombuffer(processed_jpeg, np.uint8)
                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                cv2.imshow('Processed', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

# Run
asyncio.run(process_video_stream())
```

**Example (JavaScript/Browser):**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws-stream');
ws.binaryType = 'arraybuffer';

// Send frame
function sendFrame(blob) {
    ws.send(blob);
}

// Receive processed frame
ws.onmessage = function(event) {
    const blob = new Blob([event.data], {type: 'image/jpeg'});
    const url = URL.createObjectURL(blob);
    document.getElementById('output').src = url;
};

// Capture from video element
async function streamCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({video: true});
    const video = document.createElement('video');
    video.srcObject = stream;
    video.play();

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    setInterval(() => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            sendFrame(blob);
        }, 'image/jpeg', 0.8);
    }, 33); // ~30 FPS
}

streamCamera();
```

---

### API Server Configuration

**Start Server:**
```bash
python demo/demo-api.py
```

**Server Settings:**
```python
host = "0.0.0.0"  # Listen on all interfaces
port = 8000
reload = False    # Set True for development
```

**Model Loading:**
- Models loaded once on startup
- Faster R-CNN + HRNet loaded into GPU memory
- Reduces per-request latency

**Performance:**
- Single frame processing: ~100-200ms (GPU)
- WebSocket streaming: 5-10 FPS (depends on network)
- Concurrent connections: Supported (shared model)

---

## Deployment Notes

### Development Environment

**Platform:** Ubuntu Linux
**Hardware:** Development workstation with NVIDIA GPU
**Python:** 3.7+ (conda environment: frfd01)

**Setup:**
```bash
# Clone repository
git clone <repository-url>
cd HRnet_0.1

# Create conda environment
conda create -n frfd01 python=3.8
conda activate frfd01

# Install dependencies
pip install -r requirements.txt

# Download HRNet models
# Place in: models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth
```

---

### Production Environment (Jetson AGX Orin)

**Platform:** Jetson AGX Orin
**OS:** Ubuntu 20.04 (L4T)
**JetPack:** 5.1.2
**CUDA:** 11.4
**Python:** 3.8

**Setup Steps:**

1. **Install JetPack 5.1.2**
```bash
sudo apt update
sudo apt install nvidia-jetpack
```

2. **Install Conda (Archiconda for ARM)**
```bash
wget https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh
bash Archiconda3-0.2.3-Linux-aarch64.sh
source ~/archiconda3/bin/activate
```

3. **Create Environment**
```bash
conda create -n frfd01 python=3.8
conda activate frfd01
```

4. **Install PyTorch (Jetson)**
```bash
# Use NVIDIA's pre-built PyTorch for Jetson
wget https://nvidia.box.com/shared/static/...pytorch-1.12.0-...whl
pip install pytorch-1.12.0-...whl
```

5. **Install Dependencies**
```bash
pip install -r requirements.txt
```

6. **Install RealSense SDK**
```bash
sudo apt install librealsense2-dev
pip install pyrealsense2
```

7. **Build Custom torchvision**
```bash
cd vision/
python setup.py install
cd ..
```

8. **Verify Installation**
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import mediapipe; print('MediaPipe OK')"
python -c "import pyrealsense2 as rs; print('RealSense OK')"
```

---

### Performance Optimization (Jetson)

**1. CUDA Memory Management**
```bash
# Set before running
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**2. Power Mode**
```bash
# Max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

**3. Swap Space (if needed)**
```bash
# Create 8GB swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**4. Model Selection**
- Use MediaPipe for real-time applications (30 FPS)
- Use HRNet when accuracy > speed (15 FPS)

---

### Camera Setup

#### RealSense D455

**Connection:**
- USB 3.0+ port required
- Use provided cable (3 meters max)

**Verification:**
```bash
rs-enumerate-devices
```

**Expected Output:**
```
Device 0:
  Name: Intel RealSense D455
  Serial Number: xxxxxxxxxx
  Firmware Version: 5.13.0.50
```

**Troubleshooting:**
- Update firmware: Use Intel RealSense Viewer
- USB issues: Try different ports, check USB 3.0
- Permissions: Add user to `video` group

---

#### RGB Webcam

**Connection:**
- USB 2.0/3.0 webcam
- V4L2 compatible

**List Cameras:**
```bash
v4l2-ctl --list-devices
```

**Test Camera:**
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

---

### File Paths and Configurations

**Model Files:**
```
HRnet_0.1/
├── models/
│   └── pytorch/
│       └── pose_coco/
│           └── pose_hrnet_w32_384x288.pth  # Download required
```

**Configuration:**
```
HRnet_0.1/
├── demo/
│   └── inference-config.yaml  # HRNet model config
```

**Output Directories:**
```
HRnet_0.1/
├── logs/                      # Measurement logs
├── measurement/
│   └── data/                  # Session data files
└── output/                    # Video outputs (create if needed)
```

---

### Running as Service (Production)

**Systemd Service File:** `/etc/systemd/system/pose-api.service`

```ini
[Unit]
Description=Pose Estimation API Service
After=network.target

[Service]
Type=simple
User=raviappala
WorkingDirectory=/media/raviappala/edgeextvol2/jetson-inference/3dpose/HRnet_0.1
Environment="PATH=/home/raviappala/archiconda3/envs/frfd01/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/raviappala/archiconda3/envs/frfd01/bin/python demo/demo-api.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and Start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable pose-api
sudo systemctl start pose-api
sudo systemctl status pose-api
```

**View Logs:**
```bash
sudo journalctl -u pose-api -f
```

---

## File Reference

### Complete File Listing

```
HRnet_0.1/
│
├── demo/
│   ├── capture_body.py              # NEW: Camera capture utility
│   ├── demo-api.py                  # NEW: FastAPI server
│   ├── demo.py                      # MODIFIED: Added --cam_id
│   ├── frame_data_storage.py        # NEW: Data storage module
│   ├── limb_measurement.py          # NEW: Main measurement system
│   ├── limb_usage.md                # NEW: Usage documentation
│   ├── model_mediapipe.py           # NEW: MediaPipe wrapper
│   ├── visualizer_3d_matplotlib.py  # NEW: 3D visualization
│   ├── output_processed.jpg         # NEW: Test output
│   ├── person.jpg                   # NEW: Test image
│   ├── person_processed.jpg         # NEW: Test processed
│   ├── test2.mp4                    # NEW: Test video
│   ├── test_1.mp4                   # NEW: Test video
│   ├── test_processed.mp4           # NEW: Processed video
│   ├── ravinotes                    # NEW: Developer notes
│   └── temp_reqs.txt                # NEW: Temp requirements
│
├── measurement/                     # NEW DIRECTORY
│   ├── frame_data_storage.py        # Data storage
│   ├── limb_measurement.py          # Measurement script
│   ├── model_mediapipe.py           # MediaPipe model
│   ├── smpl_data_loader.py          # SMPL utilities
│   ├── visualizer_3d_matplotlib.py  # 3D viz (matplotlib)
│   ├── visualizer_3d.py             # 3D viz (Open3D)
│   └── __pycache__/                 # Python cache
│
├── vision/                          # NEW DIRECTORY
│   ├── android/                     # Android support
│   ├── build/                       # Build artifacts
│   ├── cmake/                       # CMake scripts
│   ├── dist/                        # Built packages
│   ├── docs/                        # Documentation
│   ├── examples/                    # Examples
│   ├── gallery/                     # Model gallery
│   ├── ios/                         # iOS support
│   ├── packaging/                   # Package config
│   ├── scripts/                     # Build scripts
│   ├── test/                        # Tests
│   ├── torchvision/                 # Main library
│   ├── CMakeLists.txt
│   ├── setup.py
│   └── ...
│
├── lib/
│   └── nms/
│       └── gpu_nms.cpp              # NEW: GPU NMS
│
├── requirements.txt                 # MODIFIED: New dependencies
├── poc-usecase                      # NEW: Developer notes
├── output.avi                       # NEW: Video output
├── CHANGE_DOCUMENTATION.md          # THIS FILE
│
└── [Original HRNet files...]
```

---

### File Size Summary

| Category | Files | Total Size |
|----------|-------|------------|
| Modified | 2 | ~5 KB |
| New (demo/) | 15+ | ~60 KB |
| New (measurement/) | 7 | ~120 KB |
| New (vision/) | Full repo | ~200 MB |
| Documentation | 2 | ~20 KB |
| **Total New/Modified** | **25+** | **~220 MB** |

---

### Critical Files for Operation

**Minimum Required:**
1. `demo/limb_measurement.py` - Main application
2. `demo/model_mediapipe.py` - MediaPipe support
3. `demo/frame_data_storage.py` - Data storage
4. `demo/visualizer_3d_matplotlib.py` - 3D visualization
5. `requirements.txt` - Dependencies
6. `models/pytorch/pose_coco/*.pth` - Model weights

**Optional but Recommended:**
7. `demo/demo-api.py` - API service
8. `demo/capture_body.py` - Capture utility
9. `demo/limb_usage.md` - Documentation
10. `measurement/*` - Alternative version

---

## Migration to New Repository

### Recommended Steps

**1. Create New Repository**
```bash
# On GitHub/GitLab, create new repo
# Then locally:
cd /path/to/HRnet_0.1
git init
git remote add origin <new-repo-url>
```

**2. Initial Commit (Original State)**
```bash
# First, commit only original HRNet files
# Remove new additions temporarily
git add [original files only]
git commit -m "Initial commit: Original HRNet v0.1"
git tag v0.1-original
```

**3. Commit New Features**
```bash
# Add modified files
git add demo/demo.py requirements.txt
git commit -m "feat: Add camera device selection and update dependencies"

# Add new core features
git add demo/limb_measurement.py demo/model_mediapipe.py \
        demo/frame_data_storage.py demo/visualizer_3d_matplotlib.py
git commit -m "feat: Add comprehensive limb measurement system with dual model support"

# Add API service
git add demo/demo-api.py
git commit -m "feat: Add FastAPI REST and WebSocket server"

# Add utilities
git add demo/capture_body.py demo/limb_usage.md
git commit -m "docs: Add capture utility and usage documentation"

# Add measurement directory
git add measurement/
git commit -m "feat: Add standalone measurement module"

# Add vision (if needed)
git add vision/
git commit -m "build: Add custom torchvision build for Jetson"

# Add documentation
git add CHANGE_DOCUMENTATION.md
git commit -m "docs: Add comprehensive change documentation"
```

**4. Tag Release**
```bash
git tag -a v1.0-enhanced -m "Enhanced HRNet with measurements, API, and 3D viz"
```

**5. Push to Remote**
```bash
git push -u origin main
git push --tags
```

---

### .gitignore Recommendations

**Create `.gitignore`:**
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# Conda
*.conda
*.tar.bz2

# Model files (large, store separately)
*.pth
*.onnx
*.pb

# Data files
logs/*.json
measurement/data/*.json
measurement/data/*.npz

# Output files
output/*.mp4
output/*.avi
*.jpg
*.png
*.mp4
*.avi

# Exclude specific test files
!demo/person.jpg
!demo/person_processed.jpg

# Build artifacts
build/
dist/
*.egg-info/
vision/build/
vision/dist/

# System
.DS_Store
Thumbs.db
```

---

### README for New Repository

**Create `README.md`:**
```markdown
# HRNet Enhanced Pose Estimation System

Comprehensive 3D pose estimation and limb measurement system with multi-model support, built on HRNet.

## Features

- **Dual Model Support**: MediaPipe (fast) and HRNet (accurate)
- **Real-world Measurements**: 3D limb lengths in centimeters
- **API Services**: REST and WebSocket endpoints
- **3D Visualization**: Real-time skeleton rendering
- **Data Pipeline**: SMPL-ready frame data export
- **Platform Optimized**: Jetson AGX Orin support

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with MediaPipe (fast)
python demo/limb_measurement.py

# Run with HRNet (accurate)
python demo/limb_measurement.py --model hrnet --camera realsense
```

## Documentation

See [CHANGE_DOCUMENTATION.md](CHANGE_DOCUMENTATION.md) for complete details.

## License

[Your License]
```

---

## Conclusion

This documentation captures all changes made to the HRNet pose estimation repository. The system has evolved significantly with:

- **2 modified files**
- **15+ new files** in demo directory
- **3 new directories** (measurement, vision, lib/nms)
- **4 major new dependencies**
- **~220 MB** of new code and assets

### Key Achievements

1. ✅ Multi-model architecture (MediaPipe + HRNet)
2. ✅ Real-world limb measurements (3D, depth-based)
3. ✅ Production API service (REST + WebSocket)
4. ✅ 3D visualization (Jetson-compatible)
5. ✅ Comprehensive data pipeline (SMPL-ready)
6. ✅ Platform optimization (Jetson AGX Orin)

### Recommended Next Steps

1. Review this documentation thoroughly
2. Test all features in development environment
3. Create new git repository with clean history
4. Deploy to production (Jetson)
5. Monitor performance and optimize as needed
6. Continue development with proper version control

---

**Document prepared:** January 4, 2026
**For questions or clarifications, refer to code comments or contact the development team.**
