# HRNet Enhanced Pose Estimation System

Comprehensive 3D pose estimation and limb measurement system with multi-model support.

## Features

- 🚀 **Dual Model Support**: MediaPipe (fast, ~30 FPS) and HRNet (accurate, ~15 FPS)
- 📏 **Real-world Measurements**: 3D limb lengths in centimeters using depth camera
- 🌐 **API Services**: FastAPI REST and WebSocket endpoints
- 🎨 **3D Visualization**: Real-time skeleton rendering
- 💾 **Data Pipeline**: SMPL-ready frame data export
- ⚡ **Platform Optimized**: Jetson AGX Orin support

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/ravikumarappala/pose-hrnet-v10.git
cd pose-hrnet-v10
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Models

**Important:** Model files are excluded from this repository due to size. Download them separately:

#### Option A: Download from Official HRNet Repository
```bash
# Create models directory
mkdir -p models/pytorch/pose_coco

# Download HRNet W32 model (recommended)
wget https://drive.google.com/uc?export=download&id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38 \
  -O models/pytorch/pose_coco/pose_hrnet_w32_384x288.pth
```

#### Option B: Manual Download
1. Visit: https://github.com/HRNet/HRNet-Human-Pose-Estimation
2. Download model weights from their model zoo
3. Place in `models/pytorch/pose_coco/` directory

**Required Model Files:**
- `pose_hrnet_w32_384x288.pth` (110 MB) - For HRNet model
- Optional: Additional models for different configurations

**Model Directory Structure:**
```
models/
└── pytorch/
    └── pose_coco/
        ├── pose_hrnet_w32_384x288.pth   (required for HRNet)
        ├── pose_hrnet_w48_384x288.pth   (optional, higher accuracy)
        └── ...
```

## Quick Start

```bash
# Run with MediaPipe (no model download needed)
python demo/limb_measurement.py

# Run with HRNet (requires model download above)
python demo/limb_measurement.py --model hrnet --camera realsense

# Enable 3D visualization
python demo/limb_measurement.py --3d
```

## Documentation

- **[CHANGE_DOCUMENTATION.md](CHANGE_DOCUMENTATION.md)** - Complete technical documentation
- **[CHANGE_DOCUMENTATION.docx](CHANGE_DOCUMENTATION.docx)** - Word format
- **[demo/limb_usage.md](demo/limb_usage.md)** - Detailed usage guide

## Use Cases

1. Physiotherapy - Track joint angles and limb lengths
2. Fitness Tracking - Body metrics and form analysis
3. Motion Capture - Real-time pose data for animation
4. Research - Dataset collection for 3D reconstruction
5. Edge Deployment - API service on Jetson platforms

## Command-Line Options

```bash
python demo/limb_measurement.py [OPTIONS]

  --model {mediapipe,hrnet}     Pose model (default: mediapipe)
  --camera {auto,realsense,rgb} Camera type (default: auto)
  --input PATH                  Video file input
  --output PATH                 Video file output
  --save-data                   Enable data storage
  --session-name NAME           Session identifier
  --3d                         Enable 3D visualization
```

## API Service

```bash
# Start server
python demo/demo-api.py

# HTTP endpoint
curl -X POST http://localhost:8000/process-frame \
  --data-binary @image.jpg --output processed.jpg

# WebSocket: ws://localhost:8000/ws-stream
```

## Performance (Jetson AGX Orin)

- MediaPipe: ~30 FPS, <4GB GPU
- HRNet: ~15 FPS, <6GB GPU

See [CHANGE_DOCUMENTATION.md](CHANGE_DOCUMENTATION.md) for complete details.

## Additional Resources

- **Original HRNet:** https://github.com/HRNet/HRNet-Human-Pose-Estimation
- **MediaPipe:** https://google.github.io/mediapipe/
- **RealSense SDK:** https://github.com/IntelRealSense/librealsense

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite the original HRNet paper:
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}
```
