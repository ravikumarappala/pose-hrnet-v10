# HRNet Enhanced Pose Estimation System

Comprehensive 3D pose estimation and limb measurement system with multi-model support.

## Features

- 🚀 **Dual Model Support**: MediaPipe (fast, ~30 FPS) and HRNet (accurate, ~15 FPS)
- 📏 **Real-world Measurements**: 3D limb lengths in centimeters using depth camera
- 🌐 **API Services**: FastAPI REST and WebSocket endpoints
- 🎨 **3D Visualization**: Real-time skeleton rendering
- 💾 **Data Pipeline**: SMPL-ready frame data export
- ⚡ **Platform Optimized**: Jetson AGX Orin support

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with MediaPipe (fast)
python demo/limb_measurement.py

# Run with HRNet (accurate)
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
# pose-hrnet-v10
# pose-hrnet-v10
