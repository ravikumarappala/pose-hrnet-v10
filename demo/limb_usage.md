# Limb Measurement System - Updated Usage Guide

setup instructions

Usage instructions

Output - diectory

Modification history:
date/ change



The limb measurement system has been integrated into the HRNet demo directory for seamless operation with both MediaPipe and HRNet models.

## Location
Files are now located in: `HRnet/demo/`

## Usage Examples

### Basic Usage (from HRnet directory)

#### Using MediaPipe (Default - Fast)
```bash
cd /path/to/HRnet
python demo/limb_measurement.py
```

#### Using HRNet (Accurate)
```bash
cd /path/to/HRnet  
python demo/limb_measurement.py --model hrnet
```

### Advanced Usage

#### HRNet with RealSense Camera
```bash
python demo/limb_measurement.py --model hrnet --camera realsense
```

#### MediaPipe with RGB Webcam
```bash
python demo/limb_measurement.py --model mediapipe --camera rgb --camera-id 0
```

#### HRNet with SMPL Rendering
```bash
python demo/limb_measurement.py --model hrnet --smpl
```

## Key Improvements

### ✅ **Working Directory Solution**
- **Fixed**: All files now run from HRNet directory where paths and configs work correctly
- **No more**: Path import errors or configuration issues

### ✅ **Direct Demo Integration**
- **Uses**: Actual HRNet demo functions (get_person_detection_boxes, get_pose_estimation_prediction)
- **Leverages**: Existing, tested HRNet infrastructure
- **Maintains**: All limb measurement functionality

### ✅ **Unified Interface**
- **Same**: Command-line arguments for both models
- **Same**: Measurement output format and logging
- **Same**: Camera support (RealSense + RGB)

## Model Comparison

| Model | Speed | Accuracy | Location | Keypoints |
|-------|-------|----------|----------|-----------|
| MediaPipe | Fast (~30 FPS) | Good | Local processing | 33 points |
| HRNet | Moderate (~15 FPS) | Excellent | Uses demo config | 17 points (COCO) |

## File Structure

```
HRnet/
├── demo/
│   ├── limb_measurement.py    # ✅ Main script (copied & simplified)
│   ├── model_mediapipe.py     # ✅ MediaPipe support  
│   ├── inference-config.yaml # ✅ HRNet config (existing)
│   ├── limb_usage.md         # ✅ This documentation
│   └── demo.py               # ✅ Original HRNet demo
├── lib/                      # ✅ All imports work
├── models/                   # ✅ All models accessible
└── ...
```

## Limb Measurements

Both models measure identical limbs with format-specific keypoint indices:

### MediaPipe (33 keypoints)
- right_upper_arm: (12, 14) 
- right_lower_arm: (14, 16)
- left_upper_arm: (11, 13)
- left_lower_arm: (13, 15)
- right_upper_leg: (24, 26)
- right_lower_leg: (26, 28)
- left_upper_leg: (23, 25)
- left_lower_leg: (25, 27)

### HRNet (17 keypoints - COCO)
- right_upper_arm: (6, 8)
- right_lower_arm: (8, 10) 
- left_upper_arm: (5, 7)
- left_lower_arm: (7, 9)
- right_upper_leg: (12, 14)
- right_lower_leg: (14, 16)
- left_upper_leg: (11, 13)
- left_lower_leg: (13, 15)

## Requirements

### For HRNet
- Run from HRNet directory: `cd /path/to/HRnet`
- Pre-trained models in `models/pytorch/pose_coco/`
- CUDA-capable GPU (recommended)
- All HRNet dependencies installed

### For MediaPipe  
- MediaPipe library: `pip install mediapipe`
- Works on CPU and GPU

## Troubleshooting

### Wrong Working Directory
❌ **Error**: Import/config failures
✅ **Solution**: Always run from HRNet directory
```bash
cd /path/to/HRnet
python demo/limb_measurement.py
```

### HRNet Model Not Found
❌ **Error**: Model file not found
✅ **Solution**: Check models exist in `models/pytorch/pose_coco/`

### Import Errors
❌ **Error**: Cannot import HRNet libs
✅ **Solution**: Ensure running from HRNet directory with _init_paths.py

## Success Indicators

✅ **Working correctly when you see**:
- "Successfully loaded [Model] model"
- Real-time pose detection and measurements
- Measurements displayed on limbs in video feed
- Logs saved to `logs/measurements_*.json`

## Performance Notes

- **Jetson AGX Orin**: Optimized for both models
- **HRNet**: ~15 FPS with high accuracy
- **MediaPipe**: ~30 FPS with good accuracy
- **RealSense**: Provides 3D depth for accurate measurements
