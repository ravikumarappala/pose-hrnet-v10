# Limb Measurement Usage Guide

## Overview
This script performs real-time pose estimation and limb measurement with proper signal handling for safe shutdown.

## Features
- **Safe Shutdown**: Press Ctrl+C anytime to safely stop and save all data
- **Automatic Video Recording**: Both input and output videos are saved automatically
- **Measurement Export**: CSV files with frame-by-frame and summary statistics
- **Timestamped Output**: Each session creates a timestamped folder in `output/`

## Basic Usage

### Using RGB Camera (default)
```bash
python limb_measurement.py --model hrnet
```

### Using Video File Input
```bash
python limb_measurement.py --model hrnet --input /path/to/video.mp4
```

### Using RealSense Camera
```bash
python limb_measurement.py --model hrnet --camera realsense
```

## Command Line Options

- `--model` : Choose pose model (`mediapipe` or `hrnet`)
  - `mediapipe` - Faster, good for real-time
  - `hrnet` - More accurate, requires more processing

- `--camera` : Camera type (`auto`, `realsense`, or `rgb`)
  - `auto` - Auto-detect (default)
  - `realsense` - Force RealSense camera
  - `rgb` - Force RGB webcam

- `--input` : Path to input video file (optional)
  - If not provided, uses camera

- `--save-data` : Enable comprehensive frame data storage

- `--session-name` : Custom session name for data storage

- `--3d` : Enable real-time 3D skeleton visualization

- `--smpl` : Enable SMPL mesh rendering

## Output Structure

When you run the program, it creates a folder structure like:
```
output/
  └── 20260104_143022/          # Timestamp folder
      ├── 20260104_143022_input.avi      # Original video
      ├── 20260104_143022_output.avi     # Video with pose overlay
      ├── measurements_table.csv          # Frame-by-frame data
      ├── measurements_summary.csv        # Statistics
      └── README.txt                      # Session info
```

## Safe Shutdown

### Method 1: Press 'q' key
- While the video window is active, press 'q'
- All files will be saved automatically

### Method 2: Press Ctrl+C
- Works anytime, even if window is not active
- Signal handler ensures all files are saved
- Videos are properly closed
- Measurements are exported to CSV

## Troubleshooting

### Problem: Process still running after Ctrl+C
**Solution**: The new implementation includes proper signal handling. This should no longer occur.

### Problem: Files not saved
**Solution**:
- Make sure you exit using 'q' or Ctrl+C
- Check the `output/` folder for timestamped subdirectories
- Look for error messages in console

### Problem: Zombie processes in ps aux
**Solution**:
- New implementation includes proper cleanup
- Use `pkill -9 python` only as last resort

## Measurements Explained

The CSV files contain measurements for these limb segments:

| Limb | Description |
|------|-------------|
| `left_upper_arm` | Left shoulder to left elbow |
| `left_lower_arm` | Left elbow to left wrist |
| `right_upper_arm` | Right shoulder to right elbow |
| `right_lower_arm` | Right elbow to right wrist |
| `left_upper_leg` | Left hip to left knee |
| `left_lower_leg` | Left knee to left ankle |
| `right_upper_leg` | Right hip to right knee |
| `right_lower_leg` | Right knee to right ankle |

All measurements are in **centimeters (cm)**.

## Example Workflows

### Quick Test with Webcam
```bash
cd measurement
python limb_measurement.py --model mediapipe
# Press Ctrl+C when done
# Check output/ folder for results
```

### Process Video File with HRNet
```bash
python limb_measurement.py --model hrnet --input ../videos/test.mp4
# Let it process completely or press Ctrl+C
# Check output/YYYYMMDD_HHMMSS/ for results
```

### Real-time with RealSense and 3D Visualization
```bash
python limb_measurement.py --model hrnet --camera realsense --3d
```

## Notes

- First run may take time to download models
- HRNet requires more GPU/CPU than MediaPipe
- RealSense cameras provide depth data for more accurate measurements
- RGB cameras use estimated depth (less accurate)
- Output folder automatically created if it doesn't exist
