"""
Frame Data Storage Module for 3D SMPL Pipeline
==============================================
Comprehensive frame-by-frame data storage for pose estimation including
keypoints, depth data, and limb measurements for 3D reconstruction.

Author: Claude
Date: August 28, 2025
"""

import json
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class FrameDataStorage:
    """Handles comprehensive frame-by-frame data storage for 3D SMPL pipeline"""
    
    def __init__(self, output_dir="measurement/data", session_name=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session info
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_file = self.output_dir / f"{self.session_name}.json"
        
        # Data containers
        self.session_data = {
            "session_info": {},
            "frames": []
        }
        
        self.frame_count = 0
        self.current_session_info = {}
        
        logger.info(f"Initialized FrameDataStorage - Output: {self.session_file}")
    
    def initialize_session(self, model_type, camera_type, video_source=None, 
                          frame_dimensions=None, depth_scale=None):
        """Initialize session with metadata"""
        self.current_session_info = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_type": model_type,
            "camera_type": camera_type,
            "video_source": video_source or "camera",
            "frame_dimensions": frame_dimensions or [640, 480],
            "depth_scale": depth_scale or 0.001,
            "total_frames": 0
        }
        
        self.session_data["session_info"] = self.current_session_info
        logger.info(f"Session initialized: {model_type} with {camera_type}")
    
    def store_frame_data(self, keypoints, depth_frame, limb_measurements, 
                        detection_quality=None, timestamp_ms=None):
        """Store comprehensive data for a single frame"""
        
        if keypoints is None:
            logger.warning(f"No keypoints for frame {self.frame_count}")
            return
        
        # Ensure keypoints is numpy array
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        
        # Calculate 3D world coordinates if depth data available
        world_3d = []
        keypoint_depths = []
        depth_confidence = []
        
        if depth_frame is not None and np.any(depth_frame > 0):
            world_3d, keypoint_depths, depth_confidence = self._calculate_3d_coordinates(
                keypoints, depth_frame
            )
        
        # Determine keypoint format
        keypoint_format = "mediapipe_33" if len(keypoints) == 33 else "coco_17"
        
        # Calculate detection quality metrics
        if detection_quality is None:
            detection_quality = self._calculate_detection_quality(keypoints)
        
        # Create frame data structure
        frame_data = {
            "frame_id": self.frame_count,
            "timestamp_ms": timestamp_ms or int(datetime.now().timestamp() * 1000),
            "keypoints": {
                "raw_2d": keypoints[:, :2].tolist(),
                "confidence": keypoints[:, 2].tolist(),
                "world_3d": world_3d,
                "format": keypoint_format
            },
            "depth_data": {
                "keypoint_depths": keypoint_depths,
                "depth_confidence": depth_confidence,
                "has_valid_depth": len(keypoint_depths) > 0
            },
            "limb_measurements": dict(limb_measurements) if limb_measurements else {},
            "detection_quality": detection_quality
        }
        
        # Store frame data
        self.session_data["frames"].append(frame_data)
        self.frame_count += 1
        
        # Update session info
        self.session_data["session_info"]["total_frames"] = self.frame_count
        
        # Auto-save every 30 frames for safety
        if self.frame_count % 30 == 0:
            self._save_session_data()
            logger.debug(f"Auto-saved at frame {self.frame_count}")
    
    def _calculate_3d_coordinates(self, keypoints, depth_frame):
        """Calculate 3D world coordinates from 2D keypoints and depth"""
        world_3d = []
        keypoint_depths = []
        depth_confidence = []
        
        # Camera intrinsics (should be calibrated for actual use)
        fx = 600.0  # Focal length x
        fy = 600.0  # Focal length y
        cx = depth_frame.shape[1] / 2  # Principal point x
        cy = depth_frame.shape[0] / 2  # Principal point y
        depth_scale = self.current_session_info.get("depth_scale", 0.001)
        
        height, width = depth_frame.shape
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:  # Only process confident keypoints
                # Ensure coordinates are within bounds
                x_int, y_int = int(x), int(y)
                if 0 <= x_int < width and 0 <= y_int < height:
                    # Get depth with averaging for robustness
                    kernel_size = 5
                    half_kernel = kernel_size // 2
                    
                    y_min = max(0, y_int - half_kernel)
                    y_max = min(height, y_int + half_kernel + 1)
                    x_min = max(0, x_int - half_kernel)
                    x_max = min(width, x_int + half_kernel + 1)
                    
                    depth_area = depth_frame[y_min:y_max, x_min:x_max]
                    valid_depths = depth_area[depth_area > 0]
                    
                    if len(valid_depths) > 0:
                        # Use median depth for robustness
                        z = np.median(valid_depths) * depth_scale
                        
                        # Convert to 3D coordinates
                        world_x = (x - cx) * z / fx
                        world_y = (y - cy) * z / fy
                        world_z = z
                        
                        world_3d.append([world_x, world_y, world_z])
                        keypoint_depths.append(z)
                        depth_confidence.append(len(valid_depths) / (kernel_size * kernel_size))
                    else:
                        world_3d.append([0, 0, 0])
                        keypoint_depths.append(0)
                        depth_confidence.append(0)
                else:
                    world_3d.append([0, 0, 0])
                    keypoint_depths.append(0)
                    depth_confidence.append(0)
            else:
                world_3d.append([0, 0, 0])
                keypoint_depths.append(0)
                depth_confidence.append(0)
        
        return world_3d, keypoint_depths, depth_confidence
    
    def _calculate_detection_quality(self, keypoints):
        """Calculate detection quality metrics"""
        if len(keypoints) == 0:
            return {
                "pose_detected": False,
                "avg_confidence": 0.0,
                "valid_keypoints": 0,
                "detection_score": 0.0
            }
        
        confidences = keypoints[:, 2]
        valid_keypoints = np.sum(confidences > 0.3)
        avg_confidence = np.mean(confidences)
        
        # Calculate overall detection score
        detection_score = (valid_keypoints / len(keypoints)) * avg_confidence
        
        return {
            "pose_detected": valid_keypoints > 5,  # Need at least 5 confident keypoints
            "avg_confidence": float(avg_confidence),
            "valid_keypoints": int(valid_keypoints),
            "detection_score": float(detection_score)
        }
    
    def save_session(self):
        """Save current session data to file"""
        self._save_session_data()
        logger.info(f"Session saved: {self.session_file} ({self.frame_count} frames)")
        return str(self.session_file)
    
    def _save_session_data(self):
        """Internal method to save session data"""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def load_session(self, session_file):
        """Load existing session data"""
        try:
            with open(session_file, 'r') as f:
                self.session_data = json.load(f)
                self.frame_count = len(self.session_data.get("frames", []))
                self.current_session_info = self.session_data.get("session_info", {})
            logger.info(f"Loaded session: {session_file} ({self.frame_count} frames)")
            return True
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False
    
    def get_frames_for_smpl(self, frame_range=None):
        """Get frame data formatted for SMPL processing"""
        frames = self.session_data.get("frames", [])
        
        if frame_range:
            start, end = frame_range
            frames = frames[start:end]
        
        smpl_data = []
        for frame in frames:
            if frame["detection_quality"]["pose_detected"]:
                smpl_frame = {
                    "frame_id": frame["frame_id"],
                    "keypoints_3d": frame["keypoints"]["world_3d"],
                    "keypoints_2d": frame["keypoints"]["raw_2d"],
                    "confidence": frame["keypoints"]["confidence"],
                    "limb_measurements": frame["limb_measurements"],
                    "detection_score": frame["detection_quality"]["detection_score"]
                }
                smpl_data.append(smpl_frame)
        
        return smpl_data
    
    def export_summary(self):
        """Export session summary statistics"""
        frames = self.session_data.get("frames", [])
        if not frames:
            return {}
        
        # Calculate statistics
        valid_frames = [f for f in frames if f["detection_quality"]["pose_detected"]]
        avg_confidence = np.mean([f["detection_quality"]["avg_confidence"] for f in valid_frames])
        
        # Limb measurement statistics
        limb_stats = {}
        all_limbs = set()
        for frame in valid_frames:
            all_limbs.update(frame["limb_measurements"].keys())
        
        for limb in all_limbs:
            measurements = [f["limb_measurements"][limb] for f in valid_frames 
                          if limb in f["limb_measurements"]]
            if measurements:
                limb_stats[limb] = {
                    "avg": np.mean(measurements),
                    "std": np.std(measurements),
                    "min": np.min(measurements),
                    "max": np.max(measurements),
                    "count": len(measurements)
                }
        
        summary = {
            "session_info": self.current_session_info,
            "frame_statistics": {
                "total_frames": len(frames),
                "valid_frames": len(valid_frames),
                "detection_rate": len(valid_frames) / len(frames) if frames else 0,
                "avg_confidence": float(avg_confidence) if valid_frames else 0
            },
            "limb_statistics": limb_stats
        }
        
        # Save summary
        summary_file = self.output_dir / f"{self.session_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary exported: {summary_file}")
        return summary