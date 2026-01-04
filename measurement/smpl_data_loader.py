"""
SMPL Data Loader for 3D Pipeline
================================
Loads frame data from JSON storage and prepares it for SMPL 3D reconstruction.

Author: Claude  
Date: August 28, 2025
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class SMPLDataLoader:
    """Loads and prepares frame data for SMPL 3D pipeline"""
    
    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.session_data = None
        self.frames_data = []
        self.session_info = {}
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        self.load_data()
    
    def load_data(self):
        """Load session data from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                self.session_data = json.load(f)
            
            self.session_info = self.session_data.get("session_info", {})
            self.frames_data = self.session_data.get("frames", [])
            
            logger.info(f"Loaded {len(self.frames_data)} frames from {self.data_file}")
            logger.info(f"Model: {self.session_info.get('model_type', 'unknown')}")
            logger.info(f"Camera: {self.session_info.get('camera_type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load data file: {e}")
            raise
    
    def get_valid_frames(self, min_confidence=0.7) -> List[Dict]:
        """Get frames with valid pose detection above confidence threshold"""
        valid_frames = []
        
        for frame in self.frames_data:
            if (frame["detection_quality"]["pose_detected"] and 
                frame["detection_quality"]["avg_confidence"] >= min_confidence):
                valid_frames.append(frame)
        
        logger.info(f"Found {len(valid_frames)} valid frames (min_confidence={min_confidence})")
        return valid_frames
    
    def prepare_smpl_batch(self, frame_range: Optional[Tuple[int, int]] = None, 
                          min_confidence=0.7) -> Dict:
        """Prepare batch data for SMPL processing"""
        
        valid_frames = self.get_valid_frames(min_confidence)
        
        # Apply frame range filter if specified
        if frame_range:
            start, end = frame_range
            valid_frames = [f for f in valid_frames 
                          if start <= f["frame_id"] <= end]
            logger.info(f"Filtered to frames {start}-{end}: {len(valid_frames)} frames")
        
        if not valid_frames:
            logger.warning("No valid frames found for SMPL processing")
            return {}
        
        # Extract data arrays
        keypoints_2d = np.array([f["keypoints"]["raw_2d"] for f in valid_frames])
        confidence_scores = np.array([f["keypoints"]["confidence"] for f in valid_frames])
        
        # 3D data (if available)
        has_3d = any(f["keypoints"]["world_3d"] for f in valid_frames)
        keypoints_3d = None
        if has_3d:
            keypoints_3d = np.array([f["keypoints"]["world_3d"] for f in valid_frames])
        
        # Limb measurements
        limb_measurements = [f["limb_measurements"] for f in valid_frames]
        
        # Frame metadata
        frame_ids = [f["frame_id"] for f in valid_frames]
        timestamps = [f["timestamp_ms"] for f in valid_frames]
        
        batch_data = {
            "session_info": self.session_info,
            "keypoints_2d": keypoints_2d,  # Shape: (num_frames, num_keypoints, 2)
            "confidence_scores": confidence_scores,  # Shape: (num_frames, num_keypoints)
            "keypoints_3d": keypoints_3d,  # Shape: (num_frames, num_keypoints, 3) or None
            "limb_measurements": limb_measurements,  # List of dicts per frame
            "frame_ids": frame_ids,
            "timestamps": timestamps,
            "has_depth_data": has_3d,
            "keypoint_format": valid_frames[0]["keypoints"]["format"],
            "num_frames": len(valid_frames)
        }
        
        logger.info(f"Prepared SMPL batch: {len(valid_frames)} frames")
        logger.info(f"Keypoint format: {batch_data['keypoint_format']}")
        logger.info(f"Has 3D data: {has_3d}")
        
        return batch_data
    
    def get_frame_sequence(self, start_frame: int, num_frames: int) -> List[Dict]:
        """Get a sequence of consecutive frames for temporal analysis"""
        sequence = []
        
        for frame in self.frames_data:
            if (start_frame <= frame["frame_id"] < start_frame + num_frames and
                frame["detection_quality"]["pose_detected"]):
                sequence.append(frame)
        
        logger.info(f"Retrieved sequence: {len(sequence)} frames from frame {start_frame}")
        return sequence
    
    def export_for_smpl_tools(self, output_file: str, frame_range=None):
        """Export data in common SMPL tool formats"""
        batch_data = self.prepare_smpl_batch(frame_range)
        
        if not batch_data:
            logger.error("No valid data to export")
            return False
        
        # Create simplified format for SMPL tools
        smpl_format = {
            "keypoints": batch_data["keypoints_2d"].tolist(),
            "confidence": batch_data["confidence_scores"].tolist(),
            "keypoint_format": batch_data["keypoint_format"],
            "frame_ids": batch_data["frame_ids"],
            "has_measurements": len(batch_data["limb_measurements"]) > 0
        }
        
        # Add 3D data if available
        if batch_data["has_depth_data"]:
            smpl_format["keypoints_3d"] = batch_data["keypoints_3d"].tolist()
        
        # Add measurement data
        if batch_data["limb_measurements"]:
            smpl_format["limb_measurements"] = batch_data["limb_measurements"]
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(smpl_format, f, indent=2)
        
        logger.info(f"Exported SMPL data: {output_path}")
        return True
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the loaded data"""
        if not self.frames_data:
            return {}
        
        valid_frames = self.get_valid_frames()
        
        stats = {
            "total_frames": len(self.frames_data),
            "valid_frames": len(valid_frames),
            "detection_rate": len(valid_frames) / len(self.frames_data),
            "avg_confidence": np.mean([f["detection_quality"]["avg_confidence"] 
                                     for f in valid_frames]),
            "keypoint_format": valid_frames[0]["keypoints"]["format"] if valid_frames else "unknown",
            "has_depth_data": any(f["depth_data"]["has_valid_depth"] for f in valid_frames),
            "session_duration_ms": (self.frames_data[-1]["timestamp_ms"] - 
                                   self.frames_data[0]["timestamp_ms"]) if len(self.frames_data) > 1 else 0
        }
        
        # Limb measurement statistics
        all_limbs = set()
        for frame in valid_frames:
            all_limbs.update(frame["limb_measurements"].keys())
        
        limb_stats = {}
        for limb in all_limbs:
            measurements = [f["limb_measurements"][limb] for f in valid_frames 
                          if limb in f["limb_measurements"]]
            if measurements:
                limb_stats[limb] = {
                    "count": len(measurements),
                    "avg": np.mean(measurements),
                    "std": np.std(measurements),
                    "min": np.min(measurements),
                    "max": np.max(measurements)
                }
        
        stats["limb_statistics"] = limb_stats
        
        return stats