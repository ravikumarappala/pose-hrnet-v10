"""
MediaPipe Pose Estimation Model
==============================
Real MediaPipe implementation for human pose estimation.
Optimized for both Jetson AGX Orin and Windows platforms.

Author: Claude
Date: July 29, 2025
"""

import cv2
import numpy as np
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

# Platform detection
try:
    PLATFORM = "jetson" if "tegra" in os.uname().release.lower() else "other"
except:
    PLATFORM = "other"

class MediaPipeModel:
    """Real MediaPipe implementation for human pose estimation"""
    
    def __init__(self):
        self.name = "MediaPipe"
        self.model = None
        self.mp = None
        self.mp_pose = None
        self.mp_drawing = None
        self._load_model()
    
    def _load_model(self):
        """Load MediaPipe pose model optimized for platform"""
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Optimize for AGX Orin: use lighter model complexity
            if PLATFORM == "jetson":
                model_complexity = 1  # Lighter model for Jetson
                min_detection_confidence = 0.7  # Higher confidence for better accuracy
                min_tracking_confidence = 0.7
                logger.info("MediaPipe optimized for Jetson AGX Orin")
            else:
                model_complexity = 2  # Full complexity for development
                min_detection_confidence = 0.5
                min_tracking_confidence = 0.5
                logger.info("MediaPipe optimized for development platform")
            
            self.model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
            logger.info("Successfully loaded MediaPipe model")
            
        except ImportError:
            logger.error("MediaPipe not installed. Please install with: pip install mediapipe")
            raise
        except Exception as e:
            logger.error(f"Failed to load MediaPipe model: {e}")
            raise
    
    def process_frame(self, color_frame, depth_frame, limb_measurements=None):
        """Process a frame with MediaPipe and return results with measurements on limbs"""
        if color_frame is None:
            return None
        
        try:
            logger.debug("Starting MediaPipe frame processing")
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            logger.debug("Converted to RGB")
            
            # Process the image
            results = self.model.process(image_rgb)
            logger.debug("MediaPipe processing complete")
            
            if not results.pose_landmarks:
                logger.debug("No pose landmarks found")
                return color_frame
            
            # Create a copy for drawing
            output_image = color_frame.copy()
            
            # Extract keypoints first
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * color_frame.shape[1])
                y = int(landmark.y * color_frame.shape[0])
                keypoints.append([x, y, landmark.visibility])
            
            keypoints = np.array(keypoints)
            logger.debug(f"Keypoints shape: {keypoints.shape}, dtype: {keypoints.dtype}")
            
            # Draw custom pose with measurements on limbs
            self._draw_pose_with_measurements(output_image, keypoints, limb_measurements)
            
            # Create results object
            result_obj = type('MediaPipeResults', (), {})()
            result_obj.keypoints = keypoints
            result_obj.processed_image = output_image
            result_obj.has_detection = True
            
            return result_obj
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing MediaPipe frame: {e}")
            logger.error(f"MediaPipe Traceback: {traceback.format_exc()}")
            return None
    
    def _draw_pose_with_measurements(self, image, keypoints, limb_measurements=None):
        """Draw pose skeleton with measurements labeled on each limb"""
        if limb_measurements is None:
            limb_measurements = {}
        
        logger.debug(f"Drawing pose with keypoints shape: {keypoints.shape if isinstance(keypoints, np.ndarray) else len(keypoints)}")
        
        # Define limb connections with their measurement names
        limb_connections = {
            "right_upper_arm": (12, 14),    # Right shoulder to right elbow
            "right_lower_arm": (14, 16),    # Right elbow to right wrist  
            "left_upper_arm": (11, 13),     # Left shoulder to left elbow
            "left_lower_arm": (13, 15),     # Left elbow to left wrist
            "right_upper_leg": (24, 26),    # Right hip to right knee
            "right_lower_leg": (26, 28),    # Right knee to right ankle
            "left_upper_leg": (23, 25),     # Left hip to left knee
            "left_lower_leg": (25, 27)      # Left knee to left ankle
        }
        
        # Draw body connections (non-limbs) first
        body_connections = [
            (11, 12),  # Shoulders
            (11, 23),  # Left shoulder to left hip
            (12, 24),  # Right shoulder to right hip
            (23, 24),  # Hips
            (0, 1), (1, 2), (2, 3), (3, 7),      # Face outline
            (0, 4), (4, 5), (5, 6), (6, 8),      # Face outline
            (9, 10),   # Mouth
        ]
        
        # Draw non-limb connections in gray
        for connection in body_connections:
            start_idx, end_idx = connection
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                float(keypoints[start_idx, 2]) > 0.3 and float(keypoints[end_idx, 2]) > 0.3):
                
                start_point = tuple(keypoints[start_idx][:2].astype(int))
                end_point = tuple(keypoints[end_idx][:2].astype(int))
                cv2.line(image, start_point, end_point, (128, 128, 128), 2)
        
        # Draw limbs with measurements
        for limb_name, (start_idx, end_idx) in limb_connections.items():
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                float(keypoints[start_idx, 2]) > 0.3 and float(keypoints[end_idx, 2]) > 0.3):
                
                start_point = tuple(keypoints[start_idx][:2].astype(int))
                end_point = tuple(keypoints[end_idx][:2].astype(int))
                
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
        for i, (x, y, conf) in enumerate(keypoints):
            if float(conf) > 0.3:
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)
    
    def get_keypoints_from_results(self, results):
        """Extract keypoints from MediaPipe results"""
        if not hasattr(results, 'keypoints'):
            return np.zeros((33, 3))  # MediaPipe has 33 keypoints
        return results.keypoints
    
    def release(self):
        """Release MediaPipe resources"""
        try:
            if self.model:
                self.model.close()
        except:
            pass
    
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.release()