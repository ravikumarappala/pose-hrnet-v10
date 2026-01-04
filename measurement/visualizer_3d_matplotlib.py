"""
3D Skeleton Visualizer - Matplotlib Backend
==========================================
Fallback 3D visualization using matplotlib for Jetson compatibility.
Works when Open3D has OpenGL/GLX issues.

Author: Claude
Date: August 28, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SkeletonVisualizer3D:
    """Matplotlib-based 3D skeleton visualizer"""
    
    def __init__(self, window_size=(8, 6), window_title="3D Pose Skeleton"):
        self.window_size = window_size
        self.window_title = window_title
        
        # Matplotlib components
        self.fig = None
        self.ax = None
        self.skeleton_lines = []
        self.joint_points = None
        
        # State management
        self.is_initialized = False
        self.current_keypoints = None
        self.current_measurements = {}
        self.fade_alpha = 1.0
        
        # Performance monitoring
        self.fps_monitor = []
        self.last_update_time = time.time()
        
        # Define skeleton structure
        self._define_skeleton_structure()
        
    def _define_skeleton_structure(self):
        """Define bone connections for both COCO-17 and MediaPipe-33 formats"""
        
        # MediaPipe-33 bone connections (will auto-detect format)
        self.mediapipe_bones = [
            # Face outline
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Shoulders and arms
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            # Body
            (11, 23), (12, 24), (23, 24),
            # Legs  
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        
        # COCO-17 bone connections
        self.coco_bones = [
            # Torso
            (11, 5), (12, 6), (5, 6), (11, 12),
            # Arms  
            (5, 7), (7, 9), (6, 8), (8, 10),
            # Legs
            (11, 13), (13, 15), (12, 14), (14, 16),
            # Head
            (0, 1), (1, 2), (2, 3), (3, 4)
        ]
        
        # Default to MediaPipe format (will be updated based on input)
        self.bones = self.mediapipe_bones
        
        # Colors for bones (RGB values 0-1)
        self.bone_colors = [
            [1.0, 0.0, 0.0],   # Red
            [1.0, 0.5, 0.0],   # Orange
            [1.0, 1.0, 0.0],   # Yellow
            [0.5, 1.0, 0.0],   # Green-yellow
            [0.0, 1.0, 0.0],   # Green
            [0.0, 1.0, 0.5],   # Green-cyan
            [0.0, 1.0, 1.0],   # Cyan
            [0.0, 0.5, 1.0],   # Blue-cyan
            [0.0, 0.0, 1.0],   # Blue
            [0.5, 0.0, 1.0],   # Blue-magenta
            [1.0, 0.0, 1.0],   # Magenta
            [1.0, 0.0, 0.5]    # Red-magenta
        ]
    
    def initialize(self):
        """Initialize matplotlib 3D visualization"""
        try:
            # Create figure and 3D axis
            plt.ion()  # Interactive mode
            self.fig = plt.figure(figsize=self.window_size)
            self.fig.suptitle(self.window_title)
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Setup axis properties
            self.ax.set_xlabel('X (normalized)')
            self.ax.set_ylabel('Y (normalized)')
            self.ax.set_zlabel('Z (confidence)')
            
            # Set appropriate limits for normalized coordinates
            self.ax.set_xlim([-1.5, 1.5])
            self.ax.set_ylim([-1.5, 1.5])
            self.ax.set_zlim([0, 1])
            
            # Set equal aspect ratio
            self.ax.set_box_aspect([1,1,0.5])
            
            # Initialize empty skeleton
            self.joint_points = self.ax.scatter([], [], [], c='cyan', s=50)
            
            # Initialize skeleton lines
            self.skeleton_lines = []
            for i, (start, end) in enumerate(self.bones):
                color = self.bone_colors[i % len(self.bone_colors)]
                line, = self.ax.plot([], [], [], color=color, linewidth=2)
                self.skeleton_lines.append(line)
            
            plt.show(block=False)
            plt.draw()
            
            self.is_initialized = True
            logger.info("Matplotlib 3D visualizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize matplotlib 3D visualizer: {e}")
            return False
    
    def update_skeleton(self, keypoints, measurements=None, confidence=None, frame_dimensions=None):
        """Update 3D skeleton with new keypoint data"""
        if not self.is_initialized or keypoints is None:
            return
            
        try:
            # Store current data
            self.current_keypoints = np.array(keypoints)
            self.current_measurements = measurements or {}
            
            # Auto-detect keypoint format and update bones
            if len(self.current_keypoints) == 17:
                self.bones = self.coco_bones
                keypoint_format = "COCO-17"
            elif len(self.current_keypoints) == 33:
                self.bones = self.mediapipe_bones  
                keypoint_format = "MediaPipe-33"
            else:
                keypoint_format = f"Unknown-{len(self.current_keypoints)}"
            
            # Debug: log keypoint info every 30 frames
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                
            if self._debug_counter % 30 == 0:
                logger.info(f"3D Debug - Format: {keypoint_format}, Shape: {self.current_keypoints.shape}")
                logger.info(f"3D Debug - X range: [{np.min(self.current_keypoints[:, 0]):.1f}, {np.max(self.current_keypoints[:, 0]):.1f}]")
                logger.info(f"3D Debug - Y range: [{np.min(self.current_keypoints[:, 1]):.1f}, {np.max(self.current_keypoints[:, 1]):.1f}]")
                logger.info(f"3D Debug - Confidence: [{np.min(self.current_keypoints[:, 2]):.2f}, {np.max(self.current_keypoints[:, 2]):.2f}]")
            
            # Check if pose is detected
            if confidence and confidence > 0.3:
                self.fade_alpha = min(1.0, self.fade_alpha + 0.1)
            else:
                self.fade_alpha = max(0.2, self.fade_alpha - 0.05)
            
            # Update geometry
            self._update_skeleton_geometry()
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_update_time) if (current_time - self.last_update_time) > 0 else 0
            self.fps_monitor.append(fps)
            if len(self.fps_monitor) > 10:
                self.fps_monitor.pop(0)
            self.last_update_time = current_time
            
        except Exception as e:
            logger.warning(f"Error updating matplotlib skeleton: {e}")
    
    def _update_skeleton_geometry(self):
        """Update skeleton geometry with current keypoint data"""
        if self.current_keypoints is None or len(self.current_keypoints) == 0:
            return
        
        # Normalize and scale coordinates
        scaled_points = self._normalize_coordinates(self.current_keypoints)
        
        # Update joint points
        if len(scaled_points) > 0:
            valid_indices = self.current_keypoints[:, 2] > 0.3 if self.current_keypoints.shape[1] >= 3 else range(len(scaled_points))
            
            if len(valid_indices) > 0:
                valid_points = scaled_points[valid_indices]
                if len(valid_points) > 0:
                    # Update joint scatter plot
                    self.joint_points._offsets3d = (
                        valid_points[:, 0], 
                        valid_points[:, 1], 
                        valid_points[:, 2]
                    )
                    self.joint_points.set_alpha(self.fade_alpha)
        
        # Update skeleton lines
        for i, (start_idx, end_idx) in enumerate(self.bones):
            if (start_idx < len(scaled_points) and end_idx < len(scaled_points)):
                # Check if both keypoints are confident
                start_conf = self.current_keypoints[start_idx, 2] if self.current_keypoints.shape[1] >= 3 else 1.0
                end_conf = self.current_keypoints[end_idx, 2] if self.current_keypoints.shape[1] >= 3 else 1.0
                
                if start_conf > 0.3 and end_conf > 0.3:
                    # Get line points
                    x_data = [scaled_points[start_idx, 0], scaled_points[end_idx, 0]]
                    y_data = [scaled_points[start_idx, 1], scaled_points[end_idx, 1]]
                    z_data = [scaled_points[start_idx, 2], scaled_points[end_idx, 2]]
                    
                    # Update line
                    if i < len(self.skeleton_lines):
                        self.skeleton_lines[i].set_data_3d(x_data, y_data, z_data)
                        self.skeleton_lines[i].set_alpha(self.fade_alpha)
                else:
                    # Hide line if keypoints not confident
                    if i < len(self.skeleton_lines):
                        self.skeleton_lines[i].set_data_3d([], [], [])
        
        # Update display
        try:
            # Add measurement annotations
            self._update_measurements_display(scaled_points)
            
            # Refresh plot
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logger.debug(f"Display update error: {e}")
    
    def _update_measurements_display(self, scaled_points):
        """Add measurement text to 3D plot"""
        # Clear previous text annotations
        for txt in self.ax.texts:
            if hasattr(txt, '_measurement_label'):
                txt.remove()
        
        # Add current measurements
        measurement_limbs = {
            "left_upper_arm": (5, 7),
            "left_lower_arm": (7, 9),
            "right_upper_arm": (6, 8),
            "right_lower_arm": (8, 10),
            "left_upper_leg": (11, 13),
            "left_lower_leg": (13, 15),
            "right_upper_leg": (12, 14),
            "right_lower_leg": (14, 16)
        }
        
        for limb_name, (start_idx, end_idx) in measurement_limbs.items():
            if (limb_name in self.current_measurements and 
                start_idx < len(scaled_points) and end_idx < len(scaled_points)):
                
                measurement = self.current_measurements[limb_name]
                
                # Calculate midpoint
                mid_point = (scaled_points[start_idx] + scaled_points[end_idx]) / 2
                
                # Add text annotation
                text = self.ax.text(
                    mid_point[0], mid_point[1], mid_point[2],
                    f"{measurement:.1f}cm",
                    fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
                )
                text._measurement_label = True  # Mark as measurement label
    
    def _normalize_coordinates(self, keypoints):
        """Normalize and scale keypoints for 3D display"""
        if len(keypoints) == 0:
            return np.zeros((17, 3))
        
        keypoints = np.array(keypoints)
        
        # Handle MediaPipe format: [x_pixels, y_pixels, confidence]
        if keypoints.shape[1] == 3:
            # Convert pixel coordinates to normalized coordinates
            points_3d = np.zeros((len(keypoints), 3))
            
            # Normalize pixel coordinates to [-1, 1] range
            # Assuming typical frame size ~640x480
            points_3d[:, 0] = (keypoints[:, 0] - 320) / 320.0  # X: center and normalize
            points_3d[:, 1] = -(keypoints[:, 1] - 240) / 240.0  # Y: flip and normalize (OpenGL convention)
            
            # Use confidence as pseudo-depth, scaled to reasonable range
            points_3d[:, 2] = keypoints[:, 2] * 0.5  # Z: confidence as depth [0, 0.5]
        else:
            # Fallback for unexpected formats
            points_3d = keypoints[:, :3].copy().astype(float)
        
        # Filter valid points based on confidence
        confidence_mask = keypoints[:, 2] > 0.3 if keypoints.shape[1] >= 3 else np.ones(len(keypoints), dtype=bool)
        valid_points = points_3d[confidence_mask]
        
        # Center the skeleton on valid points only
        if len(valid_points) > 0:
            center = np.mean(valid_points, axis=0)
            points_3d[:, 0] -= center[0]  # Center X
            points_3d[:, 1] -= center[1]  # Center Y
            # Keep Z relative (don't center depth)
        
        # Scale for better visualization in the plot window
        scale_factor = 0.8  # Keep within view bounds
        points_3d *= scale_factor
        
        return points_3d
    
    def render_frame(self):
        """Render a single frame (matplotlib handles this automatically)"""
        return True  # Matplotlib updates automatically
    
    def stop(self):
        """Stop the visualizer"""
        if self.fig:
            plt.close(self.fig)
        logger.info("Matplotlib 3D visualizer stopped")

class SkeletonVisualizer3DManager:
    """Manager class for integrating 3D visualizer with pose estimation"""
    
    def __init__(self, enable_3d=False, window_size=None):
        self.enable_3d = enable_3d
        self.visualizer = None
        
        if self.enable_3d:
            self.visualizer = SkeletonVisualizer3D()
    
    def initialize(self):
        """Initialize 3D visualizer if enabled"""
        if self.enable_3d and self.visualizer:
            success = self.visualizer.initialize()
            if success:
                logger.info("3D skeleton visualizer ready (matplotlib backend)")
            else:
                logger.warning("3D visualizer initialization failed")
            return success
        return True
    
    def update(self, keypoints, measurements, confidence=None):
        """Update 3D skeleton with current pose data"""
        if self.enable_3d and self.visualizer and self.visualizer.is_initialized:
            self.visualizer.update_skeleton(
                keypoints=keypoints,
                measurements=measurements,
                confidence=confidence
            )
    
    def get_stats(self):
        """Get performance stats"""
        if self.enable_3d and self.visualizer:
            avg_fps = np.mean(self.visualizer.fps_monitor) if self.visualizer.fps_monitor else 0
            return {
                "fps": avg_fps,
                "fade_alpha": self.visualizer.fade_alpha
            }
        return {}
    
    def cleanup(self):
        """Clean up 3D resources"""
        if self.visualizer:
            self.visualizer.stop()

# Test function
if __name__ == "__main__":
    # Simple test
    visualizer = SkeletonVisualizer3D()
    if visualizer.initialize():
        print("✅ Matplotlib 3D visualizer working")
        
        # Test with dummy data
        test_keypoints = np.random.rand(17, 3)
        test_keypoints[:, 2] = 0.8  # High confidence
        test_measurements = {
            "left_upper_arm": 25.5,
            "right_upper_arm": 24.8
        }
        
        visualizer.update_skeleton(test_keypoints, test_measurements, 0.8)
        
        print("Test skeleton displayed - close window to continue")
        input("Press Enter to exit...")
        visualizer.stop()
    else:
        print("❌ Failed to initialize matplotlib 3D visualizer")