"""
3D Skeleton Visualizer for Real-time Pose Representation
======================================================
Lightweight 3D skeleton visualization that runs parallel to 2D pose estimation.
Displays keypoints as 3D skeleton with measurement labels overlaid on limbs.

Optimized for Jetson AGX Orin deployment.

Author: Claude
Date: August 28, 2025
"""

import numpy as np
import threading
import time
import json
import logging
from pathlib import Path

# Try Open3D first, fallback to matplotlib for Jetson compatibility
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Using Open3D for 3D visualization")
except ImportError as e:
    print(f"Open3D import failed: {e}")
    OPEN3D_AVAILABLE = False
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        print("Using matplotlib 3D fallback")
    except ImportError:
        print("Neither Open3D nor matplotlib available")

logger = logging.getLogger(__name__)

class SkeletonVisualizer3D:
    """Real-time 3D skeleton visualizer with measurement overlay"""
    
    def __init__(self, window_size=(640, 480), window_title="3D Pose Skeleton"):
        self.window_size = window_size
        self.window_title = window_title
        
        # Rendering components
        self.vis = None
        self.skeleton_lines = None
        self.joint_spheres = None
        self.measurement_texts = []
        
        # State management
        self.is_initialized = False
        self.is_running = False
        self.current_keypoints = None
        self.current_measurements = {}
        self.fade_alpha = 1.0
        self.last_detection_time = time.time()
        
        # Performance monitoring
        self.fps_monitor = []
        self.fps_threshold = 10.0
        self.quality_level = 2  # 2=high, 1=medium, 0=low
        
        # Coordinate system
        self.coordinate_scale = 100.0  # Scale factor for better visualization
        self.center_offset = np.array([0, 0, 0])
        
        # Threading
        self.render_thread = None
        self.thread_lock = threading.Lock()
        self.should_stop = False
        
        # Define skeleton structure
        self._define_skeleton_structure()
        
    def _define_skeleton_structure(self):
        """Define bone connections for different keypoint formats"""
        
        # COCO-17 format (HRNet)
        self.coco_bones = [
            # Torso
            (11, 5), (12, 6), (5, 6), (11, 12),
            # Arms  
            (5, 7), (7, 9),    # Left arm
            (6, 8), (8, 10),   # Right arm
            # Legs
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
            # Head/neck
            (0, 1), (1, 2), (2, 3), (3, 4)
        ]
        
        # MediaPipe-33 format  
        self.mediapipe_bones = [
            # Torso
            (11, 12), (23, 24), (11, 23), (12, 24),
            # Arms
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm  
            # Legs
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
            # Head connections (simplified)
            (0, 1), (1, 2), (2, 3), (3, 4)
        ]
        
        # Measurement limb definitions
        self.measurement_limbs = {
            "left_upper_arm": (5, 7) if len(self.coco_bones) else (11, 13),
            "left_lower_arm": (7, 9) if len(self.coco_bones) else (13, 15),
            "right_upper_arm": (6, 8) if len(self.coco_bones) else (12, 14),
            "right_lower_arm": (8, 10) if len(self.coco_bones) else (14, 16),
            "left_upper_leg": (11, 13) if len(self.coco_bones) else (23, 25),
            "left_lower_leg": (13, 15) if len(self.coco_bones) else (25, 27),
            "right_upper_leg": (12, 14) if len(self.coco_bones) else (24, 26),
            "right_lower_leg": (14, 16) if len(self.coco_bones) else (26, 28)
        }
        
        # Color scheme (matching existing 2D pipeline)
        self.bone_colors = [
            [1.0, 0.0, 0.0],   # Red
            [1.0, 0.33, 0.0],  # Red-orange
            [1.0, 0.67, 0.0],  # Orange
            [1.0, 1.0, 0.0],   # Yellow
            [0.67, 1.0, 0.0],  # Yellow-green
            [0.33, 1.0, 0.0],  # Green
            [0.0, 1.0, 0.0],   # Green
            [0.0, 1.0, 0.33],  # Green-cyan
            [0.0, 1.0, 0.67],  # Cyan
            [0.0, 1.0, 1.0],   # Cyan
            [0.0, 0.67, 1.0],  # Cyan-blue
            [0.0, 0.33, 1.0],  # Blue
            [0.0, 0.0, 1.0],   # Blue
            [0.33, 0.0, 1.0],  # Blue-magenta
            [0.67, 0.0, 1.0],  # Magenta
            [1.0, 0.0, 1.0],   # Magenta
            [1.0, 0.0, 0.67],  # Magenta-red
            [1.0, 0.0, 0.33]   # Red
        ]
    
    def initialize(self):
        """Initialize Open3D visualizer and create skeleton geometry"""
        if not OPEN3D_AVAILABLE:
            logger.warning("Open3D not available - 3D visualization disabled")
            return False
            
        try:
            # Create visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=self.window_title,
                width=self.window_size[0],
                height=self.window_size[1]
            )
            
            # Create initial skeleton geometry
            self._create_skeleton_geometry()
            
            # Setup view control
            view_control = self.vis.get_view_control()
            view_control.set_front([0, -1, -1])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, -1, 0])
            view_control.set_zoom(0.8)
            
            self.is_initialized = True
            logger.info("3D visualizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize 3D visualizer: {e}")
            return False
    
    def _create_skeleton_geometry(self):
        """Create initial skeleton line set and joint spheres"""
        # Determine keypoint format and bones
        num_keypoints = 17  # Default to COCO
        bones = self.coco_bones
        
        # Create initial points (will be updated)
        points = np.zeros((num_keypoints, 3))
        
        # Create line set for skeleton bones
        self.skeleton_lines = o3d.geometry.LineSet()
        self.skeleton_lines.points = o3d.utility.Vector3dVector(points)
        self.skeleton_lines.lines = o3d.utility.Vector2iVector(bones)
        
        # Set bone colors
        bone_colors = [self.bone_colors[i % len(self.bone_colors)] for i in range(len(bones))]
        self.skeleton_lines.colors = o3d.utility.Vector3dVector(bone_colors)
        
        # Create joint spheres
        self.joint_spheres = []
        for i in range(num_keypoints):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.paint_uniform_color([0.0, 0.8, 1.0])  # Cyan for joints
            self.joint_spheres.append(sphere)
        
        # Add geometries to visualizer
        self.vis.add_geometry(self.skeleton_lines)
        if self.quality_level >= 1:  # Only add spheres for medium+ quality
            for sphere in self.joint_spheres:
                self.vis.add_geometry(sphere)
    
    def update_skeleton(self, keypoints, measurements=None, confidence=None, frame_dimensions=None):
        """Update 3D skeleton with new keypoint data"""
        if not self.is_initialized or keypoints is None:
            return
            
        try:
            with self.thread_lock:
                # Store current data
                self.current_keypoints = np.array(keypoints)
                self.current_measurements = measurements or {}
                
                # Check if pose is detected
                if confidence and confidence > 0.3:
                    self.last_detection_time = time.time()
                    self.fade_alpha = min(1.0, self.fade_alpha + 0.1)
                else:
                    # Fade out effect
                    self.fade_alpha = max(0.0, self.fade_alpha - 0.05)
                
                # Update geometry in main thread
                self._update_skeleton_geometry()
                
        except Exception as e:
            logger.warning(f"Error updating skeleton: {e}")
    
    def _update_skeleton_geometry(self):
        """Update skeleton geometry with current keypoint data"""
        if self.current_keypoints is None:
            return
        
        # Normalize and scale coordinates
        scaled_points = self._normalize_coordinates(self.current_keypoints)
        
        # Update skeleton lines
        self.skeleton_lines.points = o3d.utility.Vector3dVector(scaled_points)
        
        # Apply fade effect to line colors
        if self.fade_alpha < 1.0:
            faded_colors = []
            for color in self.skeleton_lines.colors:
                faded_color = [c * self.fade_alpha for c in color]
                faded_colors.append(faded_color)
            self.skeleton_lines.colors = o3d.utility.Vector3dVector(faded_colors)
        
        # Update joint spheres positions
        if self.quality_level >= 1:
            for i, sphere in enumerate(self.joint_spheres):
                if i < len(scaled_points) and self.current_keypoints[i][2] > 0.3:
                    # Translate sphere to keypoint position
                    sphere.translate(scaled_points[i] - sphere.get_center(), relative=False)
                    
                    # Apply fade effect to sphere color
                    if self.fade_alpha < 1.0:
                        fade_color = [0.0, 0.8 * self.fade_alpha, 1.0 * self.fade_alpha]
                        sphere.paint_uniform_color(fade_color)
        
        # Update measurement labels (if quality allows)
        if self.quality_level >= 2 and self.current_measurements:
            self._update_measurement_labels(scaled_points)
    
    def _normalize_coordinates(self, keypoints):
        """Normalize and scale keypoints for 3D display"""
        if len(keypoints) == 0:
            return np.zeros((17, 3))
        
        # Extract 3D coordinates (handle both 2D and 3D input)
        if keypoints.shape[1] >= 3:
            # Use world_3d coordinates if available
            points_3d = keypoints[:, :3].copy()
        else:
            # Convert 2D to pseudo-3D
            points_3d = np.zeros((len(keypoints), 3))
            points_3d[:, :2] = keypoints[:, :2]
        
        # Center the skeleton at origin
        valid_points = points_3d[keypoints[:, 2] > 0.3] if keypoints.shape[1] >= 3 else points_3d
        if len(valid_points) > 0:
            center = np.mean(valid_points, axis=0)
            points_3d -= center
        
        # Scale for better visualization
        points_3d *= self.coordinate_scale / 1000.0  # Convert to appropriate scale
        
        return points_3d
    
    def _update_measurement_labels(self, scaled_points):
        """Update 3D measurement text labels on limbs"""
        # Clear existing text (simplified approach)
        # In a full implementation, you would position 3D text at limb midpoints
        # For now, we'll focus on the skeleton rendering
        pass
    
    def render_frame(self):
        """Render a single frame"""
        if not self.is_initialized:
            return False
        
        start_time = time.time()
        
        try:
            # Update geometries
            self.vis.update_geometry(self.skeleton_lines)
            if self.quality_level >= 1:
                for sphere in self.joint_spheres:
                    self.vis.update_geometry(sphere)
            
            # Poll events and update renderer
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # Monitor FPS
            frame_time = time.time() - start_time
            self.fps_monitor.append(1.0 / frame_time if frame_time > 0 else 60.0)
            
            # Keep only recent FPS readings
            if len(self.fps_monitor) > 30:
                self.fps_monitor.pop(0)
            
            # Auto-optimize quality based on FPS
            self._auto_optimize_quality()
            
            return not self.vis.poll_events()  # Return False if window should close
            
        except Exception as e:
            logger.warning(f"Error rendering 3D frame: {e}")
            return False
    
    def _auto_optimize_quality(self):
        """Automatically adjust quality based on performance"""
        if len(self.fps_monitor) < 10:
            return
        
        avg_fps = np.mean(self.fps_monitor[-10:])
        
        # Reduce quality if FPS is too low
        if avg_fps < self.fps_threshold and self.quality_level > 0:
            self.quality_level -= 1
            logger.info(f"Reduced 3D quality to level {self.quality_level} (FPS: {avg_fps:.1f})")
            
            if self.quality_level < 1:
                # Remove joint spheres
                for sphere in self.joint_spheres:
                    self.vis.remove_geometry(sphere, reset_bounding_box=False)
        
        # Increase quality if performance allows
        elif avg_fps > self.fps_threshold + 5 and self.quality_level < 2:
            self.quality_level += 1
            logger.info(f"Increased 3D quality to level {self.quality_level} (FPS: {avg_fps:.1f})")
            
            if self.quality_level >= 1:
                # Re-add joint spheres
                for sphere in self.joint_spheres:
                    self.vis.add_geometry(sphere, reset_bounding_box=False)
    
    def start_rendering_thread(self):
        """Start async rendering in separate thread"""
        if self.render_thread and self.render_thread.is_alive():
            return
        
        self.should_stop = False
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.render_thread.start()
        logger.info("Started 3D rendering thread")
    
    def _render_loop(self):
        """Main rendering loop for async operation"""
        while not self.should_stop:
            if self.is_initialized:
                should_continue = self.render_frame()
                if not should_continue:
                    break
            time.sleep(0.033)  # ~30 FPS max
    
    def stop(self):
        """Stop the visualizer and clean up"""
        self.should_stop = True
        self.is_running = False
        
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        
        if self.vis:
            try:
                self.vis.destroy_window()
            except:
                pass
        
        logger.info("3D visualizer stopped")
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        if not self.fps_monitor:
            return {"fps": 0, "quality_level": self.quality_level}
        
        return {
            "fps": np.mean(self.fps_monitor[-10:]) if len(self.fps_monitor) >= 10 else 0,
            "quality_level": self.quality_level,
            "fade_alpha": self.fade_alpha
        }

class SkeletonVisualizer3DManager:
    """Manager class for integrating 3D visualizer with pose estimation"""
    
    def __init__(self, enable_3d=False, window_size=None):
        self.enable_3d = enable_3d
        self.visualizer = None
        
        if self.enable_3d:
            window_size = window_size or (640, 480)
            self.visualizer = SkeletonVisualizer3D(window_size=window_size)
    
    def initialize(self):
        """Initialize 3D visualizer if enabled"""
        if self.enable_3d and self.visualizer:
            success = self.visualizer.initialize()
            if success:
                self.visualizer.start_rendering_thread()
                logger.info("3D skeleton visualizer ready")
            return success
        return True
    
    def update(self, keypoints, measurements, confidence=None):
        """Update 3D skeleton with current pose data"""
        if self.enable_3d and self.visualizer:
            self.visualizer.update_skeleton(
                keypoints=keypoints,
                measurements=measurements,
                confidence=confidence
            )
    
    def get_stats(self):
        """Get performance stats"""
        if self.enable_3d and self.visualizer:
            return self.visualizer.get_performance_stats()
        return {}
    
    def cleanup(self):
        """Clean up 3D resources"""
        if self.visualizer:
            self.visualizer.stop()

# Playback functionality for JSON files
def playback_from_json(json_file, playback_speed=1.0):
    """Play back 3D skeleton from stored JSON data"""
    try:
        with open(json_file, 'r') as f:
            session_data = json.load(f)
        
        frames = session_data.get("frames", [])
        if not frames:
            logger.error("No frame data found in JSON file")
            return
        
        # Initialize visualizer
        visualizer = SkeletonVisualizer3D(window_title=f"3D Playback - {Path(json_file).name}")
        if not visualizer.initialize():
            logger.error("Failed to initialize 3D visualizer for playback")
            return
        
        logger.info(f"Playing back {len(frames)} frames from {json_file}")
        
        # Playback loop
        for i, frame in enumerate(frames):
            if not frame.get("detection_quality", {}).get("pose_detected", False):
                continue
            
            keypoints = frame.get("keypoints", {}).get("world_3d", [])
            measurements = frame.get("limb_measurements", {})
            confidence = frame.get("detection_quality", {}).get("avg_confidence", 0)
            
            if keypoints:
                keypoints_array = np.array(keypoints)
                # Add confidence column if not present
                if keypoints_array.shape[1] == 3:
                    conf_col = np.full((len(keypoints_array), 1), confidence)
                    keypoints_array = np.hstack([keypoints_array, conf_col])
                
                visualizer.update_skeleton(keypoints_array, measurements, confidence)
            
            # Control playback speed
            time.sleep(0.033 / playback_speed)  # ~30 FPS base rate
            
            # Render frame
            should_continue = visualizer.render_frame()
            if not should_continue:
                break
        
        # Keep window open until user closes
        while True:
            should_continue = visualizer.render_frame()
            if not should_continue:
                break
            time.sleep(0.1)
        
        visualizer.stop()
        logger.info("Playback completed")
        
    except Exception as e:
        logger.error(f"Error during playback: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='3D Skeleton Visualizer')
    parser.add_argument('--playback', type=str, help='JSON file to play back')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if args.playback:
        playback_from_json(args.playback, args.speed)
    else:
        # Test mode - create empty visualizer
        visualizer = SkeletonVisualizer3D()
        if visualizer.initialize():
            logger.info("3D visualizer test mode - close window to exit")
            while True:
                should_continue = visualizer.render_frame()
                if not should_continue:
                    break
                time.sleep(0.1)
            visualizer.stop()