#!/usr/bin/env python3
"""
Flood Monitoring Video Demo Generator
Creates video demonstrations with water level indicators and analysis overlays.
"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import json

class FloodVideoDemo:
    def __init__(self, images_dir="sample_flood_images", output_dir="video_output"):
        """
        Initialize the video demo generator.
        
        Args:
            images_dir (str): Directory containing flood monitoring images
            output_dir (str): Directory to save output videos and frames
        """
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.frames_dir = os.path.join(output_dir, "frames")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Video parameters
        self.fps = 2  # Frames per second (slow for analysis visibility)
        self.frame_size = (1280, 720)  # HD resolution
        
        # Colors for indicators (BGR format for OpenCV)
        self.colors = {
            'safe': (0, 255, 0),      # Green
            'warning': (0, 165, 255), # Orange
            'danger': (0, 0, 255),    # Red
            'unknown': (128, 128, 128), # Gray
            'text': (255, 255, 255),   # White
            'background': (0, 0, 0)    # Black
        }
        
        print(f"Flood Video Demo initialized")
        print(f"Input directory: {images_dir}")
        print(f"Output directory: {output_dir}")
    
    def get_flood_status(self, water_level):
        """Get flood status based on water level."""
        if water_level is None:
            return 'unknown'
        elif water_level > 52:
            return 'safe'
        elif water_level > 51:
            return 'warning'
        else:
            return 'danger'
    
    def get_risk_level(self, water_level):
        """Get risk level based on water level."""
        if water_level is None:
            return 'Unknown'
        elif water_level > 52:
            return 'Low'
        elif water_level > 51:
            return 'Medium'
        else:
            return 'High'
    
    def analyze_image_water_level(self, image_path):
        """
        Analyze water level in an image and return flood status.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Analysis results with water level and status
        """
        try:
            # Try to use the real analyzer if available
            from real_image_flood_analyzer import FloodImageAnalyzer
            analyzer = FloodImageAnalyzer()
            distance = analyzer.detect_water_level(image_path)
            
            if distance is not None:
                status = self.get_flood_status(distance)
                risk_level = self.get_risk_level(distance)
                
                return {
                    'water_level': distance,
                    'status': status.lower(),
                    'risk_level': risk_level,
                    'analysis_success': True,
                    'method': 'real_analyzer'
                }
        except ImportError:
            pass
        
        # Fallback to mock analysis based on image characteristics
        return self.mock_water_level_analysis(image_path)
    
    def mock_water_level_analysis(self, image_path):
        """
        Mock water level analysis for demonstration purposes.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Mock analysis results
        """
        # Load image and analyze basic properties
        img = cv2.imread(image_path)
        if img is None:
            return {
                'water_level': 0,
                'status': 'unknown',
                'risk_level': 'Unknown',
                'analysis_success': False,
                'method': 'mock'
            }
        
        # Simple mock analysis based on image properties
        height, width = img.shape[:2]
        
        # Analyze blue channel (water typically has higher blue content)
        blue_channel = img[:, :, 0]
        blue_mean = np.mean(blue_channel)
        
        # Mock water level calculation (this is just for demo)
        # In reality, this would be much more sophisticated
        water_level = 45 + (blue_mean / 255) * 20  # Range: 45-65
        
        # Add some variation based on filename (for demo variety)
        filename = os.path.basename(image_path).lower()
        if 'morning' in filename or '08-' in filename or '09-' in filename:
            water_level += 5  # Morning tends to be higher
        elif 'evening' in filename or '18-' in filename or '19-' in filename:
            water_level -= 3  # Evening tends to be lower
        
        # Determine status based on water level
        if water_level > 55:
            status = 'safe'
            risk_level = 'Low'
        elif water_level > 48:
            status = 'warning'
            risk_level = 'Medium'
        else:
            status = 'danger'
            risk_level = 'High'
        
        return {
            'water_level': round(water_level, 1),
            'status': status,
            'risk_level': risk_level,
            'analysis_success': True,
            'method': 'mock'
        }
    
    def create_overlay_frame(self, image_path, analysis_result, frame_number, total_frames):
        """
        Create a frame with overlay information.
        
        Args:
            image_path (str): Path to the original image
            analysis_result (dict): Water level analysis results
            frame_number (int): Current frame number
            total_frames (int): Total number of frames
            
        Returns:
            numpy.ndarray: Processed frame with overlays
        """
        # Load the original image
        img = cv2.imread(image_path)
        if img is None:
            # Create a placeholder frame
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "Image not found", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors['text'], 3)
            return img
        
        # Resize image to fit our frame size
        img = cv2.resize(img, self.frame_size)
        
        # Create overlay elements
        overlay = img.copy()
        
        # Get analysis data
        water_level = analysis_result.get('water_level', 0)
        status = analysis_result.get('status', 'unknown')
        risk_level = analysis_result.get('risk_level', 'Unknown')
        
        # Choose colors based on status
        indicator_color = self.colors.get(status, self.colors['unknown'])
        
        # Add main status indicator (large circle in top-left)
        cv2.circle(overlay, (80, 80), 50, indicator_color, -1)
        cv2.circle(overlay, (80, 80), 50, self.colors['text'], 3)
        
        # Add status text
        status_text = status.upper()
        cv2.putText(overlay, status_text, (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, indicator_color, 3)
        
        # Add water level information panel
        panel_x, panel_y = 20, 150
        panel_width, panel_height = 350, 200
        
        # Semi-transparent background panel
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        
        # Panel content
        y_offset = panel_y + 40
        line_spacing = 35
        
        # Water level text
        cv2.putText(overlay, f"Water Level: {water_level} units", 
                   (panel_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
        y_offset += line_spacing
        
        # Risk level
        cv2.putText(overlay, f"Risk Level: {risk_level}", 
                   (panel_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
        y_offset += line_spacing
        
        # Timestamp from filename
        filename = os.path.basename(image_path)
        timestamp = filename.replace('.jpg', '').replace('_', ' ')
        cv2.putText(overlay, f"Time: {timestamp}", 
                   (panel_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        y_offset += line_spacing
        
        # Frame counter
        cv2.putText(overlay, f"Frame: {frame_number}/{total_frames}", 
                   (panel_x + 20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Add water level gauge (vertical bar on the right side)
        gauge_x = self.frame_size[0] - 100
        gauge_y = 100
        gauge_width = 40
        gauge_height = 300
        
        # Background gauge
        cv2.rectangle(overlay, (gauge_x, gauge_y), (gauge_x + gauge_width, gauge_y + gauge_height), 
                     (50, 50, 50), -1)
        cv2.rectangle(overlay, (gauge_x, gauge_y), (gauge_x + gauge_width, gauge_y + gauge_height), 
                     self.colors['text'], 2)
        
        # Water level fill (0-100 scale, assuming 30-70 is our range)
        level_percent = max(0, min(1, (water_level - 30) / 40))  # Normalize to 0-1
        fill_height = int(gauge_height * level_percent)
        fill_y = gauge_y + gauge_height - fill_height
        
        cv2.rectangle(overlay, (gauge_x + 2, fill_y), (gauge_x + gauge_width - 2, gauge_y + gauge_height - 2), 
                     indicator_color, -1)
        
        # Gauge labels
        cv2.putText(overlay, "HIGH", (gauge_x - 30, gauge_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(overlay, "LOW", (gauge_x - 25, gauge_y + gauge_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Add title overlay
        title = "Flood Monitoring System - Live Analysis"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        title_x = (self.frame_size[0] - title_size[0]) // 2
        cv2.putText(overlay, title, (title_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['text'], 2)
        
        # Blend overlay with original image
        alpha = 0.85  # Transparency factor
        result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        return result
    
    def create_video_from_images(self, image_paths=None):
        """
        Create a video demo from flood monitoring images.
        
        Args:
            image_paths (list): List of image paths. If None, uses all images in images_dir
            
        Returns:
            str: Path to the created video file
        """
        if image_paths is None:
            # Get all image files from the directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_paths = []
            
            if os.path.exists(self.images_dir):
                for file in sorted(os.listdir(self.images_dir)):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(os.path.join(self.images_dir, file))
            
            if not image_paths:
                print(f"No images found in {self.images_dir}")
                return None
        
        print(f"Creating video from {len(image_paths)} images...")
        
        # Analyze all images first
        analysis_results = []
        for i, image_path in enumerate(image_paths):
            print(f"Analyzing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.analyze_image_water_level(image_path)
            analysis_results.append(result)
        
        # Create overlay frames
        processed_frames = []
        for i, (image_path, analysis_result) in enumerate(zip(image_paths, analysis_results)):
            print(f"Creating frame {i+1}/{len(image_paths)}")
            
            frame = self.create_overlay_frame(image_path, analysis_result, i+1, len(image_paths))
            
            # Save frame
            frame_filename = f"frame_{i+1:03d}.jpg"
            frame_path = os.path.join(self.frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            processed_frames.append(frame_path)
        
        # Create video using OpenCV
        video_filename = f"flood_monitoring_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        
        print(f"Generating video: {video_filename}")
        
        try:
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.frame_size)
            
            # Add each frame to the video
            for frame_path in processed_frames:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            
            print(f"✅ Video created successfully: {video_path}")
            
            # Save analysis summary
            summary_data = {
                'video_path': video_path,
                'total_frames': len(image_paths),
                'fps': self.fps,
                'duration_seconds': len(image_paths) / self.fps,
                'analysis_results': analysis_results,
                'image_paths': image_paths
            }
            
            summary_path = os.path.join(self.output_dir, 'video_analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            return video_path
            
        except Exception as e:
            print(f"Error creating video: {e}")
            return None
    
    def create_demo_video(self):
        """
        Create a demo video using available sample images.
        
        Returns:
            str: Path to created video or None if failed
        """
        print("🎬 Creating Flood Monitoring Video Demo")
        print("="*50)
        
        # Check for sample images
        if not os.path.exists(self.images_dir):
            print(f"Images directory not found: {self.images_dir}")
            return None
        
        # Create the video
        video_path = self.create_video_from_images()
        
        if video_path:
            print(f"\n🎉 Video demo completed successfully!")
            print(f"📹 Video file: {video_path}")
            print(f"📁 Frames saved in: {self.frames_dir}")
            print(f"📊 Analysis summary: {os.path.join(self.output_dir, 'video_analysis_summary.json')}")
            
            # Print statistics
            with open(os.path.join(self.output_dir, 'video_analysis_summary.json'), 'r') as f:
                summary = json.load(f)
            
            print(f"\n📈 Video Statistics:")
            print(f"   Duration: {summary['duration_seconds']} seconds")
            print(f"   Frames: {summary['total_frames']}")
            print(f"   FPS: {summary['fps']}")
            
            # Analyze status distribution
            status_counts = {}
            for result in summary['analysis_results']:
                status = result['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print(f"\n🚦 Status Distribution:")
            for status, count in status_counts.items():
                print(f"   {status.upper()}: {count} frames")
        
        return video_path


def main():
    """
    Main function to create the flood monitoring video demo.
    """
    print("Flood Monitoring Video Demo Generator")
    print("="*50)
    
    # Create demo with sample images
    video_demo = FloodVideoDemo("sample_flood_images", "flood_video_output")
    video_path = video_demo.create_demo_video()
    
    if video_path:
        print(f"\n✅ Demo video created: {os.path.basename(video_path)}")
        print(f"🎯 The video includes:")
        print(f"   • Red/Green/Orange water level indicators")
        print(f"   • Real-time water level readings")
        print(f"   • Risk level assessments")
        print(f"   • Visual gauge showing water levels")
        print(f"   • Timestamp information")
    else:
        print("❌ Failed to create demo video")


if __name__ == "__main__":
    main()