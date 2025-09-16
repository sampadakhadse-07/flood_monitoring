#!/usr/bin/env python3
"""
Enhanced Flood Video Demo with CSV Integration
Creates comprehensive video demos using CSV data and Google Drive integration.
"""

import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json
from flood_video_demo import FloodVideoDemo

class EnhancedFloodVideoDemo(FloodVideoDemo):
    def __init__(self, csv_file="Flood_Data.csv", images_dir="targeted_images", output_dir="enhanced_video_output"):
        """
        Initialize enhanced video demo with CSV integration.
        
        Args:
            csv_file (str): Path to CSV file with flood data
            images_dir (str): Directory containing flood images
            output_dir (str): Output directory for videos
        """
        super().__init__(images_dir, output_dir)
        self.csv_file = csv_file
        self.csv_data = None
        
        # Load CSV data if available
        self.load_csv_data()
        
        print(f"Enhanced Flood Video Demo initialized")
        print(f"CSV file: {csv_file}")
    
    def load_csv_data(self):
        """Load CSV data for reference."""
        try:
            if os.path.exists(self.csv_file):
                self.csv_data = pd.read_csv(self.csv_file)
                print(f"✅ Loaded CSV data: {len(self.csv_data)} records")
            else:
                print(f"⚠ CSV file not found: {self.csv_file}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
    
    def get_csv_reference(self, image_filename):
        """Get CSV reference data for an image."""
        if self.csv_data is None:
            return None
        
        # Look for exact match
        matches = self.csv_data[self.csv_data['TimeStamp'] == image_filename]
        if len(matches) > 0:
            return matches.iloc[0]
        return None
    
    def create_enhanced_overlay_frame(self, image_path, analysis_result, frame_number, total_frames):
        """
        Create enhanced frame with CSV comparison.
        """
        # Get base frame
        frame = super().create_overlay_frame(image_path, analysis_result, frame_number, total_frames)
        
        # Add CSV comparison if available
        filename = os.path.basename(image_path)
        csv_ref = self.get_csv_reference(filename)
        
        if csv_ref is not None:
            # Add CSV comparison panel
            panel_x, panel_y = 400, 150
            panel_width, panel_height = 300, 150
            
            # Semi-transparent background
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), -1)
            
            # CSV data
            y_offset = panel_y + 30
            line_spacing = 30
            
            cv2.putText(frame, "CSV Reference:", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
            y_offset += line_spacing
            
            csv_distance = csv_ref.get('Distance', 'N/A')
            cv2.putText(frame, f"CSV Distance: {csv_distance}", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            y_offset += line_spacing
            
            # Comparison
            analyzed_distance = analysis_result.get('water_level', 0)
            if isinstance(csv_distance, (int, float)) and analyzed_distance:
                diff = analyzed_distance - csv_distance
                color = self.colors['safe'] if abs(diff) < 1.0 else self.colors['warning']
                cv2.putText(frame, f"Difference: {diff:.1f}", 
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y_offset += line_spacing
            
            # Match indicator
            cv2.putText(frame, "✓ CSV Match", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['safe'], 1)
        
        return frame
    
    def create_csv_matched_video(self, max_images=10):
        """
        Create video using only images that have CSV matches.
        
        Args:
            max_images (int): Maximum number of images to use
            
        Returns:
            str: Path to created video
        """
        if self.csv_data is None:
            print("No CSV data available!")
            return None
        
        print(f"Creating CSV-matched video demo...")
        
        # Find images that match CSV timestamps
        matched_images = []
        if os.path.exists(self.images_dir):
            for filename in os.listdir(self.images_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if filename in self.csv_data['TimeStamp'].values:
                        image_path = os.path.join(self.images_dir, filename)
                        matched_images.append(image_path)
                        
                        if len(matched_images) >= max_images:
                            break
        
        if not matched_images:
            print("No CSV-matched images found!")
            return None
        
        print(f"Found {len(matched_images)} CSV-matched images")
        
        # Analyze all images
        analysis_results = []
        for i, image_path in enumerate(matched_images):
            print(f"Analyzing matched image {i+1}/{len(matched_images)}: {os.path.basename(image_path)}")
            result = self.analyze_image_water_level(image_path)
            analysis_results.append(result)
        
        # Create enhanced overlay frames
        processed_frames = []
        for i, (image_path, analysis_result) in enumerate(zip(matched_images, analysis_results)):
            print(f"Creating enhanced frame {i+1}/{len(matched_images)}")
            
            frame = self.create_enhanced_overlay_frame(image_path, analysis_result, i+1, len(matched_images))
            
            # Save frame
            frame_filename = f"enhanced_frame_{i+1:03d}.jpg"
            frame_path = os.path.join(self.frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            processed_frames.append(frame_path)
        
        # Create video
        video_filename = f"enhanced_flood_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        
        print(f"Generating enhanced video: {video_filename}")
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.frame_size)
            
            for frame_path in processed_frames:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            print(f"✅ Enhanced video created: {video_path}")
            
            # Save analysis summary
            summary_data = {
                'video_path': video_path,
                'video_type': 'csv_matched',
                'total_frames': len(matched_images),
                'fps': self.fps,
                'duration_seconds': len(matched_images) / self.fps,
                'matched_images': len(matched_images),
                'csv_records': len(self.csv_data),
                'analysis_results': analysis_results,
                'image_paths': matched_images
            }
            
            summary_path = os.path.join(self.output_dir, 'enhanced_video_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            return video_path
            
        except Exception as e:
            print(f"Error creating enhanced video: {e}")
            return None
    
    def create_simulation_video(self, num_frames=20):
        """
        Create a simulation video with varying water levels for demonstration.
        
        Args:
            num_frames (int): Number of frames to create
            
        Returns:
            str: Path to created video
        """
        print(f"Creating simulation video with {num_frames} frames...")
        
        # Use a sample image as template
        sample_images = []
        if os.path.exists(self.images_dir):
            for file in os.listdir(self.images_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_images.append(os.path.join(self.images_dir, file))
                    if len(sample_images) >= 3:  # Use up to 3 different images
                        break
        
        if not sample_images and os.path.exists("sample_flood_images"):
            # Fallback to sample images
            for file in os.listdir("sample_flood_images"):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_images.append(os.path.join("sample_flood_images", file))
                    if len(sample_images) >= 3:
                        break
        
        if not sample_images:
            print("No sample images available for simulation!")
            return None
        
        # Create simulation frames with varying water levels
        processed_frames = []
        
        for i in range(num_frames):
            # Cycle through sample images
            base_image = sample_images[i % len(sample_images)]
            
            # Simulate water level changes over time
            time_factor = i / num_frames
            
            # Create a realistic water level simulation
            if time_factor < 0.3:  # Rising water (danger)
                water_level = 50.5 + (time_factor * 3)  # 50.5 to 51.4
                status = 'danger' if water_level < 51 else 'warning'
            elif time_factor < 0.7:  # Stable high water (warning to safe)
                water_level = 51.4 + ((time_factor - 0.3) * 2)  # 51.4 to 52.2
                status = 'warning' if water_level < 52 else 'safe'
            else:  # Receding water (safe)
                water_level = 52.2 + ((time_factor - 0.7) * 1)  # 52.2 to 52.5
                status = 'safe'
            
            # Create analysis result
            analysis_result = {
                'water_level': round(water_level, 1),
                'status': status,
                'risk_level': 'High' if status == 'danger' else 'Medium' if status == 'warning' else 'Low',
                'analysis_success': True,
                'method': 'simulation'
            }
            
            # Create frame
            frame = self.create_overlay_frame(base_image, analysis_result, i+1, num_frames)
            
            # Add simulation indicator
            cv2.putText(frame, "SIMULATION", (20, self.frame_size[1] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['warning'], 2)
            
            # Save frame
            frame_filename = f"sim_frame_{i+1:03d}.jpg"
            frame_path = os.path.join(self.frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            processed_frames.append(frame_path)
        
        # Create video
        video_filename = f"flood_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        
        print(f"Generating simulation video: {video_filename}")
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 3, self.frame_size)  # Slightly faster FPS
            
            for frame_path in processed_frames:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            print(f"✅ Simulation video created: {video_path}")
            
            return video_path
            
        except Exception as e:
            print(f"Error creating simulation video: {e}")
            return None


def main():
    """
    Main function to create enhanced flood monitoring videos.
    """
    print("Enhanced Flood Monitoring Video Demo")
    print("="*50)
    
    # Create enhanced demo
    enhanced_demo = EnhancedFloodVideoDemo()
    
    # Try to create CSV-matched video first
    print("\n1. Attempting CSV-matched video...")
    csv_video = enhanced_demo.create_csv_matched_video(max_images=10)
    
    # Create simulation video
    print("\n2. Creating simulation video...")
    sim_video = enhanced_demo.create_simulation_video(num_frames=15)
    
    print("\n" + "="*50)
    print("🎉 Enhanced Video Demo Complete!")
    
    if csv_video:
        print(f"✅ CSV-matched video: {os.path.basename(csv_video)}")
    else:
        print("⚠ CSV-matched video not created (no matching images)")
    
    if sim_video:
        print(f"✅ Simulation video: {os.path.basename(sim_video)}")
    
    print(f"\n📁 All videos saved in: enhanced_video_output/")
    print(f"🎯 Features included:")
    print(f"   • Red/Green/Orange water level indicators")
    print(f"   • Real-time water level readings")
    print(f"   • CSV data comparisons")
    print(f"   • Risk level assessments")
    print(f"   • Visual water level gauge")
    print(f"   • Timestamp and frame information")


if __name__ == "__main__":
    main()