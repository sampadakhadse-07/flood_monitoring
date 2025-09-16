#!/usr/bin/env python3
"""
Simple Water Level Scale Video Demo
Creates a video demonstration using available flood images and simulates water level scale detection
"""

import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SimpleScaleVideoDemo:
    def __init__(self):
        self.output_dir = Path("simple_scale_demo")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🎬 Simple Water Level Scale Video Demo")
        print("=" * 60)
        
        # Use existing flood images if available
        self.sample_images_dir = Path("sample_flood_images")
        
    def get_available_images(self):
        """Get available flood images from local directory"""
        image_files = []
        
        if self.sample_images_dir.exists():
            for ext in ['*.jpg', '*.jpeg']:
                image_files.extend(list(self.sample_images_dir.glob(ext)))
        
        # Also check current directory for JPG files only
        current_dir = Path(".")
        for ext in ['*.jpg', '*.jpeg']:
            image_files.extend([f for f in current_dir.glob(ext)])
        
        return sorted(image_files)[:10]  # Limit to 10 images
    
    def simulate_water_level_detection(self, image_path):
        """Simulate water level detection and return mock scale data"""
        # Generate realistic but simulated water levels based on filename patterns
        filename = image_path.name.lower()
        
        # Extract time info if available for more realistic simulation
        base_level = 45.0  # Base water level
        
        if '08-30' in filename or 'morning' in filename:
            water_level = base_level + np.random.uniform(5, 10)
        elif '12-20' in filename or 'noon' in filename:
            water_level = base_level + np.random.uniform(15, 25)
        elif '16-45' in filename or 'evening' in filename:
            water_level = base_level + np.random.uniform(8, 15)
        else:
            water_level = base_level + np.random.uniform(5, 20)
        
        # Add some realistic variation
        water_level += np.random.normal(0, 2)
        water_level = max(30.0, min(80.0, water_level))  # Clamp between reasonable limits
        
        # Determine status based on water level
        if water_level < 40:
            status = "NORMAL"
            color = (0, 255, 0)  # Green
        elif water_level < 60:
            status = "WARNING" 
            color = (0, 165, 255)  # Orange
        else:
            status = "DANGER"
            color = (0, 0, 255)  # Red
            
        return {
            'water_level': round(water_level, 2),
            'status': status,
            'color': color,
            'confidence': round(np.random.uniform(85, 98), 1)
        }
    
    def create_scale_overlay(self, image, scale_data):
        """Create water level scale overlay on image"""
        height, width = image.shape[:2]
        overlay = image.copy()
        
        # Create semi-transparent overlay
        alpha = 0.7
        overlay_bg = np.zeros_like(image)
        
        # Draw scale indicator box
        scale_x = width - 300
        scale_y = 50
        scale_width = 250
        scale_height = 150
        
        # Background for scale info
        cv2.rectangle(overlay_bg, (scale_x, scale_y), 
                     (scale_x + scale_width, scale_y + scale_height), 
                     (40, 40, 40), -1)
        
        # Draw scale lines (simulated measurement scale)
        for i in range(5):
            line_y = scale_y + 30 + i * 20
            cv2.line(overlay_bg, (scale_x + 10, line_y), (scale_x + 40, line_y), (200, 200, 200), 2)
            # Add scale numbers
            scale_value = 70 - i * 10
            cv2.putText(overlay_bg, f"{scale_value}", (scale_x + 45, line_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Current water level indicator
        water_level = scale_data['water_level']
        indicator_y = scale_y + 30 + int((70 - water_level) * 2)  # Scale mapping
        cv2.circle(overlay_bg, (scale_x + 25, indicator_y), 8, scale_data['color'], -1)
        cv2.circle(overlay_bg, (scale_x + 25, indicator_y), 10, (255, 255, 255), 2)
        
        # Status information
        status_color = scale_data['color']
        cv2.putText(overlay_bg, f"Water Level: {water_level}", (scale_x + 70, scale_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay_bg, f"Status: {scale_data['status']}", (scale_x + 70, scale_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(overlay_bg, f"Confidence: {scale_data['confidence']}%", (scale_x + 70, scale_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(overlay_bg, f"Timestamp: {timestamp}", (scale_x + 70, scale_y + 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Blend overlay
        result = cv2.addWeighted(overlay, alpha, overlay_bg, 1 - alpha, 0)
        
        # Add main title
        title_text = "FLOOD MONITORING - WATER LEVEL SCALE"
        text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_COMPLEX, 1.2, 3)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(result, title_text, (text_x, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(result, title_text, (text_x, 40), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 1)
        
        return result
    
    def create_demo_video(self, image_paths):
        """Create demonstration video with water level scale overlays"""
        if not image_paths:
            print("❌ No images available for demo")
            return False
        
        # Video settings
        fps = 1  # Slow for demo
        video_path = self.output_dir / f"water_scale_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        # Get first image dimensions
        sample_img = cv2.imread(str(image_paths[0]))
        if sample_img is None:
            print(f"❌ Cannot read sample image: {image_paths[0]}")
            return False
        
        height, width = sample_img.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        print(f"🎬 Creating video with {len(image_paths)} images...")
        
        results_data = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing frame {i+1}/{len(image_paths)}: {image_path.name}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠️ Skipping unreadable image: {image_path}")
                continue
            
            # Resize if needed
            image = cv2.resize(image, (width, height))
            
            # Simulate water level detection
            scale_data = self.simulate_water_level_detection(image_path)
            
            # Create overlay
            frame = self.create_scale_overlay(image, scale_data)
            
            # Write frame (hold each frame for multiple seconds by repeating)
            for _ in range(3):  # 3 seconds per frame
                writer.write(frame)
            
            # Store results
            results_data.append({
                'image': image_path.name,
                'water_level': scale_data['water_level'],
                'status': scale_data['status'],
                'confidence': scale_data['confidence']
            })
        
        writer.release()
        print(f"✅ Video created: {video_path}")
        
        # Save analysis results
        results_df = pd.DataFrame(results_data)
        results_path = self.output_dir / "scale_analysis_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"📊 Analysis results saved: {results_path}")
        
        # Print summary
        self.print_summary(results_df)
        
        return True
    
    def print_summary(self, results_df):
        """Print analysis summary"""
        print("\n📊 WATER LEVEL SCALE ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Images Analyzed: {len(results_df)}")
        print(f"Average Water Level: {results_df['water_level'].mean():.2f}")
        print(f"Min Water Level: {results_df['water_level'].min():.2f}")
        print(f"Max Water Level: {results_df['water_level'].max():.2f}")
        print(f"Average Confidence: {results_df['confidence'].mean():.1f}%")
        
        print("\n📈 Status Distribution:")
        status_counts = results_df['status'].value_counts()
        for status, count in status_counts.items():
            percentage = (count / len(results_df)) * 100
            print(f"  {status}: {count} images ({percentage:.1f}%)")
        
        print("\n🏆 Top 3 Highest Water Levels:")
        top_levels = results_df.nlargest(3, 'water_level')
        for _, row in top_levels.iterrows():
            print(f"  {row['image']}: {row['water_level']} ({row['status']})")
    
    def run_demo(self):
        """Run the complete demo"""
        print("🔍 Finding available images...")
        image_paths = self.get_available_images()
        
        if not image_paths:
            print("❌ No images found for demo")
            print("💡 Add some flood images to 'sample_flood_images/' directory or current directory")
            return False
        
        print(f"✅ Found {len(image_paths)} images")
        for img_path in image_paths:
            print(f"  - {img_path.name}")
        
        # Create demo video
        success = self.create_demo_video(image_paths)
        
        if success:
            print("\n🎉 Simple Water Level Scale Demo Complete!")
            print(f"📁 Output directory: {self.output_dir}")
            print("🎬 Video file: water_scale_demo_*.mp4")
            print("📊 Results file: scale_analysis_results.csv")
        else:
            print("❌ Demo failed")
        
        return success

def main():
    demo = SimpleScaleVideoDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()