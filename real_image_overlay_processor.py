#!/usr/bin/env python3
"""
Real Image Processor with Water Level and Status Overlay
This script processes real flood images and adds water level and status text in the top-left corner.
"""

import cv2
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path

class RealImageProcessor:
    def __init__(self, threshold=51.4):
        self.threshold = threshold
        self.processed_images = []
        
    def find_images(self, search_paths=None):
        """Find actual images that match CSV filenames."""
        print("="*60)
        print("FINDING REAL FLOOD MONITORING IMAGES")
        print("="*60)
        
        # Load CSV data
        df = pd.read_csv('Flood_Data.csv')
        df.columns = df.columns.str.strip()
        
        print(f"CSV contains {len(df)} image references")
        
        # Get image filenames from CSV
        filenames = set(df['TimeStamp'].dropna().astype(str))
        print(f"Looking for {len(filenames)} unique images")
        
        # Default search paths if none provided
        if search_paths is None:
            search_paths = [
                ".",
                "images",
                "flood_images",
                "data", 
                "../images",
                "sample_flood_images",  # Our demo images
                "/workspaces/flood_monitoring",
                os.path.expanduser("~/Downloads"),
                os.path.expanduser("~/Pictures")
            ]
        
        found_images = {}
        
        # Search for images
        for search_path in search_paths:
            if os.path.exists(search_path):
                print(f"Searching in: {search_path}")
                
                # Search for image files
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                
                for ext in extensions:
                    pattern = os.path.join(search_path, '**', ext)
                    files = glob.glob(pattern, recursive=True)
                    
                    for file_path in files:
                        filename = os.path.basename(file_path)
                        if filename in filenames:
                            found_images[filename] = file_path
                            
                        # Also check without extension for frame files
                        name_without_ext = os.path.splitext(filename)[0] + '.jpg'
                        if name_without_ext in filenames:
                            found_images[name_without_ext] = file_path
        
        print(f"Found {len(found_images)} matching images")
        
        if found_images:
            print("\nFound images:")
            for filename, path in list(found_images.items())[:10]:
                print(f"  {filename} → {path}")
            if len(found_images) > 10:
                print(f"  ... and {len(found_images) - 10} more")
        
        return found_images, df
    
    def get_csv_data_for_image(self, filename, df):
        """Get CSV data for a specific image filename."""
        row = df[df['TimeStamp'] == filename]
        if not row.empty:
            return {
                'distance': float(row['Distance'].iloc[0]),
                'linecount': int(row['LineCount'].iloc[0]),
                'timestamp': row['TimeStamp'].iloc[0]
            }
        return None
    
    def detect_water_level_simple(self, image_path):
        """Simple water level detection for demonstration."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Simple edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find horizontal lines (water surface)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Find contours
            contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the most prominent horizontal line
                best_y = None
                max_length = 0
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > max_length and w > img.shape[1] * 0.2:
                        max_length = w
                        best_y = y
                
                if best_y is not None:
                    # Convert to distance (simple calibration)
                    height = img.shape[0]
                    normalized_y = best_y / height
                    # Distance range 50.0 to 53.0
                    distance = 53.0 - (normalized_y * 3.0)
                    return round(distance, 1)
            
            # Fallback: return a reasonable value
            return 51.0
            
        except Exception as e:
            print(f"Error detecting water level: {e}")
            return None
    
    def add_text_overlay(self, image, csv_data, detected_distance=None):
        """Add water level and status text overlay to the top-left corner."""
        img_copy = image.copy()
        
        # Use CSV data if available, otherwise use detected
        if csv_data:
            water_level = csv_data['distance']
        elif detected_distance:
            water_level = detected_distance
        else:
            water_level = 51.0
        
        # Determine status based on threshold
        if water_level > self.threshold:
            status = "HIGH RISK"
            status_color = (0, 0, 255)  # Red in BGR
            bg_color = (0, 0, 200)      # Dark red background
        else:
            status = "SAFE"
            status_color = (0, 255, 0)  # Green in BGR
            bg_color = (0, 150, 0)      # Dark green background
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_color = (255, 255, 255)  # White text
        
        # Prepare text lines
        lines = [
            f"Water Level: {water_level:.1f}",
            f"Status: {status}",
            f"Threshold: {self.threshold}"
        ]
        
        # Calculate text dimensions
        line_height = 35
        padding = 10
        max_width = 0
        
        for line in lines:
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_width)
        
        # Create background rectangle
        bg_width = max_width + (padding * 2)
        bg_height = len(lines) * line_height + padding
        
        # Draw background rectangle
        cv2.rectangle(img_copy, (5, 5), (5 + bg_width, 5 + bg_height), bg_color, -1)
        cv2.rectangle(img_copy, (5, 5), (5 + bg_width, 5 + bg_height), (255, 255, 255), 2)
        
        # Add text lines
        for i, line in enumerate(lines):
            y_position = 5 + padding + (i + 1) * line_height - 5
            
            # Use status color for the status line
            if "Status:" in line:
                color = status_color
            else:
                color = text_color
            
            cv2.putText(img_copy, line, (5 + padding, y_position), 
                       font, font_scale, color, thickness)
        
        return img_copy
    
    def process_single_image(self, image_path, filename, csv_data):
        """Process a single image with overlays."""
        print(f"Processing: {filename}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Error: Could not load image")
            return None
        
        print(f"  Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Get water level data
        if csv_data:
            water_level = csv_data['distance']
            print(f"  CSV Water Level: {water_level}")
        else:
            # Try to detect water level
            water_level = self.detect_water_level_simple(image_path)
            print(f"  Detected Water Level: {water_level}")
        
        # Determine status
        status = "HIGH RISK" if water_level > self.threshold else "SAFE"
        print(f"  Status: {status}")
        
        # Add overlay
        processed_img = self.add_text_overlay(img, csv_data, water_level)
        
        # Save processed image
        output_dir = "processed_flood_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}_processed.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save image
        cv2.imwrite(output_path, processed_img)
        print(f"  Saved: {output_path}")
        
        # Store result
        result = {
            'original_filename': filename,
            'original_path': image_path,
            'processed_path': output_path,
            'water_level': water_level,
            'status': status,
            'csv_data': csv_data
        }
        
        self.processed_images.append(result)
        return result
    
    def create_comparison_grid(self, max_images=6):
        """Create a comparison grid showing original and processed images."""
        if not self.processed_images:
            print("No processed images to display")
            return
        
        # Limit number of images for display
        images_to_show = self.processed_images[:max_images]
        
        fig, axes = plt.subplots(2, len(images_to_show), figsize=(4*len(images_to_show), 8))
        if len(images_to_show) == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Real Flood Images: Original vs Processed with Water Level Overlay', 
                     fontsize=14, fontweight='bold')
        
        for i, result in enumerate(images_to_show):
            # Load original and processed images
            original = cv2.imread(result['original_path'])
            processed = cv2.imread(result['processed_path'])
            
            if original is not None and processed is not None:
                # Convert BGR to RGB for matplotlib
                original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                
                # Display original
                axes[0, i].imshow(original_rgb)
                axes[0, i].set_title(f'Original\n{result["original_filename"]}', fontsize=10)
                axes[0, i].axis('off')
                
                # Display processed
                axes[1, i].imshow(processed_rgb)
                status_color = 'red' if result['status'] == 'HIGH RISK' else 'green'
                axes[1, i].set_title(f'Processed\nLevel: {result["water_level"]:.1f}, Status: {result["status"]}', 
                                   fontsize=10, color=status_color, fontweight='bold')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        output_file = 'real_images_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison grid saved as: {output_file}")
        plt.show()
    
    def process_all_found_images(self, found_images, df, max_process=10):
        """Process all found images or up to max_process."""
        print(f"\n{'='*60}")
        print("PROCESSING REAL FLOOD IMAGES")
        print(f"{'='*60}")
        
        processed_count = 0
        
        for filename, image_path in found_images.items():
            if processed_count >= max_process:
                print(f"Reached maximum processing limit ({max_process})")
                break
            
            # Get CSV data for this image
            csv_data = self.get_csv_data_for_image(filename, df)
            
            # Process the image
            result = self.process_single_image(image_path, filename, csv_data)
            
            if result:
                processed_count += 1
                print()
        
        print(f"Successfully processed {processed_count} images")
        return processed_count

def main():
    """Main function to process real flood images with overlays."""
    print("="*80)
    print("REAL FLOOD IMAGE PROCESSOR WITH WATER LEVEL OVERLAY")
    print("="*80)
    print("This script will find your real flood images from the CSV data")
    print("and add water level and status text in the top-left corner.")
    
    # Initialize processor
    threshold = 51.4
    processor = RealImageProcessor(threshold=threshold)
    
    # Find images
    found_images, df = processor.find_images()
    
    if not found_images:
        print("\nNo matching images found!")
        print("Make sure your flood monitoring images are accessible in one of these locations:")
        print("- Current directory")
        print("- images/ folder") 
        print("- flood_images/ folder")
        print("- Or specify the correct path")
        return
    
    print(f"\nFound {len(found_images)} images that match your CSV data!")
    
    # Ask user how many to process
    max_process = min(len(found_images), 6)  # Limit for demo
    response = input(f"\nProcess {max_process} images? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Processing cancelled.")
        return
    
    # Process images
    processed_count = processor.process_all_found_images(found_images, df, max_process)
    
    if processed_count > 0:
        # Create comparison grid
        processor.create_comparison_grid()
        
        # Print summary
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETED!")
        print(f"{'='*80}")
        print(f"Processed Images: {processed_count}")
        print(f"Threshold Used: {threshold}")
        
        high_risk = len([r for r in processor.processed_images if r['status'] == 'HIGH RISK'])
        safe = len([r for r in processor.processed_images if r['status'] == 'SAFE'])
        
        print(f"High Risk Images: {high_risk}")
        print(f"Safe Images: {safe}")
        
        print(f"\nGenerated Files:")
        print(f"• processed_flood_images/ - Folder with processed images")
        print(f"• real_images_comparison.png - Comparison grid")
        
        print(f"\nProcessed Images:")
        for result in processor.processed_images:
            print(f"  {result['original_filename']} → Level: {result['water_level']:.1f}, Status: {result['status']}")
    
    else:
        print("No images were successfully processed.")

if __name__ == "__main__":
    main()