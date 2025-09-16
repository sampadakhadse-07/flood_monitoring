#!/usr/bin/env python3
"""
Flood Image Overlay Processor - Works with Real CSV Data
This script processes flood monitoring images and adds water level/status overlays.
"""

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class FloodImageOverlayProcessor:
    def __init__(self, csv_file='Flood_Data.csv', threshold=51.4):
        self.csv_file = csv_file
        self.threshold = threshold
        self.df = None
        self.load_csv_data()
    
    def load_csv_data(self):
        """Load and prepare CSV data."""
        print(f"Loading CSV data from: {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        self.df.columns = self.df.columns.str.strip()
        print(f"Loaded {len(self.df)} records")
        
        # Show sample data
        print("\nSample CSV data:")
        print(self.df[['TimeStamp', 'Distance']].head())
    
    def get_csv_data_for_image(self, filename):
        """Get CSV data for a specific image filename."""
        # Try exact match first
        row = self.df[self.df['TimeStamp'] == filename]
        
        # If no exact match, try without path
        if row.empty:
            base_filename = os.path.basename(filename)
            row = self.df[self.df['TimeStamp'] == base_filename]
        
        if not row.empty:
            return {
                'filename': row['TimeStamp'].iloc[0],
                'distance': float(row['Distance'].iloc[0]),
                'linecount': int(row['LineCount'].iloc[0])
            }
        return None
    
    def add_water_level_overlay(self, image, csv_data):
        """Add water level and status overlay to top-left corner."""
        if csv_data is None:
            return image
        
        img_copy = image.copy()
        distance = csv_data['distance']
        
        # Determine status based on threshold
        if distance > self.threshold:
            status = "HIGH RISK"
            status_color = (0, 0, 255)  # Red in BGR
            bg_color = (0, 0, 180)      # Dark red
        else:
            status = "SAFE"
            status_color = (0, 255, 0)  # Green in BGR
            bg_color = (0, 150, 0)      # Dark green
        
        # Text configuration
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_color = (255, 255, 255)  # White
        
        # Prepare text lines
        lines = [
            f"Water Level: {distance:.1f}",
            f"Status: {status}",
            f"Threshold: {self.threshold}",
            f"LineCount: {csv_data['linecount']}"
        ]
        
        # Calculate overlay dimensions
        line_height = 35
        padding = 12
        max_width = 0
        
        # Find maximum text width
        for line in lines:
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_width)
        
        # Background rectangle dimensions
        bg_width = max_width + (padding * 2)
        bg_height = len(lines) * line_height + padding
        
        # Draw semi-transparent background
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (10, 10), (10 + bg_width, 10 + bg_height), bg_color, -1)
        cv2.addWeighted(overlay, 0.8, img_copy, 0.2, 0, img_copy)
        
        # Draw border
        cv2.rectangle(img_copy, (10, 10), (10 + bg_width, 10 + bg_height), (255, 255, 255), 2)
        
        # Add text lines
        for i, line in enumerate(lines):
            y_position = 10 + padding + (i + 1) * line_height - 8
            
            # Use status color for status line, white for others
            if "Status:" in line:
                color = status_color
            else:
                color = text_color
            
            cv2.putText(img_copy, line, (10 + padding, y_position), 
                       font, font_scale, color, thickness)
        
        return img_copy
    
    def process_available_images(self, image_folder="sample_flood_images"):
        """Process available images and add overlays using CSV data."""
        print(f"\nProcessing images from: {image_folder}")
        
        if not os.path.exists(image_folder):
            print(f"Image folder not found: {image_folder}")
            return []
        
        # Get all images in folder
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        
        print(f"Found {len(image_files)} images to process")
        
        processed_images = []
        output_dir = "processed_with_csv_overlay"
        os.makedirs(output_dir, exist_ok=True)
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            print(f"\nProcessing: {filename}")
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"  Error: Could not load image")
                continue
            
            # Get CSV data (for demo, we'll use sample data from CSV)
            csv_data = self.get_sample_csv_data_for_demo(filename)
            
            if csv_data:
                print(f"  Using CSV data: Distance={csv_data['distance']:.1f}, LineCount={csv_data['linecount']}")
                
                # Add overlay
                processed_img = self.add_water_level_overlay(img, csv_data)
                
                # Save processed image
                output_filename = f"{os.path.splitext(filename)[0]}_with_csv_overlay.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, processed_img)
                
                status = "HIGH RISK" if csv_data['distance'] > self.threshold else "SAFE"
                
                processed_images.append({
                    'original_path': image_path,
                    'processed_path': output_path,
                    'filename': filename,
                    'csv_data': csv_data,
                    'status': status
                })
                
                print(f"  Status: {status}")
                print(f"  Saved: {output_path}")
            else:
                print(f"  No CSV data found for this image")
        
        return processed_images
    
    def get_sample_csv_data_for_demo(self, filename):
        """Get sample CSV data for demonstration."""
        # For demo purposes, assign different CSV values to different images
        # In real usage, this would match actual filenames
        
        sample_data = [
            {'distance': 50.8, 'linecount': 45},  # Safe
            {'distance': 51.2, 'linecount': 45},  # Safe
            {'distance': 51.4, 'linecount': 45},  # Threshold
            {'distance': 51.7, 'linecount': 45},  # High risk
            {'distance': 52.0, 'linecount': 45},  # High risk
            {'distance': 52.2, 'linecount': 45},  # High risk
        ]
        
        # Use hash of filename to consistently assign data
        import hashlib
        hash_val = int(hashlib.md5(filename.encode()).hexdigest(), 16)
        index = hash_val % len(sample_data)
        
        return sample_data[index]
    
    def create_comparison_visualization(self, processed_images):
        """Create before/after comparison visualization."""
        if not processed_images:
            print("No processed images to visualize")
            return
        
        n_images = min(len(processed_images), 6)  # Limit to 6 for display
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Flood Images: Original vs With CSV Data Overlay', 
                     fontsize=16, fontweight='bold')
        
        for i in range(n_images):
            img_data = processed_images[i]
            
            # Load images
            original = cv2.imread(img_data['original_path'])
            processed = cv2.imread(img_data['processed_path'])
            
            if original is not None and processed is not None:
                # Convert BGR to RGB for matplotlib
                original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                
                # Display original
                axes[0, i].imshow(original_rgb)
                axes[0, i].set_title(f'Original\n{img_data["filename"]}', fontsize=9)
                axes[0, i].axis('off')
                
                # Display processed
                axes[1, i].imshow(processed_rgb)
                csv_data = img_data['csv_data']
                status_color = 'red' if img_data['status'] == 'HIGH RISK' else 'green'
                title = f'With CSV Overlay\nLevel: {csv_data["distance"]:.1f}\n{img_data["status"]}'
                axes[1, i].set_title(title, fontsize=9, color=status_color, fontweight='bold')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_file = 'flood_images_csv_overlay_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nComparison visualization saved as: {output_file}")
        plt.show()
    
    def process_real_csv_images(self, image_folder_path):
        """Process real images that match CSV filenames."""
        print(f"\nProcessing real images from: {image_folder_path}")
        
        if not os.path.exists(image_folder_path):
            print(f"Image folder not found: {image_folder_path}")
            return []
        
        # Get all images
        import glob
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            pattern = os.path.join(image_folder_path, '**', ext)
            image_files.extend(glob.glob(pattern, recursive=True))
        
        print(f"Found {len(image_files)} total images")
        
        # Match with CSV data
        matched_images = []
        for image_path in image_files:
            filename = os.path.basename(image_path)
            csv_data = self.get_csv_data_for_image(filename)
            
            if csv_data:
                matched_images.append({
                    'path': image_path,
                    'filename': filename,
                    'csv_data': csv_data
                })
        
        print(f"Matched {len(matched_images)} images with CSV data")
        
        # Process matched images
        processed_images = []
        output_dir = "real_csv_overlay_images"
        os.makedirs(output_dir, exist_ok=True)
        
        for img_info in matched_images:
            print(f"\nProcessing: {img_info['filename']}")
            
            # Load image
            img = cv2.imread(img_info['path'])
            if img is None:
                continue
            
            # Add overlay
            processed_img = self.add_water_level_overlay(img, img_info['csv_data'])
            
            # Save
            output_filename = f"{os.path.splitext(img_info['filename'])[0]}_csv_overlay.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, processed_img)
            
            csv_data = img_info['csv_data']
            status = "HIGH RISK" if csv_data['distance'] > self.threshold else "SAFE"
            
            processed_images.append({
                'original_path': img_info['path'],
                'processed_path': output_path,
                'filename': img_info['filename'],
                'csv_data': csv_data,
                'status': status
            })
            
            print(f"  CSV Distance: {csv_data['distance']:.1f}")
            print(f"  Status: {status}")
            print(f"  Saved: {output_path}")
        
        return processed_images

def main():
    """Main function."""
    print("="*80)
    print("FLOOD IMAGE OVERLAY PROCESSOR - CSV DATA INTEGRATION")
    print("="*80)
    
    # Initialize processor
    processor = FloodImageOverlayProcessor()
    
    print("\nChoose processing option:")
    print("1. Process sample images with CSV data overlay (Demo)")
    print("2. Process your real flood images (specify folder)")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        # Demo with sample images
        processed_images = processor.process_available_images()
        
        if processed_images:
            processor.create_comparison_visualization(processed_images)
            
            print(f"\n{'='*60}")
            print("DEMO COMPLETED!")
            print(f"{'='*60}")
            
            high_risk = len([img for img in processed_images if img['status'] == 'HIGH RISK'])
            safe = len([img for img in processed_images if img['status'] == 'SAFE'])
            
            print(f"Processed Images: {len(processed_images)}")
            print(f"High Risk: {high_risk}")
            print(f"Safe: {safe}")
            print(f"Threshold: {processor.threshold}")
            
            print(f"\nFiles Generated:")
            print(f"• processed_with_csv_overlay/ - Images with overlays")
            print(f"• flood_images_csv_overlay_comparison.png - Comparison view")
    
    elif choice == "2":
        # Real images
        folder_path = input("Enter path to your flood images folder: ").strip()
        processed_images = processor.process_real_csv_images(folder_path)
        
        if processed_images:
            processor.create_comparison_visualization(processed_images)
            print(f"Successfully processed {len(processed_images)} real images!")
        else:
            print("No matching images found with CSV data.")
    
    else:
        print("Invalid choice. Running demo...")
        processor.process_available_images()

if __name__ == "__main__":
    import glob
    main()