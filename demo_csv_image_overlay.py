#!/usr/bin/env python3
"""
Demo: Real Image Processing with CSV Data Integration
This creates sample images that match CSV filenames and processes them with overlays.
"""

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

def create_realistic_flood_images_from_csv():
    """Create realistic flood images based on actual CSV data."""
    print("Creating realistic flood monitoring images based on CSV data...")
    
    # Load CSV data
    df = pd.read_csv('Flood_Data.csv')
    df.columns = df.columns.str.strip()
    
    # Create output directory
    os.makedirs("csv_matched_images", exist_ok=True)
    
    # Select diverse samples from CSV
    sample_data = []
    
    # Get samples with different distance values
    for distance in [51.0, 51.2, 51.4, 51.6, 51.8, 52.0, 52.2]:
        sample_rows = df[df['Distance'] == distance].head(1)
        if not sample_rows.empty:
            sample_data.append(sample_rows.iloc[0])
    
    print(f"Creating {len(sample_data)} sample images matching CSV data...")
    
    created_images = []
    
    for i, row in enumerate(sample_data):
        filename = row['TimeStamp']
        distance = row['Distance']
        linecount = row['LineCount']
        
        # Create realistic flood monitoring image
        img = create_flood_monitoring_image(distance, filename)
        
        # Save image
        filepath = os.path.join("csv_matched_images", filename)
        cv2.imwrite(filepath, img)
        
        created_images.append({
            'filename': filename,
            'filepath': filepath,
            'distance': distance,
            'linecount': linecount
        })
        
        print(f"Created: {filename} (Distance: {distance}, LineCount: {linecount})")
    
    return created_images

def create_flood_monitoring_image(distance, filename):
    """Create a realistic flood monitoring image based on distance value."""
    width, height = 800, 600
    
    # Create base image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate water level position based on distance
    # Distance 51.0 = low water (bottom 80% of image)
    # Distance 52.2 = high water (bottom 30% of image)
    water_ratio = 0.8 - ((distance - 51.0) / (52.2 - 51.0)) * 0.5
    water_y = int(height * water_ratio)
    
    # Sky/background (gradient blue)
    for y in range(int(height * 0.3)):
        intensity = int(200 + (y / (height * 0.3)) * 55)
        img[y, :] = [intensity, intensity + 20, intensity + 35]
    
    # Embankment/structure (gray gradient)
    for y in range(int(height * 0.3), water_y):
        intensity = int(120 + (y - height * 0.3) / (water_y - height * 0.3) * 60)
        img[y, :] = [intensity, intensity, intensity]
    
    # Water (blue, intensity based on level)
    water_intensity = int(50 + (distance - 51.0) / (52.2 - 51.0) * 100)
    for y in range(water_y, height):
        # Add some variation for realism
        variation = int(np.sin(y * 0.1) * 10)
        blue = min(255, water_intensity + variation)
        green = min(255, water_intensity // 2 + variation)
        red = min(255, water_intensity // 4 + variation)
        img[y, :] = [blue, green, red]
    
    # Add measurement scale on right side
    scale_x = width - 80
    for i in range(8):
        scale_y = int(height * (0.2 + i * 0.1))
        scale_value = 52.5 - i * 0.25
        
        # Scale marks
        cv2.line(img, (scale_x, scale_y), (scale_x + 40, scale_y), (255, 255, 0), 2)
        
        # Scale text
        cv2.putText(img, f"{scale_value:.1f}", (scale_x - 60, scale_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add prominent water surface line
    cv2.line(img, (0, water_y), (width, water_y), (255, 255, 255), 4)
    
    # Add realistic noise and texture
    noise = np.random.randint(-20, 20, (height, width, 3))
    img = cv2.add(img, noise.astype(np.int8))
    
    # Add timestamp in corner
    cv2.putText(img, filename.replace('.jpg', ''), (10, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add distance indicator
    cv2.putText(img, f"Actual: {distance:.1f}", (10, height - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return img

def add_water_level_status_overlay(image, distance, threshold=51.4):
    """Add water level and status overlay to top-left corner."""
    img_copy = image.copy()
    
    # Determine status
    if distance > threshold:
        status = "HIGH RISK"
        status_color = (0, 0, 255)  # Red
        bg_color = (0, 0, 150)      # Dark red
    else:
        status = "SAFE"
        status_color = (0, 255, 0)  # Green
        bg_color = (0, 100, 0)      # Dark green
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    text_color = (255, 255, 255)
    
    # Text lines
    lines = [
        f"Water Level: {distance:.1f}",
        f"Status: {status}",
        f"Threshold: {threshold}"
    ]
    
    # Calculate overlay dimensions
    line_height = 40
    padding = 15
    max_width = 0
    
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_width)
    
    # Background rectangle
    bg_width = max_width + (padding * 2)
    bg_height = len(lines) * line_height + padding
    
    # Draw rounded background
    overlay = img_copy.copy()
    cv2.rectangle(overlay, (10, 10), (10 + bg_width, 10 + bg_height), bg_color, -1)
    alpha = 0.8
    img_copy = cv2.addWeighted(img_copy, 1 - alpha, overlay, alpha, 0)
    
    # Border
    cv2.rectangle(img_copy, (10, 10), (10 + bg_width, 10 + bg_height), (255, 255, 255), 3)
    
    # Add text
    for i, line in enumerate(lines):
        y_pos = 10 + padding + (i + 1) * line_height - 10
        
        # Use appropriate color
        if "Status:" in line:
            color = status_color
        else:
            color = text_color
        
        cv2.putText(img_copy, line, (10 + padding, y_pos), 
                   font, font_scale, color, thickness)
    
    return img_copy

def process_csv_matched_images():
    """Process the CSV-matched images with overlays."""
    print("\n" + "="*60)
    print("PROCESSING CSV-MATCHED IMAGES WITH OVERLAYS")
    print("="*60)
    
    # Create images
    created_images = create_realistic_flood_images_from_csv()
    
    # Process each image
    processed_images = []
    output_dir = "processed_csv_images"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_data in created_images:
        print(f"\nProcessing: {img_data['filename']}")
        
        # Load image
        img = cv2.imread(img_data['filepath'])
        if img is None:
            continue
        
        # Add overlay
        processed_img = add_water_level_status_overlay(img, img_data['distance'])
        
        # Save processed image
        output_filename = f"{os.path.splitext(img_data['filename'])[0]}_with_overlay.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, processed_img)
        
        processed_images.append({
            'original_filename': img_data['filename'],
            'original_path': img_data['filepath'],
            'processed_path': output_path,
            'distance': img_data['distance'],
            'status': 'HIGH RISK' if img_data['distance'] > 51.4 else 'SAFE'
        })
        
        print(f"  Distance: {img_data['distance']:.1f}")
        print(f"  Status: {'HIGH RISK' if img_data['distance'] > 51.4 else 'SAFE'}")
        print(f"  Saved: {output_path}")
    
    return processed_images

def create_before_after_comparison(processed_images):
    """Create before/after comparison showing original and processed images."""
    if not processed_images:
        return
    
    # Create comparison grid
    n_images = len(processed_images)
    fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Real CSV Data: Flood Images with Water Level & Status Overlay', 
                 fontsize=16, fontweight='bold')
    
    for i, img_data in enumerate(processed_images):
        # Load images
        original = cv2.imread(img_data['original_path'])
        processed = cv2.imread(img_data['processed_path'])
        
        if original is not None and processed is not None:
            # Convert BGR to RGB
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Original image
            axes[0, i].imshow(original_rgb)
            axes[0, i].set_title(f'Original\n{img_data["original_filename"]}', fontsize=10)
            axes[0, i].axis('off')
            
            # Processed image
            axes[1, i].imshow(processed_rgb)
            status_color = 'red' if img_data['status'] == 'HIGH RISK' else 'green'
            axes[1, i].set_title(f'With Overlay\nLevel: {img_data["distance"]:.1f} - {img_data["status"]}', 
                               fontsize=10, color=status_color, fontweight='bold')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_file = 'csv_images_before_after_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nBefore/After comparison saved as: {output_file}")
    plt.show()

def main():
    """Main demo function."""
    print("="*80)
    print("DEMO: REAL FLOOD IMAGES WITH WATER LEVEL & STATUS OVERLAY")
    print("="*80)
    print("This demo creates realistic flood images based on your CSV data")
    print("and adds water level and status overlays in the top-left corner.")
    
    # Process images
    processed_images = process_csv_matched_images()
    
    if processed_images:
        # Create comparison
        create_before_after_comparison(processed_images)
        
        # Print summary
        print(f"\n{'='*80}")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        high_risk = len([img for img in processed_images if img['status'] == 'HIGH RISK'])
        safe = len([img for img in processed_images if img['status'] == 'SAFE'])
        
        print(f"Images Created: {len(processed_images)}")
        print(f"High Risk (Red): {high_risk}")
        print(f"Safe (Green): {safe}")
        print(f"Threshold: 51.4")
        
        print(f"\nGenerated Files:")
        print(f"• csv_matched_images/ - Original realistic flood images")
        print(f"• processed_csv_images/ - Images with water level overlays")
        print(f"• csv_images_before_after_comparison.png - Before/after comparison")
        
        print(f"\nImage Details:")
        for img in processed_images:
            print(f"  {img['original_filename']}: Level {img['distance']:.1f} - {img['status']}")
        
        print(f"\n{'='*80}")
        print("TO USE WITH YOUR REAL IMAGES:")
        print("1. Place your flood monitoring images in a folder")
        print("2. Modify the script to point to your image folder")
        print("3. The script will match CSV filenames with actual images")
        print("4. Water level and status will be overlaid on each image")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()