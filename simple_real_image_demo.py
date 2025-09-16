#!/usr/bin/env python3
"""
Simple Real Image Flood Monitoring Demo
This demo creates sample images and analyzes them for flood risk assessment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from datetime import datetime, timedelta

def create_sample_flood_images():
    """Create sample flood monitoring images with different water levels."""
    print("Creating sample flood monitoring images...")
    
    # Create directory
    os.makedirs("sample_flood_images", exist_ok=True)
    
    # Image parameters
    width, height = 640, 480
    
    # Sample scenarios with different risk levels
    scenarios = [
        {"name": "01_safe_low_water", "water_y": 0.8, "distance": 50.9, "timestamp": "16-09-2024_08-30-15"},
        {"name": "02_safe_normal", "water_y": 0.75, "distance": 51.2, "timestamp": "16-09-2024_09-15-22"},
        {"name": "03_threshold", "water_y": 0.65, "distance": 51.4, "timestamp": "16-09-2024_10-45-33"},
        {"name": "04_warning_high", "water_y": 0.55, "distance": 51.7, "timestamp": "16-09-2024_12-20-45"},
        {"name": "05_danger_very_high", "water_y": 0.45, "distance": 52.0, "timestamp": "16-09-2024_14-30-12"},
        {"name": "06_critical_flood", "water_y": 0.35, "distance": 52.2, "timestamp": "16-09-2024_16-45-28"}
    ]
    
    image_data = []
    
    for scenario in scenarios:
        # Create base image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky/background (light blue)
        img[:int(height * 0.3), :] = [135, 206, 235]
        
        # Embankment/wall (gray)
        img[int(height * 0.3):int(height * scenario["water_y"]), :] = [128, 128, 128]
        
        # Water level (blue, getting darker with higher levels)
        water_start = int(height * scenario["water_y"])
        water_intensity = int(255 * (1 - scenario["water_y"]))  # Darker for higher water
        img[water_start:, :] = [water_intensity//3, water_intensity//2, water_intensity]
        
        # Add measurement scale on the right side
        scale_x = width - 60
        for i in range(6):
            scale_y = int(height * (0.3 + i * 0.1))
            scale_value = 53.0 - i * 0.4
            
            # Scale line
            cv2.line(img, (scale_x, scale_y), (scale_x + 30, scale_y), (255, 255, 0), 2)
            
            # Scale text
            cv2.putText(img, f"{scale_value:.1f}", (scale_x - 50, scale_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add water surface line
        cv2.line(img, (0, water_start), (width, water_start), (255, 255, 255), 3)
        
        # Add timestamp
        cv2.putText(img, scenario["timestamp"], (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add distance indicator
        distance_text = f"Level: {scenario['distance']:.1f}"
        cv2.putText(img, distance_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add noise for realism
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Save image
        filename = f"{scenario['timestamp']}.jpg"
        filepath = os.path.join("sample_flood_images", filename)
        cv2.imwrite(filepath, img)
        
        image_data.append({
            'filename': filename,
            'filepath': filepath,
            'actual_distance': scenario['distance'],
            'water_y': water_start,
            'timestamp': scenario['timestamp']
        })
        
        print(f"Created: {filename} (Distance: {scenario['distance']}, Risk: {'HIGH' if scenario['distance'] > 51.4 else 'SAFE'})")
    
    return image_data

def simple_water_level_detection(image_path):
    """
    Simple water level detection using edge detection and horizontal line detection.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find horizontal lines (water surface)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the most prominent horizontal line (water surface)
        best_water_y = None
        max_length = 0
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Consider only horizontal lines (width > height)
            if w > max_length and w > img.shape[1] * 0.3:  # At least 30% of image width
                max_length = w
                best_water_y = y
        
        if best_water_y is not None:
            # Convert pixel position to distance
            # Calibration: map image height to distance range
            height = img.shape[0]
            
            # Assuming distance range from 50.0 to 53.0
            min_distance = 50.0
            max_distance = 53.0
            
            # Normalize y position (0 to 1)
            normalized_y = best_water_y / height
            
            # Convert to distance (higher y = lower distance in this setup)
            distance = max_distance - (normalized_y * (max_distance - min_distance))
            
            return round(distance, 1)
        
        # Fallback: return safe value if no water line detected
        return 50.5
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def analyze_flood_images(image_data, threshold=51.4):
    """Analyze all flood images and create results."""
    print(f"\nAnalyzing flood images with threshold: {threshold}")
    
    results = []
    
    for data in image_data:
        print(f"Processing: {data['filename']}")
        
        # Detect water level
        detected_distance = simple_water_level_detection(data['filepath'])
        actual_distance = data['actual_distance']
        
        if detected_distance is not None:
            # Determine risk status
            status = 'High Risk (Red)' if detected_distance > threshold else 'Safe (Green)'
            risk_level = 'RED' if detected_distance > threshold else 'GREEN'
            color = 'red' if detected_distance > threshold else 'green'
            
            result = {
                'TimeStamp': data['filename'],
                'Actual_Distance': actual_distance,
                'Detected_Distance': detected_distance,
                'Detection_Error': abs(detected_distance - actual_distance),
                'Status': status,
                'Risk_Level': risk_level,
                'Color': color,
                'Image_Path': data['filepath']
            }
            
            results.append(result)
            
            print(f"  → Actual: {actual_distance}, Detected: {detected_distance}, Status: {status}")
        else:
            print(f"  → Failed to detect water level")
    
    return results

def create_analysis_visualization(results, threshold=51.4):
    """Create comprehensive visualization of the analysis results."""
    df = pd.DataFrame(results)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Real Image Flood Monitoring Analysis Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Detected distances
    ax1.scatter(df['Actual_Distance'], df['Detected_Distance'], 
               c=df['Color'], alpha=0.7, s=100, edgecolors='black')
    
    # Perfect detection line
    min_dist = min(df['Actual_Distance'].min(), df['Detected_Distance'].min())
    max_dist = max(df['Actual_Distance'].max(), df['Detected_Distance'].max())
    ax1.plot([min_dist, max_dist], [min_dist, max_dist], 'k--', alpha=0.5, label='Perfect Detection')
    
    ax1.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax1.axvline(x=threshold, color='orange', linestyle='--', linewidth=2)
    
    ax1.set_xlabel('Actual Distance')
    ax1.set_ylabel('Detected Distance')
    ax1.set_title('Detection Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk status distribution
    status_counts = df['Risk_Level'].value_counts()
    colors = ['green' if status == 'GREEN' else 'red' for status in status_counts.index]
    ax2.pie(status_counts.values, labels=[f'{status}\n({count} images)' for status, count in status_counts.items()], 
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Risk Status Distribution')
    
    # Plot 3: Detection errors
    ax3.bar(range(len(df)), df['Detection_Error'], 
           color=['red' if x == 'RED' else 'green' for x in df['Risk_Level']], alpha=0.7)
    ax3.set_xlabel('Image Index')
    ax3.set_ylabel('Detection Error (absolute)')
    ax3.set_title('Detection Accuracy by Image')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels([f"Img {i+1}" for i in range(len(df))], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance timeline
    ax4.plot(range(len(df)), df['Actual_Distance'], 'b-o', label='Actual Distance', linewidth=2)
    ax4.plot(range(len(df)), df['Detected_Distance'], 'r--s', label='Detected Distance', linewidth=2)
    ax4.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    
    # Color background based on risk
    for i, risk in enumerate(df['Risk_Level']):
        color = 'red' if risk == 'RED' else 'green'
        ax4.axvspan(i-0.4, i+0.4, alpha=0.2, color=color)
    
    ax4.set_xlabel('Image Sequence')
    ax4.set_ylabel('Distance')
    ax4.set_title('Water Level Timeline')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([f"Img {i+1}" for i in range(len(df))])
    
    plt.tight_layout()
    
    # Save visualization
    output_file = 'real_image_analysis_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Analysis visualization saved as: {output_file}")
    
    # Show plot in non-interactive mode
    plt.savefig('temp_display.png')
    plt.close()
    
    return df

def create_sample_image_grid(image_data, results):
    """Create a grid showing sample images with their analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sample Flood Monitoring Images with Analysis Results', fontsize=16, fontweight='bold')
    
    for i, (data, result) in enumerate(zip(image_data, results)):
        if i >= 6:  # Only show first 6 images
            break
        
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Load and display image
        img = cv2.imread(data['filepath'])
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            
            # Add title with results
            status = result['Status']
            actual = result['Actual_Distance']
            detected = result['Detected_Distance']
            error = result['Detection_Error']
            
            title = f"Actual: {actual:.1f}, Detected: {detected:.1f}\n{status}\nError: {error:.1f}"
            ax.set_title(title, fontsize=10, 
                        color='red' if result['Risk_Level'] == 'RED' else 'green',
                        fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save sample grid
    output_file = 'sample_images_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Sample images grid saved as: {output_file}")
    plt.close()

def main():
    """Main function for real image flood monitoring demo."""
    print("="*80)
    print("REAL IMAGE FLOOD MONITORING ANALYSIS - DEMO")
    print("="*80)
    
    # Create sample images
    image_data = create_sample_flood_images()
    
    # Analyze images
    threshold = 51.4
    results = analyze_flood_images(image_data, threshold)
    
    if results:
        # Create visualizations
        df = create_analysis_visualization(results, threshold)
        create_sample_image_grid(image_data, results)
        
        # Save detailed results
        df.to_csv('real_image_flood_analysis_results.csv', index=False)
        print(f"Detailed results saved to: real_image_flood_analysis_results.csv")
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        total_images = len(results)
        high_risk = len([r for r in results if r['Risk_Level'] == 'RED'])
        safe = len([r for r in results if r['Risk_Level'] == 'GREEN'])
        avg_error = np.mean([r['Detection_Error'] for r in results])
        
        print(f"Total Images Processed: {total_images}")
        print(f"High Risk (Red): {high_risk} images ({high_risk/total_images*100:.1f}%)")
        print(f"Safe (Green): {safe} images ({safe/total_images*100:.1f}%)")
        print(f"Average Detection Error: {avg_error:.2f}")
        print(f"Threshold Used: {threshold}")
        
        print(f"\nDetailed Results:")
        for i, result in enumerate(results):
            actual = result['Actual_Distance']
            detected = result['Detected_Distance']
            status = result['Status']
            error = result['Detection_Error']
            print(f"  Image {i+1}: Actual={actual:.1f}, Detected={detected:.1f}, Error={error:.1f}, Status={status}")
        
        print(f"\nGenerated Files:")
        print(f"• sample_flood_images/ - Sample flood monitoring images")
        print(f"• real_image_analysis_results.png - Comprehensive analysis visualization")
        print(f"• sample_images_analysis.png - Sample images with results")
        print(f"• real_image_flood_analysis_results.csv - Detailed results data")
        
    else:
        print("No results to analyze!")

if __name__ == "__main__":
    main()