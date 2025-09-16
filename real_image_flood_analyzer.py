#!/usr/bin/env python3
"""
Real-Time Flood Monitoring Analysis with Image Processing
This script processes real flood monitoring images and detects water levels with risk assessment.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob
from pathlib import Path

class FloodImageAnalyzer:
    def __init__(self, threshold=51.4):
        self.threshold = threshold
        self.results = []
        
    def detect_water_level(self, image_path):
        """
        Detect water level in flood monitoring image using computer vision techniques.
        This is a template method that can be customized based on your specific setup.
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                return None
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Method 1: Edge detection to find water surface
            edges = cv2.Canny(gray, 50, 150)
            
            # Method 2: Water detection using color thresholding
            # Adjust these HSV ranges based on your water color
            lower_water = np.array([100, 50, 50])  # Lower HSV for water
            upper_water = np.array([130, 255, 255])  # Upper HSV for water
            water_mask = cv2.inRange(hsv, lower_water, upper_water)
            
            # Method 3: Find horizontal lines (water surface typically appears as horizontal line)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Find the lowest water line (highest y-coordinate)
            contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the lowest point (water surface)
                water_level_y = 0
                for contour in contours:
                    for point in contour:
                        y = point[0][1]
                        if y > water_level_y:
                            water_level_y = y
                
                # Convert pixel position to distance measurement
                # This needs calibration based on your camera setup
                image_height = img.shape[0]
                distance = self.pixel_to_distance(water_level_y, image_height)
                
                return distance
            else:
                # Fallback: analyze water mask
                if np.sum(water_mask) > 0:
                    # Find the topmost water pixel
                    water_pixels = np.where(water_mask > 0)
                    if len(water_pixels[0]) > 0:
                        top_water_y = np.min(water_pixels[0])
                        distance = self.pixel_to_distance(top_water_y, img.shape[0])
                        return distance
                
                # If no water detected, return a default safe value
                return 50.0
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def pixel_to_distance(self, pixel_y, image_height):
        """
        Convert pixel position to distance measurement.
        This needs to be calibrated based on your specific camera setup and reference points.
        """
        # Example conversion - adjust based on your setup
        # Assuming the image shows a range from 50.0 to 53.0 distance units
        min_distance = 50.0
        max_distance = 53.0
        
        # Normalize pixel position (0 to 1)
        normalized_y = pixel_y / image_height
        
        # Convert to distance (inverse relationship - higher pixel = lower distance)
        distance = max_distance - (normalized_y * (max_distance - min_distance))
        
        return round(distance, 1)
    
    def process_single_image(self, image_path):
        """Process a single image and return analysis results."""
        filename = os.path.basename(image_path)
        print(f"Processing: {filename}")
        
        # Detect water level
        distance = self.detect_water_level(image_path)
        
        if distance is not None:
            # Determine risk status
            status = 'High Risk (Red)' if distance > self.threshold else 'Safe (Green)'
            risk_level = 'RED' if distance > self.threshold else 'GREEN'
            color = 'red' if distance > self.threshold else 'green'
            
            # Extract timestamp from filename if possible
            timestamp = self.extract_timestamp(filename)
            
            result = {
                'TimeStamp': filename,
                'DateTime': timestamp,
                'Distance': distance,
                'Status': status,
                'Risk_Level': risk_level,
                'Color': color,
                'Image_Path': image_path
            }
            
            self.results.append(result)
            print(f"  → Distance: {distance}, Status: {status}")
            return result
        else:
            print(f"  → Failed to process image")
            return None
    
    def extract_timestamp(self, filename):
        """Extract timestamp from filename if it follows standard format."""
        try:
            # Try to extract timestamp from format: DD-MM-YYYY_HH-MM-SS.jpg
            if '_' in filename and '-' in filename:
                parts = filename.replace('.jpg', '').replace('.JPG', '').split('_')
                if len(parts) >= 2:
                    date_part = parts[0]
                    time_part = parts[1]
                    datetime_str = f"{date_part} {time_part.replace('-', ':')}"
                    return pd.to_datetime(datetime_str, format='%d-%m-%Y %H:%M:%S', errors='coerce')
        except:
            pass
        return None
    
    def process_image_folder(self, folder_path):
        """Process all images in a folder."""
        print(f"Processing images in: {folder_path}")
        
        # Supported image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
        
        print(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("No images found in the specified folder!")
            return
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"Progress: {i+1}/{len(image_files)}")
            self.process_single_image(image_path)
        
        print(f"Completed processing {len(self.results)} images successfully")
    
    def create_annotated_image(self, image_path, result):
        """Create an annotated version of the image with risk status."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Add annotations
            height, width = img_rgb.shape[:2]
            
            # Choose color based on risk
            color = (255, 0, 0) if result['Risk_Level'] == 'RED' else (0, 255, 0)  # RGB
            
            # Add text annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Status text
            status_text = f"Status: {result['Status']}"
            cv2.putText(img_rgb, status_text, (10, 30), font, font_scale, color, thickness)
            
            # Distance text
            distance_text = f"Water Level: {result['Distance']}"
            cv2.putText(img_rgb, distance_text, (10, 70), font, font_scale, color, thickness)
            
            # Threshold line
            threshold_text = f"Threshold: {self.threshold}"
            cv2.putText(img_rgb, threshold_text, (10, 110), font, font_scale, (255, 165, 0), thickness)
            
            # Add risk indicator border
            border_thickness = 10
            cv2.rectangle(img_rgb, (0, 0), (width-1, height-1), color, border_thickness)
            
            return img_rgb
            
        except Exception as e:
            print(f"Error creating annotated image: {str(e)}")
            return None
    
    def save_results(self, output_csv='real_image_flood_analysis.csv'):
        """Save analysis results to CSV."""
        if not self.results:
            print("No results to save!")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
        return df
    
    def create_comprehensive_report(self, df):
        """Create comprehensive analysis report with visualizations."""
        print("\n" + "="*70)
        print("REAL IMAGE FLOOD MONITORING ANALYSIS REPORT")
        print("="*70)
        
        # Statistics
        total_images = len(df)
        high_risk_count = len(df[df['Risk_Level'] == 'RED'])
        safe_count = len(df[df['Risk_Level'] == 'GREEN'])
        
        print(f"Total Images Processed: {total_images}")
        print(f"High Risk (Red): {high_risk_count} images ({high_risk_count/total_images*100:.1f}%)")
        print(f"Safe (Green): {safe_count} images ({safe_count/total_images*100:.1f}%)")
        print(f"Threshold: {self.threshold}")
        
        if 'Distance' in df.columns:
            print(f"\nWater Level Statistics:")
            print(f"  Average: {df['Distance'].mean():.2f}")
            print(f"  Minimum: {df['Distance'].min():.2f}")
            print(f"  Maximum: {df['Distance'].max():.2f}")
            print(f"  Std Dev: {df['Distance'].std():.2f}")
        
        # Create visualization
        self.create_real_image_visualization(df)
    
    def create_real_image_visualization(self, df):
        """Create visualization for real image analysis results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Real Image Flood Monitoring Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Distance distribution with risk colors
        ax1.hist(df[df['Risk_Level'] == 'GREEN']['Distance'], bins=15, alpha=0.7, 
                color='green', label='Safe (Green)', edgecolor='black')
        ax1.hist(df[df['Risk_Level'] == 'RED']['Distance'], bins=15, alpha=0.7, 
                color='red', label='High Risk (Red)', edgecolor='black')
        ax1.axvline(x=self.threshold, color='orange', linestyle='--', linewidth=2, 
                   label=f'Threshold ({self.threshold})')
        ax1.set_xlabel('Water Level (Distance)')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Water Level Distribution from Images')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Risk status pie chart
        status_counts = df['Risk_Level'].value_counts()
        colors = ['green' if status == 'GREEN' else 'red' for status in status_counts.index]
        ax2.pie(status_counts.values, labels=[f'{status} ({count})' for status, count in status_counts.items()], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Risk Status Distribution')
        
        # Plot 3: Time series if datetime available
        if 'DateTime' in df.columns and df['DateTime'].notna().any():
            df_time = df.dropna(subset=['DateTime']).sort_values('DateTime')
            colors = ['red' if x == 'RED' else 'green' for x in df_time['Risk_Level']]
            ax3.scatter(df_time['DateTime'], df_time['Distance'], c=colors, alpha=0.7)
            ax3.axhline(y=self.threshold, color='orange', linestyle='--', linewidth=2)
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Water Level')
            ax3.set_title('Water Level Over Time')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        else:
            # Alternative: Distance vs Image index
            colors = ['red' if x == 'RED' else 'green' for x in df['Risk_Level']]
            ax3.scatter(range(len(df)), df['Distance'], c=colors, alpha=0.7)
            ax3.axhline(y=self.threshold, color='orange', linestyle='--', linewidth=2)
            ax3.set_xlabel('Image Index')
            ax3.set_ylabel('Water Level')
            ax3.set_title('Water Level by Image Sequence')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Risk assessment summary
        ax4.axis('off')
        summary_text = f"""
ANALYSIS SUMMARY

Total Images: {len(df)}
Threshold: {self.threshold}

Risk Distribution:
• High Risk: {len(df[df['Risk_Level'] == 'RED'])} images
• Safe: {len(df[df['Risk_Level'] == 'GREEN'])} images

Water Level Stats:
• Average: {df['Distance'].mean():.2f}
• Range: {df['Distance'].min():.2f} - {df['Distance'].max():.2f}
• Std Dev: {df['Distance'].std():.2f}

Risk Percentage:
• High Risk: {len(df[df['Risk_Level'] == 'RED'])/len(df)*100:.1f}%
• Safe: {len(df[df['Risk_Level'] == 'GREEN'])/len(df)*100:.1f}%
        """
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        output_file = 'real_image_flood_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as: {output_file}")
        plt.show()

def main():
    """Main function for real image flood monitoring analysis."""
    print("="*70)
    print("REAL IMAGE FLOOD MONITORING ANALYSIS")
    print("="*70)
    
    # Initialize analyzer with threshold
    threshold = 51.4  # Adjust based on your requirements
    analyzer = FloodImageAnalyzer(threshold=threshold)
    
    # Option 1: Process images from a folder
    print("\nChoose processing option:")
    print("1. Process images from a folder")
    print("2. Process a single image")
    print("3. Use sample/demo mode")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        folder_path = input("Enter the path to your images folder: ").strip()
        if os.path.exists(folder_path):
            analyzer.process_image_folder(folder_path)
        else:
            print(f"Folder not found: {folder_path}")
            print("Using demo mode with sample data...")
            create_demo_analysis(analyzer)
    
    elif choice == "2":
        image_path = input("Enter the path to your image: ").strip()
        if os.path.exists(image_path):
            result = analyzer.process_single_image(image_path)
            if result:
                # Create annotated image
                annotated = analyzer.create_annotated_image(image_path, result)
                if annotated is not None:
                    plt.figure(figsize=(12, 8))
                    plt.imshow(annotated)
                    plt.title(f"Flood Analysis: {result['Status']}")
                    plt.axis('off')
                    plt.show()
        else:
            print(f"Image not found: {image_path}")
    
    else:
        print("Using demo mode with sample data...")
        create_demo_analysis(analyzer)
    
    # Save and analyze results
    if analyzer.results:
        df = analyzer.save_results()
        analyzer.create_comprehensive_report(df)
        
        # Show sample results
        print("\nSample Results:")
        print(df[['TimeStamp', 'Distance', 'Status', 'Risk_Level']].head(10).to_string(index=False))
    else:
        print("No results to analyze!")

def create_demo_analysis(analyzer):
    """Create demo analysis with simulated data for testing."""
    print("Creating demo analysis with simulated image data...")
    
    # Simulate processing results for demo
    demo_data = [
        {'TimeStamp': 'demo_image_001.jpg', 'Distance': 50.8, 'Status': 'Safe (Green)', 'Risk_Level': 'GREEN'},
        {'TimeStamp': 'demo_image_002.jpg', 'Distance': 51.8, 'Status': 'High Risk (Red)', 'Risk_Level': 'RED'},
        {'TimeStamp': 'demo_image_003.jpg', 'Distance': 51.2, 'Status': 'Safe (Green)', 'Risk_Level': 'GREEN'},
        {'TimeStamp': 'demo_image_004.jpg', 'Distance': 52.1, 'Status': 'High Risk (Red)', 'Risk_Level': 'RED'},
        {'TimeStamp': 'demo_image_005.jpg', 'Distance': 50.9, 'Status': 'Safe (Green)', 'Risk_Level': 'GREEN'},
    ]
    
    analyzer.results = demo_data
    print(f"Demo: Processed {len(demo_data)} simulated images")

if __name__ == "__main__":
    main()