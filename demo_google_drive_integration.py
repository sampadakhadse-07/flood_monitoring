#!/usr/bin/env python3
"""
Simple Google Drive Flood Monitoring Demo
Demonstrates CSV timestamp matching and image processing workflow.
"""

import pandas as pd
import os
import requests
import gdown
from pathlib import Path
from datetime import datetime

def simple_drive_folder_download(folder_url, output_dir="google_drive_images", limit=5):
    """
    Simple function to download images from a Google Drive folder.
    
    Args:
        folder_url (str): Google Drive folder URL
        output_dir (str): Directory to save images
        limit (int): Maximum number of files to download
    
    Returns:
        list: List of downloaded files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Attempting to download from: {folder_url}")
    print(f"Output directory: {output_dir}")
    print(f"Download limit: {limit} files")
    
    try:
        # Use gdown to download the folder
        print("Downloading folder contents...")
        gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
        
        # List downloaded files
        downloaded_files = []
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    downloaded_files.append(file)
                    print(f"✓ Found: {file}")
        
        return downloaded_files[:limit] if limit else downloaded_files
        
    except Exception as e:
        print(f"Error downloading folder: {e}")
        print("\nAlternative approaches:")
        print("1. Make sure the Google Drive folder is publicly accessible")
        print("2. Download images manually to the 'google_drive_images' folder")
        print("3. Use Google Drive API with proper authentication")
        return []

def match_timestamps_with_csv(csv_file, image_dir="google_drive_images"):
    """
    Match available images with CSV timestamps.
    
    Args:
        csv_file (str): Path to CSV file
        image_dir (str): Directory containing images
    
    Returns:
        tuple: (matched_data, available_images, missing_images)
    """
    print(f"\nMatching images with CSV data from: {csv_file}")
    
    # Load CSV
    df = pd.read_csv(csv_file)
    unique_timestamps = df['TimeStamp'].unique()
    print(f"CSV contains {len(unique_timestamps)} unique timestamps")
    
    # Check available images
    available_images = []
    missing_images = []
    
    for timestamp in unique_timestamps:
        image_path = os.path.join(image_dir, timestamp)
        if os.path.exists(image_path):
            available_images.append(timestamp)
        else:
            missing_images.append(timestamp)
    
    print(f"Available images: {len(available_images)}")
    print(f"Missing images: {len(missing_images)}")
    
    # Create matched data
    matched_data = {}
    for timestamp in available_images:
        csv_rows = df[df['TimeStamp'] == timestamp]
        matched_data[timestamp] = {
            'image_path': os.path.join(image_dir, timestamp),
            'csv_data': csv_rows.iloc[0].to_dict() if len(csv_rows) > 0 else {}
        }
    
    return matched_data, available_images, missing_images

def analyze_sample_images(matched_data):
    """
    Analyze a sample of matched images.
    
    Args:
        matched_data (dict): Matched image and CSV data
    
    Returns:
        list: Analysis results
    """
    print(f"\nAnalyzing {len(matched_data)} matched images...")
    
    results = []
    
    # Import the analyzer if available
    try:
        from real_image_flood_analyzer import FloodImageAnalyzer
        analyzer = FloodImageAnalyzer()
        analyzer_available = True
        print("✓ FloodImageAnalyzer loaded successfully")
    except ImportError:
        print("⚠ FloodImageAnalyzer not available - using mock analysis")
        analyzer = None
        analyzer_available = False
    
    for i, (timestamp, data) in enumerate(matched_data.items()):
        print(f"Processing {i+1}/{len(matched_data)}: {timestamp}")
        
        image_path = data['image_path']
        csv_data = data['csv_data']
        
        # Analyze image
        if analyzer_available and analyzer:
            try:
                analyzed_distance = analyzer.detect_water_level(image_path)
                flood_status = analyzer.get_flood_status(analyzed_distance) if analyzed_distance else 'Unknown'
                risk_level = analyzer.get_risk_level(analyzed_distance) if analyzed_distance else 'Unknown'
                analysis_success = analyzed_distance is not None
            except Exception as e:
                print(f"  ✗ Analysis failed: {e}")
                analyzed_distance = None
                flood_status = 'Analysis Failed'
                risk_level = 'Unknown'
                analysis_success = False
        else:
            # Mock analysis for demo
            import random
            analyzed_distance = random.uniform(30, 70)  # Mock distance
            flood_status = 'Normal' if analyzed_distance > 50 else 'Warning'
            risk_level = 'Low' if analyzed_distance > 50 else 'Medium'
            analysis_success = True
            print(f"  ℹ Mock analysis: distance = {analyzed_distance:.1f}")
        
        # Compile results
        result = {
            'timestamp': timestamp,
            'csv_line_count': csv_data.get('LineCount', 'N/A'),
            'csv_distance': csv_data.get('Distance', 'N/A'),
            'analyzed_distance': analyzed_distance,
            'distance_difference': (analyzed_distance - csv_data.get('Distance', 0)) if analyzed_distance and 'Distance' in csv_data else None,
            'flood_status': flood_status,
            'risk_level': risk_level,
            'analysis_success': analysis_success,
            'image_path': image_path
        }
        
        results.append(result)
        print(f"  ✓ Status: {flood_status}, Risk: {risk_level}")
    
    return results

def generate_demo_report(results, available_images, missing_images):
    """
    Generate a comprehensive demo report.
    
    Args:
        results (list): Analysis results
        available_images (list): List of available images
        missing_images (list): List of missing images
    """
    print(f"\n" + "="*60)
    print("DEMO ANALYSIS REPORT")
    print("="*60)
    
    # Summary
    print(f"Images Available: {len(available_images)}")
    print(f"Images Missing: {len(missing_images)}")
    print(f"Images Analyzed: {len(results)}")
    
    if results:
        # Analysis summary
        successful_analyses = [r for r in results if r['analysis_success']]
        print(f"Successful Analyses: {len(successful_analyses)}")
        
        if successful_analyses:
            avg_csv_distance = sum(r['csv_distance'] for r in successful_analyses if isinstance(r['csv_distance'], (int, float))) / len(successful_analyses)
            avg_analyzed_distance = sum(r['analyzed_distance'] for r in successful_analyses if r['analyzed_distance']) / len(successful_analyses)
            
            print(f"\nDistance Analysis:")
            print(f"Average CSV Distance: {avg_csv_distance:.2f}")
            print(f"Average Analyzed Distance: {avg_analyzed_distance:.2f}")
            
            # Status distribution
            status_counts = {}
            risk_counts = {}
            for result in successful_analyses:
                status = result['flood_status']
                risk = result['risk_level']
                status_counts[status] = status_counts.get(status, 0) + 1
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            print(f"\nFlood Status Distribution:")
            for status, count in status_counts.items():
                print(f"  {status}: {count} images")
            
            print(f"\nRisk Level Distribution:")
            for risk, count in risk_counts.items():
                print(f"  {risk}: {count} images")
    
    # Save detailed results to CSV
    if results:
        df_results = pd.DataFrame(results)
        output_file = "demo_analysis_results.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    # Missing images report
    if missing_images:
        print(f"\n⚠ Missing Images ({len(missing_images)}):")
        for i, img in enumerate(missing_images[:10]):  # Show first 10
            print(f"  {i+1}. {img}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
    
    print("="*60)


def main():
    """
    Main demo function.
    """
    print("Google Drive Flood Monitoring Demo")
    print("="*50)
    
    # Configuration
    drive_url = "https://drive.google.com/drive/folders/1vOCy_L8_yH-DQQ_O6QuaYfCUWxKSJ0ZN?usp=drive_link"
    csv_file = "Flood_Data.csv"
    image_dir = "google_drive_images"
    
    print(f"Drive URL: {drive_url}")
    print(f"CSV File: {csv_file}")
    print(f"Image Directory: {image_dir}")
    
    # Step 1: Download images from Google Drive
    print(f"\nStep 1: Downloading images from Google Drive...")
    downloaded_files = simple_drive_folder_download(drive_url, image_dir, limit=10)
    
    if not downloaded_files:
        print("No images downloaded. Using existing images in folder if available...")
    
    # Step 2: Match timestamps with CSV
    print(f"\nStep 2: Matching images with CSV timestamps...")
    matched_data, available_images, missing_images = match_timestamps_with_csv(csv_file, image_dir)
    
    if not available_images:
        print("No images available for analysis!")
        print(f"Please ensure images are in the '{image_dir}' folder")
        return
    
    # Step 3: Analyze matched images
    print(f"\nStep 3: Analyzing matched images...")
    results = analyze_sample_images(matched_data)
    
    # Step 4: Generate report
    print(f"\nStep 4: Generating demo report...")
    generate_demo_report(results, available_images, missing_images)
    
    print(f"\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()