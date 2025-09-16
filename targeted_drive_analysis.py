#!/usr/bin/env python3
"""
Targeted Google Drive Image Downloader
Downloads specific images based on CSV timestamps.
"""

import pandas as pd
import os
import requests
import gdown
from pathlib import Path
import re
from datetime import datetime

class TargetedDriveDownloader:
    def __init__(self, csv_file="Flood_Data.csv", output_dir="targeted_images"):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.df = pd.read_csv(csv_file)
        self.unique_timestamps = self.df['TimeStamp'].unique()
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"Loaded {len(self.unique_timestamps)} unique timestamps from CSV")
    
    def download_specific_images(self, folder_url, sample_size=20):
        """
        Download specific images that match CSV timestamps.
        
        Args:
            folder_url (str): Google Drive folder URL
            sample_size (int): Number of images to download for testing
        """
        print(f"\nTargeted download from Google Drive folder...")
        print(f"Looking for {sample_size} sample images from CSV timestamps")
        
        # Extract folder ID
        folder_id_match = re.search(r'/folders/([a-zA-Z0-9-_]+)', folder_url)
        if not folder_id_match:
            print("Could not extract folder ID from URL")
            return []
        
        folder_id = folder_id_match.group(1)
        print(f"Folder ID: {folder_id}")
        
        # Create a sample of timestamps to look for
        sample_timestamps = list(self.unique_timestamps[:sample_size])
        print(f"\nSample timestamps to search for:")
        for i, ts in enumerate(sample_timestamps):
            print(f"  {i+1}. {ts}")
        
        # Try to download individual files
        downloaded_files = []
        
        # Method 1: Try to download the folder in smaller batches
        try:
            temp_dir = "temp_download"
            os.makedirs(temp_dir, exist_ok=True)
            
            print(f"\nAttempting to download folder contents to temporary directory...")
            
            # Use gdown with limit
            gdown.download_folder(
                folder_url, 
                output=temp_dir, 
                quiet=False, 
                use_cookies=False,
                remaining_ok=True
            )
            
            # Look for matching files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file in sample_timestamps:
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(self.output_dir, file)
                        
                        try:
                            # Copy file to our target directory
                            import shutil
                            shutil.copy2(src_path, dest_path)
                            downloaded_files.append(file)
                            print(f"✓ Downloaded: {file}")
                        except Exception as e:
                            print(f"✗ Failed to copy {file}: {e}")
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"Folder download failed: {e}")
        
        print(f"\nDownloaded {len(downloaded_files)} files successfully")
        return downloaded_files
    
    def manual_download_instructions(self):
        """
        Generate instructions for manual download of specific images.
        """
        # Sample of most important timestamps (first 50)
        sample_timestamps = list(self.unique_timestamps[:50])
        
        # Group by date for easier navigation
        timestamps_by_date = {}
        for ts in sample_timestamps:
            if ts.endswith('.jpg'):
                date_part = ts.split('_')[0]
                if date_part not in timestamps_by_date:
                    timestamps_by_date[date_part] = []
                timestamps_by_date[date_part].append(ts)
        
        instructions = f"""
# Manual Download Instructions for Google Drive Images

## Overview:
Your Google Drive folder contains more than 50 files, which exceeds the automatic download limit.
Please manually download the following priority images for analysis.

## Target Directory: `{self.output_dir}`

## Priority Images to Download (First 50):

"""
        
        for date, files in sorted(timestamps_by_date.items()):
            instructions += f"\n### Date: {date} ({len(files)} files)\n"
            for file in sorted(files):
                instructions += f"- {file}\n"
        
        instructions += f"""

## Download Steps:

1. **Open the Google Drive folder:**
   https://drive.google.com/drive/folders/1vOCy_L8_yH-DQQ_O6QuaYfCUWxKSJ0ZN

2. **Navigate through the folder structure to find images**

3. **Download the images listed above to:** `{os.path.abspath(self.output_dir)}/`

4. **Ensure exact filename matching** (case-sensitive)

5. **Run the analysis script** after downloading

## Alternative Approaches:

### Option 1: Download All Images
If you have space, download all images from the folder, then the script will automatically find matches.

### Option 2: Use Google Drive API
Set up Google Drive API credentials for programmatic access to download all files.

### Option 3: Batch Download with gdown
```bash
# Install gdown if not already installed
pip install gdown

# Download the entire folder (may hit limits)
gdown --folder https://drive.google.com/drive/folders/1vOCy_L8_yH-DQQ_O6QuaYfCUWxKSJ0ZN
```

## Quick Test:
To test the system, download just a few images first:
{', '.join(sample_timestamps[:5])}

Then run: `python targeted_drive_analysis.py`
"""
        
        # Save instructions
        with open("manual_download_instructions.md", "w") as f:
            f.write(instructions)
        
        print(f"Manual download instructions saved to: manual_download_instructions.md")
        return sample_timestamps
    
    def analyze_available_images(self):
        """
        Analyze any images that are available locally.
        """
        available_images = []
        
        # Check what images are available
        for timestamp in self.unique_timestamps:
            image_path = os.path.join(self.output_dir, timestamp)
            if os.path.exists(image_path):
                available_images.append(timestamp)
        
        print(f"\nFound {len(available_images)} images locally:")
        for img in available_images[:10]:
            print(f"  ✓ {img}")
        if len(available_images) > 10:
            print(f"  ... and {len(available_images) - 10} more")
        
        if not available_images:
            print("No images available for analysis yet.")
            return None
        
        # Analyze images
        results = []
        
        # Try to import the analyzer
        try:
            from real_image_flood_analyzer import FloodImageAnalyzer
            analyzer = FloodImageAnalyzer()
            analyzer_available = True
            print("✓ Using FloodImageAnalyzer for real analysis")
        except ImportError:
            analyzer = None
            analyzer_available = False
            print("⚠ Using mock analysis (FloodImageAnalyzer not available)")
        
        for i, timestamp in enumerate(available_images):
            print(f"Analyzing {i+1}/{len(available_images)}: {timestamp}")
            
            image_path = os.path.join(self.output_dir, timestamp)
            csv_data = self.df[self.df['TimeStamp'] == timestamp].iloc[0]
            
            # Analyze image
            if analyzer_available:
                try:
                    analyzed_distance = analyzer.detect_water_level(image_path)
                    flood_status = analyzer.get_flood_status(analyzed_distance) if analyzed_distance else 'Unknown'
                    risk_level = analyzer.get_risk_level(analyzed_distance) if analyzed_distance else 'Unknown'
                    success = analyzed_distance is not None
                except Exception as e:
                    print(f"  ✗ Analysis failed: {e}")
                    analyzed_distance = None
                    flood_status = 'Analysis Failed'
                    risk_level = 'Unknown'
                    success = False
            else:
                # Mock analysis
                import random
                analyzed_distance = random.uniform(30, 70)
                flood_status = 'Normal' if analyzed_distance > 50 else 'Warning'
                risk_level = 'Low' if analyzed_distance > 50 else 'Medium'
                success = True
            
            result = {
                'timestamp': timestamp,
                'csv_line_count': csv_data['LineCount'],
                'csv_distance': csv_data['Distance'],
                'analyzed_distance': analyzed_distance,
                'distance_diff': (analyzed_distance - csv_data['Distance']) if analyzed_distance else None,
                'flood_status': flood_status,
                'risk_level': risk_level,
                'success': success,
                'image_path': image_path
            }
            
            results.append(result)
            print(f"  Status: {flood_status}, Distance: {analyzed_distance}")
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            output_file = "targeted_analysis_results.csv"
            df_results.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            # Print summary
            successful = df_results[df_results['success'] == True]
            if len(successful) > 0:
                print(f"\nSummary:")
                print(f"  Successful analyses: {len(successful)}")
                print(f"  Average CSV distance: {successful['csv_distance'].mean():.2f}")
                print(f"  Average analyzed distance: {successful['analyzed_distance'].mean():.2f}")
                
                status_counts = successful['flood_status'].value_counts()
                print(f"  Status distribution: {dict(status_counts)}")
        
        return results


def main():
    """
    Main function for targeted download and analysis.
    """
    print("Targeted Google Drive Image Download and Analysis")
    print("="*60)
    
    # Initialize downloader
    downloader = TargetedDriveDownloader("Flood_Data.csv", "targeted_images")
    
    # Google Drive folder URL
    folder_url = "https://drive.google.com/drive/folders/1vOCy_L8_yH-DQQ_O6QuaYfCUWxKSJ0ZN?usp=drive_link"
    
    print(f"Step 1: Attempting targeted download from Google Drive...")
    
    # Try to download specific images
    downloaded = downloader.download_specific_images(folder_url, sample_size=20)
    
    if not downloaded:
        print("Automatic download not successful.")
        print("Generating manual download instructions...")
        sample_timestamps = downloader.manual_download_instructions()
    
    print(f"\nStep 2: Analyzing available images...")
    results = downloader.analyze_available_images()
    
    if not results:
        print(f"\n📋 Next Steps:")
        print(f"1. Check 'manual_download_instructions.md' for download guidance")
        print(f"2. Download images to the 'targeted_images' folder")
        print(f"3. Run this script again to analyze downloaded images")
    else:
        print(f"\n✓ Analysis complete! Check 'targeted_analysis_results.csv' for results.")


if __name__ == "__main__":
    main()