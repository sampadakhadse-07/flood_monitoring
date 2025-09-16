#!/usr/bin/env python3
"""
Google Drive Image Processor for Flood Monitoring
This script helps process images from Google Drive based on timestamps in CSV files.
"""

import pandas as pd
import os
import requests
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import time
from real_image_flood_analyzer import FloodImageAnalyzer

class GoogleDriveImageProcessor:
    def __init__(self, csv_file_path, download_folder="downloaded_images"):
        """
        Initialize the processor with CSV file and download folder.
        
        Args:
            csv_file_path (str): Path to the CSV file containing timestamps
            download_folder (str): Folder to store downloaded images
        """
        self.csv_file_path = csv_file_path
        self.download_folder = download_folder
        self.analyzer = FloodImageAnalyzer()
        
        # Create download folder if it doesn't exist
        os.makedirs(self.download_folder, exist_ok=True)
        
        # Load CSV data
        self.df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(self.df)} records from {csv_file_path}")
    
    def extract_file_id_from_url(self, drive_url):
        """
        Extract Google Drive file ID from various Google Drive URL formats.
        
        Args:
            drive_url (str): Google Drive URL
            
        Returns:
            str: File ID if found, None otherwise
        """
        # Handle different Google Drive URL formats
        patterns = [
            r'/file/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/folders/([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drive_url)
            if match:
                return match.group(1)
        return None
    
    def get_drive_folder_contents(self, folder_id):
        """
        Get the list of files in a Google Drive folder (requires public access).
        Note: This is a simplified approach. For production use, implement proper Google Drive API.
        
        Args:
            folder_id (str): Google Drive folder ID
            
        Returns:
            list: List of file information
        """
        # This is a placeholder for Google Drive API integration
        # For now, we'll work with locally downloaded files
        print(f"Note: To fully integrate with Google Drive folder {folder_id},")
        print("you would need to set up Google Drive API authentication.")
        print("For now, please manually download images to the 'downloaded_images' folder.")
        return []
    
    def download_image_from_drive(self, file_id, filename):
        """
        Download an image from Google Drive using file ID.
        
        Args:
            file_id (str): Google Drive file ID
            filename (str): Local filename to save as
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Google Drive direct download URL
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                file_path = os.path.join(self.download_folder, filename)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded: {filename}")
                return True
            else:
                print(f"Failed to download {filename}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            return False
    
    def process_csv_timestamps(self):
        """
        Process all timestamps from CSV and prepare for image analysis.
        
        Returns:
            list: List of timestamps that need images
        """
        timestamps = self.df['TimeStamp'].tolist()
        unique_timestamps = list(set(timestamps))
        
        print(f"Found {len(unique_timestamps)} unique image timestamps:")
        for i, timestamp in enumerate(unique_timestamps[:10]):  # Show first 10
            print(f"  {i+1}. {timestamp}")
        
        if len(unique_timestamps) > 10:
            print(f"  ... and {len(unique_timestamps) - 10} more")
        
        return unique_timestamps
    
    def check_local_images(self):
        """
        Check which images are already available locally.
        
        Returns:
            tuple: (available_images, missing_images)
        """
        timestamps = self.process_csv_timestamps()
        available_images = []
        missing_images = []
        
        for timestamp in timestamps:
            file_path = os.path.join(self.download_folder, timestamp)
            if os.path.exists(file_path):
                available_images.append(timestamp)
            else:
                missing_images.append(timestamp)
        
        print(f"\nLocal image status:")
        print(f"  Available: {len(available_images)} images")
        print(f"  Missing: {len(missing_images)} images")
        
        return available_images, missing_images
    
    def analyze_available_images(self):
        """
        Analyze all locally available images and update results.
        
        Returns:
            pd.DataFrame: Results dataframe with analysis
        """
        available_images, _ = self.check_local_images()
        
        if not available_images:
            print("No images available for analysis. Please download images first.")
            return None
        
        print(f"\nAnalyzing {len(available_images)} available images...")
        
        results = []
        for i, timestamp in enumerate(available_images):
            print(f"Processing {i+1}/{len(available_images)}: {timestamp}")
            
            image_path = os.path.join(self.download_folder, timestamp)
            distance = self.analyzer.detect_water_level(image_path)
            
            if distance is not None:
                # Get the corresponding CSV data
                csv_row = self.df[self.df['TimeStamp'] == timestamp].iloc[0]
                
                result = {
                    'TimeStamp': timestamp,
                    'CSV_Distance': csv_row['Distance'] if 'Distance' in csv_row else None,
                    'Analyzed_Distance': distance,
                    'LineCount': csv_row['LineCount'] if 'LineCount' in csv_row else None,
                    'Status': self.analyzer.get_flood_status(distance),
                    'Risk_Level': self.analyzer.get_risk_level(distance)
                }
                results.append(result)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Save results
        output_file = "google_drive_image_analysis_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return results_df
    
    def create_download_instructions(self, drive_url):
        """
        Create instructions for manually downloading images from Google Drive.
        
        Args:
            drive_url (str): Google Drive folder URL
        """
        folder_id = self.extract_file_id_from_url(drive_url)
        timestamps = self.process_csv_timestamps()
        
        instructions = f"""
# Google Drive Image Download Instructions

## Drive Folder Information:
- URL: {drive_url}
- Folder ID: {folder_id}
- Local Download Folder: {self.download_folder}

## Required Images ({len(timestamps)} total):
The following images need to be downloaded from the Google Drive folder:

"""
        
        # Group timestamps by date for easier organization
        timestamps_by_date = {}
        for timestamp in timestamps:
            if timestamp.endswith('.jpg'):
                date_part = timestamp.split('_')[0]
                if date_part not in timestamps_by_date:
                    timestamps_by_date[date_part] = []
                timestamps_by_date[date_part].append(timestamp)
        
        for date, files in sorted(timestamps_by_date.items()):
            instructions += f"\n### {date} ({len(files)} files):\n"
            for file in sorted(files):
                instructions += f"- {file}\n"
        
        instructions += f"""

## Download Steps:
1. Open the Google Drive folder: {drive_url}
2. Download all the images listed above
3. Place them in the folder: {os.path.abspath(self.download_folder)}
4. Ensure filenames match exactly (including case sensitivity)
5. Run the analysis script to process the images

## Alternative Approach:
If you have access to Google Drive API:
1. Set up Google Drive API credentials
2. Use the batch download functionality in this script
3. Or use the Google Drive API directly

## Quick Download Commands:
For Linux/Mac users with gdown installed:
```bash
pip install gdown
# Then download specific files using their individual file IDs
```
"""
        
        # Save instructions to file
        with open("download_instructions.md", "w") as f:
            f.write(instructions)
        
        print("\nDownload instructions saved to: download_instructions.md")
        print(f"Please download {len(timestamps)} images to: {os.path.abspath(self.download_folder)}")


def main():
    """
    Main function to demonstrate usage.
    """
    print("Google Drive Image Processor for Flood Monitoring")
    print("=" * 50)
    
    # Initialize processor
    processor = GoogleDriveImageProcessor("flood_data_labeled.csv")
    
    # Google Drive folder URL
    drive_url = "https://drive.google.com/drive/folders/1vOCy_L8_yH-DQQ_O6QuaYfCUWxKSJ0ZN?usp=drive_link"
    
    # Create download instructions
    processor.create_download_instructions(drive_url)
    
    # Check what images are available locally
    available, missing = processor.check_local_images()
    
    if available:
        print(f"\nFound {len(available)} images locally. Starting analysis...")
        results = processor.analyze_available_images()
        
        if results is not None:
            print("\nAnalysis Summary:")
            print(results.describe())
    else:
        print(f"\nNo images found locally in '{processor.download_folder}'")
        print("Please follow the download instructions in 'download_instructions.md'")


if __name__ == "__main__":
    main()