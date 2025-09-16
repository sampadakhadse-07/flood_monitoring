#!/usr/bin/env python3
"""
Enhanced Google Drive Image Processor for Flood Monitoring
Automatically matches CSV timestamps with Google Drive images and processes them.
"""

import pandas as pd
import os
import requests
import re
import gdown
from pathlib import Path
from datetime import datetime
import time
from real_image_flood_analyzer import FloodImageAnalyzer

class EnhancedGoogleDriveProcessor:
    def __init__(self, csv_file_path="Flood_Data.csv", download_folder="google_drive_images"):
        """
        Initialize the enhanced processor.
        
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
        
        # Extract unique timestamps
        self.unique_timestamps = self.df['TimeStamp'].unique()
        print(f"Found {len(self.unique_timestamps)} unique timestamps")
    
    def extract_folder_id_from_url(self, drive_url):
        """
        Extract Google Drive folder ID from the URL.
        
        Args:
            drive_url (str): Google Drive folder URL
            
        Returns:
            str: Folder ID
        """
        # Extract folder ID from various Google Drive URL formats
        patterns = [
            r'/folders/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, drive_url)
            if match:
                return match.group(1)
        return None
    
    def download_folder_contents(self, drive_folder_url, max_files=None):
        """
        Download all images from a Google Drive folder that match CSV timestamps.
        
        Args:
            drive_folder_url (str): Google Drive folder URL
            max_files (int): Maximum number of files to download (None for all)
            
        Returns:
            tuple: (successful_downloads, failed_downloads)
        """
        folder_id = self.extract_folder_id_from_url(drive_folder_url)
        if not folder_id:
            print("Could not extract folder ID from URL")
            return [], []
        
        print(f"Processing Google Drive folder: {folder_id}")
        
        # Try to download the entire folder using gdown
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        try:
            print("Attempting to download folder contents using gdown...")
            # Download the folder to a temporary location
            temp_folder = "temp_drive_download"
            gdown.download_folder(folder_url, output=temp_folder, quiet=False, use_cookies=False)
            
            # Move relevant files to our download folder
            successful_downloads = []
            failed_downloads = []
            
            if os.path.exists(temp_folder):
                for root, dirs, files in os.walk(temp_folder):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            # Check if this file matches any of our CSV timestamps
                            if file in self.unique_timestamps:
                                src_path = os.path.join(root, file)
                                dest_path = os.path.join(self.download_folder, file)
                                
                                try:
                                    os.rename(src_path, dest_path)
                                    successful_downloads.append(file)
                                    print(f"✓ Downloaded: {file}")
                                    
                                    if max_files and len(successful_downloads) >= max_files:
                                        break
                                except Exception as e:
                                    failed_downloads.append(file)
                                    print(f"✗ Failed to move {file}: {e}")
                
                # Clean up temporary folder
                import shutil
                shutil.rmtree(temp_folder, ignore_errors=True)
                
            return successful_downloads, failed_downloads
            
        except Exception as e:
            print(f"Error downloading folder: {e}")
            print("Falling back to individual file download method...")
            return self.download_individual_files(folder_id, max_files)
    
    def download_individual_files(self, folder_id, max_files=None):
        """
        Fallback method to download individual files.
        This requires the files to be publicly accessible.
        
        Args:
            folder_id (str): Google Drive folder ID
            max_files (int): Maximum number of files to download
            
        Returns:
            tuple: (successful_downloads, failed_downloads)
        """
        print("Using individual file download method...")
        print("Note: This requires files to be publicly accessible.")
        
        successful_downloads = []
        failed_downloads = []
        
        # For demonstration, we'll try to download files that match our timestamps
        # In a real scenario, you'd need to use the Google Drive API to list folder contents
        for i, timestamp in enumerate(self.unique_timestamps):
            if max_files and i >= max_files:
                break
                
            # Try to guess the file ID or use a direct download approach
            # This is a simplified approach - in practice, you'd need the actual file IDs
            print(f"Would attempt to download: {timestamp}")
            # Add actual download logic here if file IDs are known
            
        return successful_downloads, failed_downloads
    
    def match_csv_with_images(self):
        """
        Match CSV timestamps with available local images.
        
        Returns:
            dict: Mapping of timestamp to analysis data
        """
        matched_data = {}
        available_images = []
        
        # Check which images are available locally
        for timestamp in self.unique_timestamps:
            image_path = os.path.join(self.download_folder, timestamp)
            if os.path.exists(image_path):
                available_images.append(timestamp)
                
                # Get all CSV rows for this timestamp
                csv_rows = self.df[self.df['TimeStamp'] == timestamp]
                matched_data[timestamp] = {
                    'image_path': image_path,
                    'csv_data': csv_rows.to_dict('records'),
                    'image_exists': True
                }
            else:
                matched_data[timestamp] = {
                    'image_path': image_path,
                    'csv_data': self.df[self.df['TimeStamp'] == timestamp].to_dict('records'),
                    'image_exists': False
                }
        
        print(f"Matched {len(available_images)} images with CSV data")
        print(f"Missing {len(self.unique_timestamps) - len(available_images)} images")
        
        return matched_data, available_images
    
    def analyze_matched_images(self):
        """
        Analyze all matched images and create comprehensive results.
        
        Returns:
            pd.DataFrame: Results with CSV data and analysis
        """
        matched_data, available_images = self.match_csv_with_images()
        
        if not available_images:
            print("No images available for analysis!")
            return None
        
        print(f"\nAnalyzing {len(available_images)} matched images...")
        
        results = []
        for i, timestamp in enumerate(available_images):
            print(f"Processing {i+1}/{len(available_images)}: {timestamp}")
            
            image_data = matched_data[timestamp]
            image_path = image_data['image_path']
            
            # Analyze the image
            try:
                analyzed_distance = self.analyzer.detect_water_level(image_path)
                
                # Process each CSV row for this timestamp
                for csv_row in image_data['csv_data']:
                    result = {
                        'TimeStamp': timestamp,
                        'CSV_LineCount': csv_row.get('LineCount'),
                        'CSV_Distance': csv_row.get('Distance'),
                        'Analyzed_Distance': analyzed_distance,
                        'Distance_Difference': (analyzed_distance - csv_row.get('Distance', 0)) if analyzed_distance else None,
                        'Flood_Status': self.analyzer.get_flood_status(analyzed_distance) if analyzed_distance else 'Analysis Failed',
                        'Risk_Level': self.analyzer.get_risk_level(analyzed_distance) if analyzed_distance else 'Unknown',
                        'Analysis_Success': analyzed_distance is not None,
                        'Image_Path': image_path
                    }
                    results.append(result)
                    
            except Exception as e:
                print(f"Error analyzing {timestamp}: {e}")
                # Add failed analysis record
                for csv_row in image_data['csv_data']:
                    result = {
                        'TimeStamp': timestamp,
                        'CSV_LineCount': csv_row.get('LineCount'),
                        'CSV_Distance': csv_row.get('Distance'),
                        'Analyzed_Distance': None,
                        'Distance_Difference': None,
                        'Flood_Status': 'Analysis Failed',
                        'Risk_Level': 'Unknown',
                        'Analysis_Success': False,
                        'Image_Path': image_path,
                        'Error': str(e)
                    }
                    results.append(result)
        
        # Create results DataFrame
        if results:
            results_df = pd.DataFrame(results)
            
            # Save comprehensive results
            output_file = "enhanced_drive_analysis_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            # Print summary
            self.print_analysis_summary(results_df)
            
            return results_df
        else:
            print("No results generated!")
            return None
    
    def print_analysis_summary(self, results_df):
        """
        Print a summary of the analysis results.
        
        Args:
            results_df (pd.DataFrame): Results DataFrame
        """
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        total_analyses = len(results_df)
        successful_analyses = results_df['Analysis_Success'].sum()
        failed_analyses = total_analyses - successful_analyses
        
        print(f"Total Images Analyzed: {total_analyses}")
        print(f"Successful Analyses: {successful_analyses}")
        print(f"Failed Analyses: {failed_analyses}")
        
        if successful_analyses > 0:
            successful_df = results_df[results_df['Analysis_Success'] == True]
            
            print(f"\nDistance Analysis (Successful only):")
            print(f"Average CSV Distance: {successful_df['CSV_Distance'].mean():.2f}")
            print(f"Average Analyzed Distance: {successful_df['Analyzed_Distance'].mean():.2f}")
            print(f"Average Distance Difference: {successful_df['Distance_Difference'].mean():.2f}")
            
            print(f"\nFlood Status Distribution:")
            status_counts = successful_df['Flood_Status'].value_counts()
            for status, count in status_counts.items():
                print(f"  {status}: {count} images")
            
            print(f"\nRisk Level Distribution:")
            risk_counts = successful_df['Risk_Level'].value_counts()
            for risk, count in risk_counts.items():
                print(f"  {risk}: {count} images")
        
        print("="*60)
    
    def create_missing_images_report(self):
        """
        Create a report of missing images that need to be downloaded.
        """
        matched_data, available_images = self.match_csv_with_images()
        missing_images = [ts for ts in self.unique_timestamps if ts not in available_images]
        
        if not missing_images:
            print("All required images are available!")
            return
        
        # Group missing images by date for easier organization
        missing_by_date = {}
        for timestamp in missing_images:
            if timestamp.endswith('.jpg'):
                date_part = timestamp.split('_')[0]
                if date_part not in missing_by_date:
                    missing_by_date[date_part] = []
                missing_by_date[date_part].append(timestamp)
        
        report = f"""
# Missing Images Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary:
- Total required images: {len(self.unique_timestamps)}
- Available images: {len(available_images)}
- Missing images: {len(missing_images)}

## Missing Images by Date:
"""
        
        for date, files in sorted(missing_by_date.items()):
            report += f"\n### {date} ({len(files)} files):\n"
            for file in sorted(files):
                report += f"- {file}\n"
        
        # Save report
        with open("missing_images_report.md", "w") as f:
            f.write(report)
        
        print(f"\nMissing images report saved to: missing_images_report.md")
        print(f"Please download {len(missing_images)} missing images")


def main():
    """
    Main function to demonstrate the enhanced processor.
    """
    print("Enhanced Google Drive Image Processor for Flood Monitoring")
    print("="*60)
    
    # Initialize processor
    processor = EnhancedGoogleDriveProcessor("Flood_Data.csv")
    
    # Google Drive folder URL (provided by user)
    drive_url = "https://drive.google.com/drive/folders/1vOCy_L8_yH-DQQ_O6QuaYfCUWxKSJ0ZN?usp=drive_link"
    
    print(f"\nStep 1: Attempting to download images from Google Drive...")
    print(f"Drive URL: {drive_url}")
    
    # Download images (limit to 10 for demo)
    successful, failed = processor.download_folder_contents(drive_url, max_files=10)
    
    print(f"\nDownload Results:")
    print(f"  Successfully downloaded: {len(successful)} files")
    print(f"  Failed downloads: {len(failed)} files")
    
    # Analyze available images
    print(f"\nStep 2: Analyzing matched images...")
    results = processor.analyze_matched_images()
    
    # Create missing images report
    print(f"\nStep 3: Generating missing images report...")
    processor.create_missing_images_report()
    
    if results is not None:
        print(f"\nProcessing complete! Check the output files for detailed results.")
    else:
        print(f"\nNo analysis results generated. Please check if images were downloaded successfully.")


if __name__ == "__main__":
    main()