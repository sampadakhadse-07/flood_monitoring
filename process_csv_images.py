#!/usr/bin/env python3
"""
Process Real Flood Images from CSV Data
This script matches your CSV data with actual flood monitoring images.
"""

import pandas as pd
import os
import glob
from real_image_flood_analyzer import FloodImageAnalyzer

def find_images_for_csv(csv_file='Flood_Data.csv', search_folders=None):
    """
    Find actual images that match the filenames in your CSV.
    """
    print("="*70)
    print("MATCHING CSV DATA WITH REAL IMAGES")
    print("="*70)
    
    # Load CSV data
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    print(f"CSV contains {len(df)} image references")
    
    # Get unique filenames
    filenames = df['TimeStamp'].dropna().unique()
    print(f"Unique image filenames: {len(filenames)}")
    
    # If no search folders provided, ask user
    if search_folders is None:
        print("\nWhere should I look for your images?")
        folder = input("Enter the path to your images folder: ").strip()
        if folder:
            search_folders = [folder]
        else:
            # Search in common locations
            search_folders = [
                ".",  # Current directory
                "images",
                "flood_images", 
                "data",
                "../images",
                "~/Downloads",
                "~/Pictures"
            ]
    
    # Search for images
    found_images = {}
    total_found = 0
    
    print(f"\nSearching for images in: {search_folders}")
    
    for folder in search_folders:
        if os.path.exists(folder):
            print(f"\nSearching in: {folder}")
            
            # Search for all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            
            for ext in image_extensions:
                pattern = os.path.join(folder, '**', ext)
                files = glob.glob(pattern, recursive=True)
                
                for file_path in files:
                    filename = os.path.basename(file_path)
                    if filename in filenames:
                        found_images[filename] = file_path
                        total_found += 1
            
            print(f"  Found {len([f for f in found_images.values() if folder in f])} matching images")
    
    print(f"\nSUMMARY:")
    print(f"  CSV references: {len(filenames)} images")
    print(f"  Found images: {total_found} images")
    print(f"  Missing images: {len(filenames) - total_found} images")
    
    # Show sample matches
    if found_images:
        print(f"\nSample found images:")
        for i, (filename, path) in enumerate(list(found_images.items())[:10]):
            print(f"  {filename} → {path}")
        if len(found_images) > 10:
            print(f"  ... and {len(found_images) - 10} more")
    
    return found_images, df

def process_real_flood_images(found_images, df, threshold=51.4):
    """
    Process the real flood images and compare with CSV data.
    """
    if not found_images:
        print("No images found to process!")
        return
    
    print(f"\nPROCESSING {len(found_images)} REAL FLOOD IMAGES")
    print("="*70)
    
    # Initialize analyzer
    analyzer = FloodImageAnalyzer(threshold=threshold)
    
    results = []
    processed_count = 0
    
    for filename, image_path in found_images.items():
        # Get CSV data for this image
        csv_row = df[df['TimeStamp'] == filename]
        
        if not csv_row.empty:
            csv_distance = float(csv_row['Distance'].iloc[0])
            csv_linecount = csv_row['LineCount'].iloc[0]
            
            print(f"Processing: {filename}")
            print(f"  CSV Distance: {csv_distance}")
            
            # Process with image analyzer
            result = analyzer.process_single_image(image_path)
            
            if result:
                # Compare detected vs CSV values
                detected_distance = result['Distance']
                error = abs(detected_distance - csv_distance)
                
                # Create comprehensive result
                combined_result = {
                    'TimeStamp': filename,
                    'CSV_Distance': csv_distance,
                    'CSV_LineCount': csv_linecount,
                    'Detected_Distance': detected_distance,
                    'Detection_Error': error,
                    'CSV_Status': 'High Risk (Red)' if csv_distance > threshold else 'Safe (Green)',
                    'Detected_Status': result['Status'],
                    'Status_Match': (csv_distance > threshold) == (detected_distance > threshold),
                    'Image_Path': image_path
                }
                
                results.append(combined_result)
                processed_count += 1
                
                print(f"  Detected Distance: {detected_distance}")
                print(f"  Error: {error:.2f}")
                print(f"  CSV Status: {combined_result['CSV_Status']}")
                print(f"  Detected Status: {combined_result['Detected_Status']}")
                print(f"  Status Match: {combined_result['Status_Match']}")
                print()
            else:
                print(f"  Failed to process image")
    
    print(f"Successfully processed {processed_count} images")
    
    if results:
        # Save results
        results_df = pd.DataFrame(results)
        output_file = 'real_flood_images_analysis.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        # Generate statistics
        print_analysis_statistics(results_df, threshold)
        
        return results_df
    
    return None

def print_analysis_statistics(df, threshold):
    """Print comprehensive analysis statistics."""
    print("\n" + "="*70)
    print("REAL IMAGE ANALYSIS STATISTICS")
    print("="*70)
    
    total_images = len(df)
    
    # Accuracy statistics
    avg_error = df['Detection_Error'].mean()
    max_error = df['Detection_Error'].max()
    min_error = df['Detection_Error'].min()
    
    # Status comparison
    csv_high_risk = len(df[df['CSV_Distance'] > threshold])
    detected_high_risk = len(df[df['Detected_Distance'] > threshold])
    status_matches = len(df[df['Status_Match'] == True])
    
    print(f"Detection Accuracy:")
    print(f"  Average Error: {avg_error:.2f}")
    print(f"  Maximum Error: {max_error:.2f}")
    print(f"  Minimum Error: {min_error:.2f}")
    
    print(f"\nRisk Classification Comparison:")
    print(f"  CSV High Risk: {csv_high_risk} images ({csv_high_risk/total_images*100:.1f}%)")
    print(f"  Detected High Risk: {detected_high_risk} images ({detected_high_risk/total_images*100:.1f}%)")
    print(f"  Status Matches: {status_matches} images ({status_matches/total_images*100:.1f}%)")
    
    print(f"\nDistance Comparison:")
    print(f"  CSV Average: {df['CSV_Distance'].mean():.2f}")
    print(f"  Detected Average: {df['Detected_Distance'].mean():.2f}")
    print(f"  CSV Range: {df['CSV_Distance'].min():.1f} - {df['CSV_Distance'].max():.1f}")
    print(f"  Detected Range: {df['Detected_Distance'].min():.1f} - {df['Detected_Distance'].max():.1f}")

def main():
    """Main function to process real flood images from CSV."""
    print("REAL FLOOD IMAGE PROCESSOR")
    print("This script will find and process your actual flood monitoring images")
    print("based on the filenames in your Flood_Data.csv file.")
    
    # Find images
    found_images, df = find_images_for_csv()
    
    if found_images:
        print(f"\nFound {len(found_images)} matching images!")
        
        # Ask if user wants to process them
        response = input("\nDo you want to process these images? (y/n): ").strip().lower()
        
        if response == 'y':
            # Process the images
            results_df = process_real_flood_images(found_images, df)
            
            if results_df is not None:
                print(f"\n" + "="*70)
                print("PROCESSING COMPLETED SUCCESSFULLY!")
                print("="*70)
                print("Generated files:")
                print("• real_flood_images_analysis.csv - Detailed comparison results")
                print("\nYou can now:")
                print("1. Review the CSV for detailed comparisons")
                print("2. Use the detected distances for further analysis")
                print("3. Calibrate the system for better accuracy")
        else:
            print("Processing cancelled.")
    else:
        print("\nNo matching images found.")
        print("\nTips:")
        print("1. Make sure your images are in an accessible folder")
        print("2. Check that filenames match exactly (case-sensitive)")
        print("3. Try organizing images in a single folder")
        print("4. Run with the correct image folder path")

if __name__ == "__main__":
    main()