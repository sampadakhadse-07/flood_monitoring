#!/usr/bin/env python3
"""
Example usage of the Real Image Flood Monitoring System
This script demonstrates how to use the flood monitoring system with real images.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from real_image_flood_analyzer import FloodImageAnalyzer

def create_sample_flood_images():
    """
    Create sample flood monitoring images for demonstration.
    This simulates different water levels in monitoring images.
    """
    print("Creating sample flood monitoring images...")
    
    # Create sample images directory
    os.makedirs("sample_flood_images", exist_ok=True)
    
    # Image parameters
    width, height = 640, 480
    
    sample_data = [
        {"filename": "low_water_safe.jpg", "water_level": 0.7, "risk": "safe"},
        {"filename": "medium_water_warning.jpg", "water_level": 0.5, "risk": "warning"},
        {"filename": "high_water_danger.jpg", "water_level": 0.3, "risk": "danger"},
        {"filename": "very_high_water_critical.jpg", "water_level": 0.2, "risk": "critical"},
        {"filename": "normal_level.jpg", "water_level": 0.8, "risk": "safe"},
    ]
    
    for data in sample_data:
        # Create a simple flood monitoring image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background (sky/environment)
        img[:int(height * 0.3), :] = [135, 206, 235]  # Sky blue
        
        # Water level
        water_start_y = int(height * data["water_level"])
        
        # Water area
        img[water_start_y:, :] = [65, 105, 225]  # Royal blue for water
        
        # Add some noise and texture
        noise = np.random.randint(0, 30, (height, width, 3))
        img = cv2.add(img, noise.astype(np.uint8))
        
        # Add water surface line (horizontal edge)
        cv2.line(img, (0, water_start_y), (width, water_start_y), (255, 255, 255), 2)
        
        # Add measurement reference lines
        for i in range(5):
            y_pos = int(height * (0.2 + i * 0.15))
            cv2.line(img, (width-50, y_pos), (width-10, y_pos), (255, 255, 0), 2)
            cv2.putText(img, f"{52.0 - i*0.5:.1f}", (width-100, y_pos+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add timestamp
        cv2.putText(img, f"2024-09-16 {10+len(sample_data)-sample_data.index(data):02d}:30:00", 
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save image
        filepath = os.path.join("sample_flood_images", data["filename"])
        cv2.imwrite(filepath, img)
        print(f"Created: {filepath} (Water level: {data['water_level']:.1f}, Risk: {data['risk']})")
    
    return "sample_flood_images"

def demonstrate_single_image_analysis():
    """
    Demonstrate analysis of a single image.
    """
    print("\n" + "="*60)
    print("SINGLE IMAGE ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create sample images if they don't exist
    sample_folder = create_sample_flood_images()
    
    # Initialize analyzer
    analyzer = FloodImageAnalyzer(threshold=51.4)
    
    # Get the first sample image
    sample_image = os.path.join(sample_folder, "high_water_danger.jpg")
    
    if os.path.exists(sample_image):
        print(f"\nAnalyzing: {sample_image}")
        
        # Process the image
        result = analyzer.process_single_image(sample_image)
        
        if result:
            # Display original and annotated images
            original_img = cv2.imread(sample_image)
            annotated_img = analyzer.create_annotated_image(sample_image, result)
            
            if original_img is not None and annotated_img is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Original image
                ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
                ax1.set_title('Original Flood Monitoring Image')
                ax1.axis('off')
                
                # Annotated image
                ax2.imshow(annotated_img)
                ax2.set_title(f'Analysis Result: {result["Status"]}')
                ax2.axis('off')
                
                plt.tight_layout()
                plt.savefig('single_image_analysis_demo.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                # Print detailed results
                print(f"\nAnalysis Results:")
                print(f"  Image: {result['TimeStamp']}")
                print(f"  Water Level: {result['Distance']}")
                print(f"  Status: {result['Status']}")
                print(f"  Risk Level: {result['Risk_Level']}")
                print(f"  Threshold: {analyzer.threshold}")
    
    return analyzer

def demonstrate_batch_analysis():
    """
    Demonstrate batch analysis of multiple images.
    """
    print("\n" + "="*60)
    print("BATCH IMAGE ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create sample images
    sample_folder = create_sample_flood_images()
    
    # Initialize analyzer
    analyzer = FloodImageAnalyzer(threshold=51.4)
    
    # Process all images in the folder
    analyzer.process_image_folder(sample_folder)
    
    if analyzer.results:
        # Save results
        df = analyzer.save_results('demo_flood_analysis.csv')
        
        # Create comprehensive report
        analyzer.create_comprehensive_report(df)
        
        # Show results summary
        print(f"\nBatch Analysis Summary:")
        print(f"  Total images processed: {len(analyzer.results)}")
        print(f"  High risk images: {len([r for r in analyzer.results if r['Risk_Level'] == 'RED'])}")
        print(f"  Safe images: {len([r for r in analyzer.results if r['Risk_Level'] == 'GREEN'])}")
    
    return analyzer

def demonstrate_custom_threshold():
    """
    Demonstrate analysis with different threshold values.
    """
    print("\n" + "="*60)
    print("CUSTOM THRESHOLD ANALYSIS DEMONSTRATION")
    print("="*60)
    
    sample_folder = create_sample_flood_images()
    
    # Test different thresholds
    thresholds = [51.0, 51.2, 51.4, 51.6, 51.8]
    
    results_summary = []
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        analyzer = FloodImageAnalyzer(threshold=threshold)
        analyzer.process_image_folder(sample_folder)
        
        if analyzer.results:
            high_risk_count = len([r for r in analyzer.results if r['Risk_Level'] == 'RED'])
            safe_count = len([r for r in analyzer.results if r['Risk_Level'] == 'GREEN'])
            
            results_summary.append({
                'threshold': threshold,
                'high_risk': high_risk_count,
                'safe': safe_count,
                'total': len(analyzer.results)
            })
            
            print(f"  High risk: {high_risk_count}, Safe: {safe_count}")
    
    # Visualize threshold comparison
    if results_summary:
        thresholds = [r['threshold'] for r in results_summary]
        high_risk_counts = [r['high_risk'] for r in results_summary]
        safe_counts = [r['safe'] for r in results_summary]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        x = np.arange(len(thresholds))
        width = 0.35
        
        ax.bar(x - width/2, high_risk_counts, width, label='High Risk', color='red', alpha=0.7)
        ax.bar(x + width/2, safe_counts, width, label='Safe', color='green', alpha=0.7)
        
        ax.set_xlabel('Threshold Value')
        ax.set_ylabel('Number of Images')
        ax.set_title('Risk Classification by Threshold Value')
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main demonstration function.
    """
    print("="*80)
    print("REAL IMAGE FLOOD MONITORING SYSTEM - DEMONSTRATION")
    print("="*80)
    print("\nThis demonstration shows how to use the flood monitoring system with real images.")
    print("Sample images will be created to simulate flood monitoring scenarios.")
    
    # Check if OpenCV is available
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("Error: OpenCV not installed. Please install with: pip install opencv-python")
        return
    
    print("\nSelect demonstration:")
    print("1. Single image analysis")
    print("2. Batch image analysis")
    print("3. Custom threshold testing")
    print("4. Run all demonstrations")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        demonstrate_single_image_analysis()
    elif choice == "2":
        demonstrate_batch_analysis()
    elif choice == "3":
        demonstrate_custom_threshold()
    elif choice == "4":
        print("Running all demonstrations...")
        demonstrate_single_image_analysis()
        demonstrate_batch_analysis()
        demonstrate_custom_threshold()
    else:
        print("Invalid choice. Running single image demo...")
        demonstrate_single_image_analysis()
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED!")
    print("="*80)
    print("Files created:")
    print("• sample_flood_images/ - Sample flood monitoring images")
    print("• demo_flood_analysis.csv - Analysis results")
    print("• real_image_flood_analysis.png - Visualization")
    print("• single_image_analysis_demo.png - Single image demo")
    print("• threshold_comparison.png - Threshold comparison")
    print("\nTo use with your own images:")
    print("1. Run: python real_image_flood_analyzer.py")
    print("2. Follow the prompts to process your images")
    print("3. Use camera_calibration_tool.py to calibrate for your setup")

if __name__ == "__main__":
    main()