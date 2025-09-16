#!/usr/bin/env python3
"""
Camera Calibration Tool for Flood Monitoring
This script helps calibrate the image analysis system for your specific camera setup.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

class FloodCameraCalibrator:
    def __init__(self):
        self.reference_points = []
        self.calibration_data = {}
    
    def interactive_calibration(self, image_path):
        """
        Interactive calibration tool to set reference points on an image.
        """
        print("Interactive Camera Calibration")
        print("=" * 40)
        print("Instructions:")
        print("1. Click on reference points in the image")
        print("2. Press 'r' to reset points")
        print("3. Press 's' to save calibration")
        print("4. Press 'q' to quit")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        # Create a copy for drawing
        img_display = img.copy()
        
        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Ask user for the distance value at this point
                distance = input(f"Enter distance value for point ({x}, {y}): ")
                try:
                    distance = float(distance)
                    self.reference_points.append({'x': x, 'y': y, 'distance': distance})
                    # Draw point on image
                    cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(img_display, f'{distance}', (x+10, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    print(f"Added reference point: ({x}, {y}) = {distance}")
                except ValueError:
                    print("Invalid distance value!")
        
        cv2.namedWindow('Calibration Image')
        cv2.setMouseCallback('Calibration Image', mouse_callback)
        
        while True:
            cv2.imshow('Calibration Image', img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset points
                self.reference_points = []
                img_display = img.copy()
                print("Reference points reset")
            elif key == ord('s'):
                # Save calibration
                if len(self.reference_points) >= 2:
                    self.save_calibration()
                    print("Calibration saved!")
                else:
                    print("Need at least 2 reference points!")
        
        cv2.destroyAllWindows()
        return self.reference_points
    
    def calculate_calibration_function(self):
        """
        Calculate calibration function from reference points.
        """
        if len(self.reference_points) < 2:
            print("Need at least 2 reference points for calibration!")
            return None
        
        # Extract x, y, and distance values
        points = np.array([[p['x'], p['y'], p['distance']] for p in self.reference_points])
        
        # For simple linear calibration, use y-coordinate vs distance
        y_coords = points[:, 1]
        distances = points[:, 2]
        
        # Fit linear relationship
        coeffs = np.polyfit(y_coords, distances, 1)
        
        self.calibration_data = {
            'coefficients': coeffs.tolist(),
            'reference_points': self.reference_points,
            'calibration_type': 'linear_y_to_distance'
        }
        
        return coeffs
    
    def pixel_to_distance(self, y_pixel):
        """
        Convert pixel y-coordinate to distance using calibration.
        """
        if 'coefficients' not in self.calibration_data:
            print("No calibration data available!")
            return None
        
        coeffs = self.calibration_data['coefficients']
        distance = coeffs[0] * y_pixel + coeffs[1]
        return distance
    
    def save_calibration(self, filename='camera_calibration.json'):
        """
        Save calibration data to file.
        """
        if not self.reference_points:
            print("No calibration data to save!")
            return
        
        # Calculate calibration function
        self.calculate_calibration_function()
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename='camera_calibration.json'):
        """
        Load calibration data from file.
        """
        try:
            with open(filename, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"Calibration loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Calibration file {filename} not found!")
            return False
    
    def test_calibration(self, image_path):
        """
        Test the calibration on an image.
        """
        if not self.calibration_data:
            print("No calibration data available!")
            return
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image {image_path}")
            return
        
        height, width = img.shape[:2]
        
        # Create test points along the y-axis
        test_y_points = np.linspace(0, height-1, 10)
        
        print("Calibration Test Results:")
        print("Y-Pixel → Distance")
        print("-" * 20)
        
        for y in test_y_points:
            distance = self.pixel_to_distance(y)
            print(f"{int(y):4d} → {distance:.2f}")
    
    def visualize_calibration(self, image_path):
        """
        Visualize the calibration on an image.
        """
        if not self.calibration_data:
            print("No calibration data available!")
            return
        
        img = cv2.imread(image_path)
        if img is None:
            return
        
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Plot the image and calibration
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image with reference points
        ax1.imshow(img_rgb)
        ax1.set_title('Calibration Reference Points')
        
        for point in self.reference_points:
            ax1.plot(point['x'], point['y'], 'ro', markersize=8)
            ax1.text(point['x']+10, point['y'], f"{point['distance']}", 
                    color='white', fontsize=10, weight='bold')
        
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        
        # Show calibration curve
        y_range = np.linspace(0, height-1, 100)
        distances = [self.pixel_to_distance(y) for y in y_range]
        
        ax2.plot(distances, y_range, 'b-', linewidth=2, label='Calibration Curve')
        
        # Plot reference points
        ref_y = [p['y'] for p in self.reference_points]
        ref_dist = [p['distance'] for p in self.reference_points]
        ax2.plot(ref_dist, ref_y, 'ro', markersize=8, label='Reference Points')
        
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_title('Pixel to Distance Calibration')
        ax2.legend()
        ax2.grid(True)
        ax2.invert_yaxis()  # Invert y-axis to match image coordinates
        
        plt.tight_layout()
        plt.savefig('calibration_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main calibration workflow.
    """
    print("="*60)
    print("FLOOD MONITORING CAMERA CALIBRATION TOOL")
    print("="*60)
    
    calibrator = FloodCameraCalibrator()
    
    print("\nSelect option:")
    print("1. Create new calibration")
    print("2. Load existing calibration")
    print("3. Test existing calibration")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Create new calibration
        image_path = input("Enter path to calibration image: ").strip()
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        print("\nStarting interactive calibration...")
        print("Click on known reference points in the image.")
        
        calibrator.interactive_calibration(image_path)
        
        if calibrator.reference_points:
            calibrator.visualize_calibration(image_path)
            print(f"\nCalibration completed with {len(calibrator.reference_points)} reference points")
    
    elif choice == "2":
        # Load existing calibration
        filename = input("Enter calibration file name (default: camera_calibration.json): ").strip()
        if not filename:
            filename = "camera_calibration.json"
        
        if calibrator.load_calibration(filename):
            image_path = input("Enter image path to visualize calibration: ").strip()
            if os.path.exists(image_path):
                calibrator.visualize_calibration(image_path)
    
    elif choice == "3":
        # Test calibration
        filename = input("Enter calibration file name (default: camera_calibration.json): ").strip()
        if not filename:
            filename = "camera_calibration.json"
        
        if calibrator.load_calibration(filename):
            image_path = input("Enter test image path: ").strip()
            if os.path.exists(image_path):
                calibrator.test_calibration(image_path)
    
    print("\nCalibration process completed!")

if __name__ == "__main__":
    import os
    main()