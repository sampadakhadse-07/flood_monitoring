# Real Image Flood Monitoring System - User Guide

## 🌊 Overview

This system provides comprehensive flood monitoring analysis using real images with threshold-based risk assessment. The system can:

- Process real flood monitoring images
- Detect water levels using computer vision
- Apply threshold-based risk classification (Red/Green status)
- Generate comprehensive analysis reports and visualizations
- Calibrate for your specific camera setup

## 📁 System Components

### Core Analysis Files:
- **`real_image_flood_analyzer.py`** - Main image processing and analysis system
- **`camera_calibration_tool.py`** - Interactive calibration tool for your camera setup
- **`simple_real_image_demo.py`** - Complete demonstration with sample images

### Data Analysis Files:
- **`flood_monitoring_analysis.py`** - CSV data analysis (your original data)
- **`flood_data_labeled.csv`** - Your original data with risk labels

### Generated Results:
- **`sample_flood_images/`** - Sample flood monitoring images (6 scenarios)
- **`real_image_analysis_results.png`** - Comprehensive analysis visualization
- **`sample_images_analysis.png`** - Sample images with analysis results
- **`real_image_flood_analysis_results.csv`** - Detailed analysis results

## 🚀 Quick Start Guide

### Option 1: Use Your Own Images

1. **Prepare your images:**
   ```bash
   # Organize your flood monitoring images in a folder
   mkdir my_flood_images
   # Copy your JPG images to this folder
   ```

2. **Run the main analyzer:**
   ```bash
   python3 real_image_flood_analyzer.py
   ```
   
3. **Follow the prompts:**
   - Choose "1" to process images from a folder
   - Enter your images folder path
   - The system will analyze all images and generate reports

### Option 2: Try the Demo First

1. **Run the demo to see how it works:**
   ```bash
   python3 simple_real_image_demo.py
   ```
   
   This will:
   - Create 6 sample flood monitoring images
   - Analyze them with different water levels
   - Generate comprehensive visualizations
   - Show detection accuracy and risk assessment

### Option 3: Calibrate for Your Camera

1. **Run the calibration tool:**
   ```bash
   python3 camera_calibration_tool.py
   ```

2. **Interactive calibration:**
   - Choose "1" to create new calibration
   - Select a reference image from your setup
   - Click on known reference points
   - Enter the actual distance values
   - Save the calibration for future use

## 🔧 System Configuration

### Threshold Settings

The default threshold is **51.4**. You can modify this in any script:

```python
# In real_image_flood_analyzer.py, line 199:
threshold = 51.4  # Adjust based on your requirements

# Or when creating the analyzer:
analyzer = FloodImageAnalyzer(threshold=52.0)  # Custom threshold
```

### Risk Classification

- 🟢 **GREEN (Safe)**: Water level ≤ threshold
- 🔴 **RED (High Risk)**: Water level > threshold
- 🟠 **Orange Line**: Threshold boundary

## 📊 Understanding the Results

### Analysis Outputs

1. **CSV Results** (`real_image_flood_analysis_results.csv`):
   ```csv
   TimeStamp,Actual_Distance,Detected_Distance,Detection_Error,Status,Risk_Level,Color,Image_Path
   16-09-2024_08-30-15.jpg,50.9,50.5,0.4,Safe (Green),GREEN,green,sample_flood_images/16-09-2024_08-30-15.jpg
   ```

2. **Comprehensive Visualization** (`real_image_analysis_results.png`):
   - Detection accuracy plot
   - Risk status distribution
   - Detection error analysis
   - Water level timeline

3. **Sample Images Grid** (`sample_images_analysis.png`):
   - Original images with analysis overlay
   - Actual vs detected values
   - Risk status for each image

### Key Metrics

- **Total Images Processed**: Number of images analyzed
- **High Risk (Red)**: Images with water level > threshold
- **Safe (Green)**: Images with water level ≤ threshold
- **Average Detection Error**: Accuracy of the detection algorithm
- **Detection Accuracy**: How close detected values are to actual values

## 🎯 Customization for Your Setup

### 1. Calibration (Recommended)

For best results with your specific camera setup:

1. Take a reference image with known water level markers
2. Run `camera_calibration_tool.py`
3. Mark reference points with known distances
4. Save the calibration file
5. The system will use your calibration for future analysis

### 2. Modify Detection Algorithm

In `real_image_flood_analyzer.py`, you can customize the `detect_water_level()` method:

```python
def detect_water_level(self, image_path):
    # Customize based on your image characteristics:
    # - Water color (adjust HSV ranges)
    # - Edge detection parameters
    # - Scale calibration
    # - Noise filtering
```

### 3. Adjust Image Processing Parameters

```python
# Edge detection sensitivity
edges = cv2.Canny(gray, 50, 150)  # Adjust 50, 150 for your images

# Water color detection
lower_water = np.array([100, 50, 50])   # Adjust for your water color
upper_water = np.array([130, 255, 255]) # Adjust for your water color

# Horizontal line detection
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))  # Adjust size
```

## 📈 Working with Real Data

### Your Original CSV Analysis

Your flood data has been analyzed and labeled:

- **Original data**: `Flood_Data.csv` (1,778 records)
- **Labeled data**: `flood_data_labeled.csv` (with risk status)
- **Visualization**: `flood_monitoring_analysis.png`

**Results Summary:**
- Threshold: 51.4
- High Risk: 796 records (44.8%)
- Safe: 982 records (55.2%)
- Distance range: 51.0 to 52.2

### Integrating Image and CSV Data

You can combine image analysis with your CSV data by matching timestamps:

```python
# Load your CSV data
csv_data = pd.read_csv('Flood_Data.csv')

# Process corresponding images
for index, row in csv_data.iterrows():
    image_filename = row['TimeStamp']
    image_path = f"your_images_folder/{image_filename}"
    
    if os.path.exists(image_path):
        detected_level = analyzer.detect_water_level(image_path)
        # Compare with CSV distance value
        csv_distance = row['Distance']
```

## 🔍 Troubleshooting

### Common Issues

1. **Images not detected properly:**
   - Ensure images are in supported formats (.jpg, .png, .bmp)
   - Check image quality and lighting
   - Calibrate for your specific setup

2. **Poor detection accuracy:**
   - Run calibration tool with reference images
   - Adjust edge detection parameters
   - Modify water color detection ranges

3. **OpenCV installation issues:**
   ```bash
   pip install opencv-python-headless  # For headless environments
   # or
   pip install opencv-python  # For GUI environments
   ```

### Performance Tips

1. **Batch Processing**: Use folder processing for multiple images
2. **Calibration**: Always calibrate for best accuracy
3. **Image Quality**: Use high-resolution, well-lit images
4. **Consistent Setup**: Keep camera position and angle consistent

## 📝 Example Workflow

### Complete Analysis Workflow:

1. **Setup and Demo:**
   ```bash
   python3 simple_real_image_demo.py
   ```

2. **Calibrate for your camera:**
   ```bash
   python3 camera_calibration_tool.py
   ```

3. **Analyze your images:**
   ```bash
   python3 real_image_flood_analyzer.py
   ```

4. **Review results:**
   - Check CSV files for detailed data
   - Review PNG visualizations
   - Analyze detection accuracy

5. **Integrate with existing data:**
   - Compare with your CSV data
   - Validate detection accuracy
   - Adjust thresholds if needed

## 📊 Expected Outputs

After running the complete system, you'll have:

- ✅ **Risk-classified images** with color coding
- ✅ **Comprehensive analysis reports** with statistics
- ✅ **Visual dashboards** showing trends and distributions  
- ✅ **CSV data files** for further analysis
- ✅ **Calibrated detection system** for your specific setup
- ✅ **Automated monitoring capability** for new images

## 🎯 Next Steps

1. **Test with your real images**
2. **Calibrate the system for your camera setup**
3. **Adjust threshold based on your risk criteria**
4. **Integrate with your existing monitoring workflow**
5. **Set up automated processing for new images**

The system is now ready to process your real flood monitoring images with accurate water level detection and risk assessment!