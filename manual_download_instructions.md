
# Manual Download Instructions for Google Drive Images

## Overview:
Your Google Drive folder contains more than 50 files, which exceeds the automatic download limit.
Please manually download the following priority images for analysis.

## Target Directory: `targeted_images`

## Priority Images to Download (First 50):


### Date: 13-03-2024 (1 files)
- 13-03-2024_16-48_53.jpg

### Date: 13-09-2024 (1 files)
- 13-09-2024_17-60-16.jpg

### Date: 19-03-2024 (3 files)
- 19-03-2024_16-10-52.jpg
- 19-03-2024_16-26-52.jpg
- 19-03-2024_18-05-52.jpg

### Date: 19-05-2024 (8 files)
- 19-05-2024_14-28-43.jpg
- 19-05-2024_14-32-43.jpg
- 19-05-2024_16-45-53.jpg
- 19-05-2024_17-09-16.jpg
- 19-05-2024_17-48-15.jpg
- 19-05-2024_17-59-15.jpg
- 19-05-2024_19-21-54.jpg
- 19-05-2024_22-46-18.jpg

### Date: 19-09-2024 (35 files)
- 19-09-2024_10-41-56.jpg
- 19-09-2024_14-46-43.jpg
- 19-09-2024_14-49-43.jpg
- 19-09-2024_15-34-47.jpg
- 19-09-2024_15-49-47.jpg
- 19-09-2024_16-47-53.jpg
- 19-09-2024_16-55-16.jpg
- 19-09-2024_17-02-16.jpg
- 19-09-2024_17-08-16.jpg
- 19-09-2024_17-29-15.jpg
- 19-09-2024_17-40-15.jpg
- 19-09-2024_17-51-15.jpg
- 19-09-2024_18-56-56.jpg
- 19-09-2024_20-45-59.jpg
- 19-09-2024_20-52-59.jpg
- 19-09-2024_20-57-59.jpg
- 19-09-2024_21-07-38.jpg
- 19-09-2024_21-10-38.jpg
- 19-09-2024_21-11-38.jpg
- 19-09-2024_21-46-18.jpg
- 19-09-2024_21-49-18.jpg
- 19-09-2024_21-50-53.jpg
- 19-09-2024_21-53-53.jpg
- 19-09-2024_22-06-05.jpg
- 19-09-2024_22-14-53.jpg
- 19-09-2024_22-15-53.jpg
- 19-09-2024_22-27-38.jpg
- 19-09-2024_22-33-38.jpg
- 19-09-2024_22-49-18.jpg
- 19-09-2024_22-52-18.jpg
- 19-09-2024_22-54-18.jpg
- 19-09-2024_23-15-59.jpg
- 19-09-2024_23-22-59.jpg
- 19-09-2024_23-47-38.jpg
- 19-09-2024_23-50-38.jpg

### Date: frame (1 files)
- frame_45000.jpg

### Date: is (1 files)
- is_-09-2024_23-33-59.jpg


## Download Steps:

1. **Open the Google Drive folder:**
   https://drive.google.com/drive/folders/1vOCy_L8_yH-DQQ_O6QuaYfCUWxKSJ0ZN

2. **Navigate through the folder structure to find images**

3. **Download the images listed above to:** `/workspaces/flood_monitoring/targeted_images/`

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
19-05-2024_17-59-15.jpg, 19-09-2024_21-50-53.jpg, 19-05-2024_19-21-54.jpg, frame_45000.jpg, 19-09-2024_20-45-59.jpg

Then run: `python targeted_drive_analysis.py`
