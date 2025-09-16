#!/usr/bin/env python3
"""
Extract sample frames from the water level scale demo video
"""

import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, max_frames=3):
    """Extract sample frames from video"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return False
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 Video Info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {duration:.1f} seconds")
    
    # Extract frames at intervals
    frame_interval = max(1, total_frames // max_frames)
    extracted = 0
    frame_count = 0
    
    print(f"\n🖼️ Extracting {max_frames} sample frames...")
    
    while cap.isOpened() and extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = output_path / f"demo_frame_{extracted+1:02d}.png"
            cv2.imwrite(str(frame_filename), frame)
            print(f"  ✅ Saved: {frame_filename.name}")
            extracted += 1
        
        frame_count += 1
    
    cap.release()
    print(f"\n🎉 Extracted {extracted} frames to: {output_path}")
    return True

def main():
    # Find the latest video file
    demo_dir = Path("simple_scale_demo")
    video_files = list(demo_dir.glob("water_scale_demo_*.mp4"))
    
    if not video_files:
        print("❌ No video files found in simple_scale_demo/")
        return
    
    # Use the latest video
    latest_video = sorted(video_files)[-1]
    print(f"🎬 Processing video: {latest_video.name}")
    
    # Extract frames
    frames_dir = demo_dir / "sample_frames"
    extract_frames(latest_video, frames_dir, max_frames=5)

if __name__ == "__main__":
    main()