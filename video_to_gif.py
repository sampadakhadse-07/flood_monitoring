#!/usr/bin/env python3
"""
Convert video to animated GIF for easier viewing
"""

import cv2
from PIL import Image
import numpy as np
from pathlib import Path

def video_to_gif(video_path, gif_path, resize_factor=0.5, frame_skip=1):
    """Convert MP4 video to animated GIF"""
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return False
    
    frames = []
    frame_count = 0
    
    print("🎬 Converting video to GIF...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            if resize_factor != 1.0:
                height, width = frame_rgb.shape[:2]
                new_width = int(width * resize_factor)
                new_height = int(height * resize_factor)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
        
        frame_count += 1
    
    cap.release()
    
    if frames:
        # Save as animated GIF
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=2000,  # 2 seconds per frame
            loop=0
        )
        print(f"✅ GIF created: {gif_path}")
        print(f"📊 Frames: {len(frames)}, Duration: 2s per frame")
        return True
    else:
        print("❌ No frames extracted")
        return False

def main():
    demo_dir = Path("simple_scale_demo")
    video_files = list(demo_dir.glob("water_scale_demo_*.mp4"))
    
    if not video_files:
        print("❌ No video files found")
        return
    
    latest_video = sorted(video_files)[-1]
    gif_path = demo_dir / f"{latest_video.stem}.gif"
    
    print(f"🎬 Converting: {latest_video.name}")
    video_to_gif(latest_video, gif_path, resize_factor=0.7, frame_skip=2)

if __name__ == "__main__":
    main()