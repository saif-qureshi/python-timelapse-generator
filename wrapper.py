#!/usr/bin/env python3
"""
Timelapse Wrapper for Laravel Integration
This script wraps the main timelapse generator to work with Laravel's timestamp-based photo structure.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import json

def filter_photos_by_timestamp(photo_dir, start_date, end_date, start_hour, end_hour):
    """
    Filter photos based on their timestamp filenames.
    
    Args:
        photo_dir: Directory containing photos with timestamp names (YYYYMMDDHHmmss.jpg)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        start_hour: Start hour (0-23)
        end_hour: End hour (0-23)
    
    Returns:
        List of filtered photo paths
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(hour=start_hour, minute=0, second=0)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=end_hour, minute=59, second=59)
    
    filtered_photos = []
    
    # Get all jpg and png files in the directory
    photo_files = []
    for pattern in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        photo_files.extend(Path(photo_dir).glob(pattern))
    photo_files = sorted(photo_files)
    
    for photo_path in photo_files:
        filename = photo_path.stem  # Get filename without extension
        
        try:
            # Parse timestamp from filename (YYYYMMDDHHmmss)
            if len(filename) == 14 and filename.isdigit():
                photo_dt = datetime.strptime(filename, "%Y%m%d%H%M%S")
                
                # Check if photo is within date range
                if start_dt <= photo_dt <= end_dt:
                    # Check if photo is within daily hour range
                    if start_hour <= photo_dt.hour <= end_hour:
                        filtered_photos.append(str(photo_path))
        except ValueError:
            # Skip files that don't match timestamp format
            continue
    
    return filtered_photos

def create_temp_symlink_dir(filtered_photos, temp_dir):
    """
    Create a temporary directory with symlinks to filtered photos.
    This avoids copying files and maintains sequential naming.
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks with sequential names
    for idx, photo_path in enumerate(filtered_photos):
        # Preserve the original extension
        ext = Path(photo_path).suffix
        link_name = temp_path / f"{idx:06d}{ext}"
        if link_name.exists():
            link_name.unlink()
        link_name.symlink_to(photo_path)
    
    return str(temp_path)

def main():
    parser = argparse.ArgumentParser(description='Timelapse wrapper for Laravel integration')
    
    # Required arguments
    parser.add_argument('camera_dir', help='Camera photo directory')
    parser.add_argument('output', help='Output video path')
    
    # Date/time filtering
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--start-hour', type=int, default=8, help='Start hour (0-23)')
    parser.add_argument('--end-hour', type=int, default=17, help='End hour (0-23)')
    
    # Pass through arguments to main.py
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--quality', default='HD')
    parser.add_argument('--duration', type=float, default=30, help='Video duration in seconds')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--gpu', action='store_true')
    
    # Overlays
    parser.add_argument('--datetime', action='store_true')
    parser.add_argument('--logo', help='Logo path')
    parser.add_argument('--logo-opacity', type=float, default=1.0)
    parser.add_argument('--watermark', help='Watermark text')
    parser.add_argument('--watermark-opacity', type=float, default=0.3)
    parser.add_argument('--watermark-image', help='Watermark image path')
    parser.add_argument('--watermark-size', type=float, default=1.0, help='Watermark size multiplier')
    parser.add_argument('--watermark-transparency', type=float, default=0.3)
    
    # Music
    parser.add_argument('--music', help='Music file path')
    parser.add_argument('--music-volume', type=float, default=1.0)
    parser.add_argument('--no-loop', action='store_true')
    
    args = parser.parse_args()
    
    # Filter photos by timestamp
    print(f"Filtering photos from {args.start_date} {args.start_hour}:00 to {args.end_date} {args.end_hour}:59")
    filtered_photos = filter_photos_by_timestamp(
        args.camera_dir,
        args.start_date,
        args.end_date,
        args.start_hour,
        args.end_hour
    )
    
    if not filtered_photos:
        print("Error: No photos found in the specified date/time range")
        sys.exit(1)
    
    print(f"Found {len(filtered_photos)} photos matching criteria")
    
    # Create temp directory with symlinks
    temp_dir = f"/tmp/timelapse_{os.getpid()}"
    try:
        symlink_dir = create_temp_symlink_dir(filtered_photos, temp_dir)
        
        # Build command for main.py
        main_script = Path(__file__).parent / "main.py"
        cmd = [
            sys.executable,
            str(main_script),
            symlink_dir,
            args.output,
            "--pattern", "*.*",  # Match all files since we're creating symlinks
            "--fps", str(args.fps),
            "--resolution", args.quality,  # Changed from --quality to --resolution
            "--duration", str(args.duration),
            "--batch-size", str(args.batch_size),
        ]
        
        # Add optional arguments
        if args.workers:
            cmd.extend(["--workers", str(args.workers)])
        if args.gpu:
            cmd.append("--gpu")
        if args.datetime:
            cmd.append("--show-date")
        if args.logo:
            cmd.extend(["--logo", args.logo])
        if args.watermark:
            # For custom text
            cmd.extend(["--text", args.watermark])
        if args.watermark_image:
            # For watermark image
            cmd.extend([
                "--watermark", args.watermark_image,
                "--watermark-size", str(int(300 * args.watermark_size)), str(int(300 * args.watermark_size)),
                "--watermark-transparency", str(args.watermark_transparency)
            ])
        if args.music:
            cmd.extend(["--music", args.music])
        
        # Execute main timelapse script
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        # Cleanup temp directory
        import shutil
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        
        sys.exit(result.returncode)
        
    except Exception as e:
        print(f"Error: {e}")
        # Cleanup on error
        import shutil
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        sys.exit(1)

if __name__ == '__main__':
    main()