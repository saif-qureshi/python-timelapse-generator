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
import tempfile
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    photo_files = []
    for pattern in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        photo_files.extend(Path(photo_dir).glob(pattern))
    photo_files = sorted(photo_files)
    
    for photo_path in photo_files:
        filename = photo_path.stem  # Get filename without extension
        
        try:
            if len(filename) == 14 and filename.isdigit():
                photo_dt = datetime.strptime(filename, "%Y%m%d%H%M%S")
                
                if start_dt <= photo_dt <= end_dt:
                    if start_hour <= photo_dt.hour <= end_hour:
                        filtered_photos.append(str(photo_path))
        except ValueError:
            continue
    
    return filtered_photos

def create_temp_symlink_dir(filtered_photos, temp_dir, preserve_names=False):
    """
    Create a temporary directory with symlinks to filtered photos.
    This avoids copying files and maintains sequential naming for video or original names for export.
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    
    for idx, photo_path in enumerate(filtered_photos):
        if preserve_names:
            link_name = temp_path / Path(photo_path).name
        else:
            ext = Path(photo_path).suffix
            link_name = temp_path / f"{idx:06d}{ext}"
        
        if link_name.exists():
            link_name.unlink()
        link_name.symlink_to(photo_path)
    
    return str(temp_path)

def main():
    parser = argparse.ArgumentParser(description='Timelapse wrapper for Laravel integration')
    
    parser.add_argument('--mode', choices=['video', 'export'], default='video',
                       help='Operation mode: video generation or photo export (default: video)')
    
    parser.add_argument('camera_dir', help='Camera photo directory')
    parser.add_argument('output', help='Output file path (video or zip)')
    
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--start-hour', type=int, default=8, help='Start hour (0-23)')
    parser.add_argument('--end-hour', type=int, default=17, help='End hour (0-23)')
    
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--quality', default='HD')
    parser.add_argument('--duration', type=float, default=30, help='Video duration in seconds')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--gpu', action='store_true')
    
    parser.add_argument('--datetime', action='store_true')
    parser.add_argument('--logo', help='Logo path')
    parser.add_argument('--logo-opacity', type=float, default=1.0)
    parser.add_argument('--watermark', help='Watermark text')
    parser.add_argument('--watermark-opacity', type=float, default=0.3)
    parser.add_argument('--watermark-image', help='Watermark image path')
    parser.add_argument('--watermark-size', type=float, default=1.0, help='Watermark size multiplier')
    parser.add_argument('--watermark-transparency', type=float, default=0.3)
    
    parser.add_argument('--music', help='Music file path')
    parser.add_argument('--music-volume', type=float, default=1.0)
    parser.add_argument('--no-loop', action='store_true')
    
    args = parser.parse_args()
    
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
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="timelapse_")
        logger.info(f"Created temporary directory: {temp_dir}")
        
        preserve_names = args.mode == 'export'
        symlink_dir = create_temp_symlink_dir(filtered_photos, temp_dir, preserve_names)
        
        main_script = Path(__file__).parent / "main.py"
        cmd = [
            sys.executable,
            str(main_script),
            "--mode", args.mode,
            symlink_dir,
            args.output,
            "--pattern", "*.*",
            "--batch-size", str(args.batch_size),
        ]
        
        if args.mode == 'video':
            cmd.extend([
                "--fps", str(args.fps),
                "--resolution", args.quality,
                "--duration", str(args.duration),
            ])
        
        if args.workers:
            cmd.extend(["--workers", str(args.workers)])
        
        if args.mode == 'video':
            if args.gpu:
                cmd.append("--gpu")
            if args.datetime:
                cmd.append("--show-date")
            if args.logo:
                cmd.extend(["--logo", args.logo])
            if args.watermark:
                cmd.extend(["--text", args.watermark])
            if args.watermark_image:
                cmd.extend([
                    "--watermark", args.watermark_image,
                    "--watermark-size", str(int(300 * args.watermark_size)), str(int(300 * args.watermark_size)),
                    "--watermark-transparency", str(args.watermark_transparency)
                ])
            if args.music:
                cmd.extend(["--music", args.music])
        else:
            if args.datetime:
                cmd.append("--show-date")
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, timeout=7200)
        except subprocess.TimeoutExpired:
            logger.error("Command timed out after 2 hours")
            sys.exit(1)
        
        sys.exit(result.returncode)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")

if __name__ == '__main__':
    main()