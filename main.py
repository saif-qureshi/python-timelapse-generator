#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from photo_exporter import PhotoExporter
from timelapse_generator import TimelapseGenerator
from utils import ValidationError, SecurityError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Timelapse video generator and photo exporter')
    
    parser.add_argument('--mode', choices=['video', 'export'], default='video',
                       help='Operation mode: video generation or photo export (default: video)')
    
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output', help='Output file path (video or zip)')
    parser.add_argument('--pattern', default='*.jpg', help='Image file pattern (default: *.jpg)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--workers', type=int, help='Number of worker threads')
    parser.add_argument('--batch-size', type=int, default=100, 
                       help='Images to process per batch')
    parser.add_argument('--gpu', action='store_true', 
                       help='Use GPU acceleration (requires NVENC)')
    
    parser.add_argument('--resolution', choices=['720', 'HD', '4K'], 
                       default='720', help='Video resolution (default: 720)')
    parser.add_argument('--duration', type=float, default=30,
                       choices=[30, 60, 90, 120, 150, 180],
                       help='Video duration in seconds (30-180, default: 30)')
    
    parser.add_argument('--show-date', action='store_true', 
                       help='Show date on images')
    
    parser.add_argument('--text', help='Custom text to add')
    
    parser.add_argument('--logo', help='Path to logo image (auto-placed top-right at 50x50)')
    
    parser.add_argument('--watermark', help='Path to watermark image')
    parser.add_argument('--watermark-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Watermark size')
    parser.add_argument('--watermark-transparency', type=float, default=0.3,
                       help='Watermark transparency (0-1, default: 0.3)')
    
    parser.add_argument('--music', help='Path to music file')
    
    parser.add_argument('--brightness', type=float, default=0.5,
                       help='Brightness (0-1, default: 0.5)')
    parser.add_argument('--contrast', type=float, default=1.0,
                       help='Contrast (0-3, default: 1.0)')
    parser.add_argument('--saturation', type=float, default=1.0,
                       help='Saturation (0-3, default: 1.0)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'export':
            exporter = PhotoExporter(args.input_dir, args.output, args.workers)
            exporter.load_images(args.pattern)
            exporter.export_photos(
                show_date=args.show_date,
                batch_size=args.batch_size
            )
        else:
            generator = TimelapseGenerator(
                args.input_dir, args.output, args.fps, args.workers
            )
            
            generator.load_images(args.pattern)
            
            generator.create_video_ffmpeg(
                quality=args.resolution,
                duration=args.duration,
                show_date=args.show_date,
                text=args.text,
                logo_path=args.logo,
                watermark_path=args.watermark,
                watermark_size=tuple(args.watermark_size) if args.watermark_size else None,
                watermark_transparency=args.watermark_transparency,
                brightness=args.brightness,
                contrast=args.contrast,
                saturation=args.saturation,
                batch_size=args.batch_size,
                use_gpu=args.gpu
            )
            
            if args.music:
                generator.add_music(
                    args.music, 
                    loop=True,
                    volume=1.0
                )
            else:
                if generator.temp_video_path.exists():
                    os.rename(generator.temp_video_path, generator.output_path)
                    logger.info(f"Final video saved: {generator.output_path}")
                
    except (ValidationError, SecurityError) as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()