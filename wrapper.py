#!/usr/bin/env python3
"""
Timelapse Wrapper for Laravel Integration
This script wraps the timelapse generator to work with S3-stored photos.
Called by Laravel's ProcessMediaRequest job with S3 credentials passed as env vars.
"""

import os
import sys
import argparse
import tempfile
import shutil
import logging

from s3_client import S3Client, filter_photos_by_time
from timelapse_generator import TimelapseGenerator
from photo_exporter import PhotoExporter

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Timelapse wrapper for Laravel integration (S3)')

    parser.add_argument('--mode', choices=['video', 'export'], default='video',
                       help='Operation mode: video generation or photo export (default: video)')

    parser.add_argument('output_path', help='Local temp output file path')
    parser.add_argument('--s3-prefix', required=True, help='S3 prefix for camera photos (e.g. photos/camera_1)')
    parser.add_argument('--s3-output-key', required=True, help='S3 key to upload the output to')

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
    parser.add_argument('--logo', help='S3 key for logo image')
    parser.add_argument('--logo-opacity', type=float, default=1.0)
    parser.add_argument('--watermark', help='Watermark text')
    parser.add_argument('--watermark-opacity', type=float, default=0.3)
    parser.add_argument('--watermark-image', help='S3 key for watermark image')
    parser.add_argument('--watermark-size', type=float, default=1.0, help='Watermark size multiplier')
    parser.add_argument('--watermark-transparency', type=float, default=0.3)

    parser.add_argument('--music', help='S3 key for music file')
    parser.add_argument('--music-volume', type=float, default=1.0)
    parser.add_argument('--no-loop', action='store_true')

    args = parser.parse_args()

    # Create S3 client from environment variables
    try:
        s3 = S3Client.from_env()
    except ValueError as e:
        logger.error(f"S3 configuration error: {e}")
        sys.exit(1)

    # List and filter photos from S3
    logger.info(f"Listing photos from s3://{s3.bucket}/{args.s3_prefix}")
    all_photos = s3.list_photos(args.s3_prefix)

    if not all_photos:
        logger.error(f"No photos found under prefix: {args.s3_prefix}")
        sys.exit(1)

    logger.info(f"Filtering photos from {args.start_date} {args.start_hour}:00 to {args.end_date} {args.end_hour}:59")
    filtered_photos = filter_photos_by_time(
        all_photos,
        args.start_date,
        args.end_date,
        args.start_hour,
        args.end_hour
    )

    if not filtered_photos:
        logger.error("No photos found in the specified date/time range")
        sys.exit(1)

    logger.info(f"Found {len(filtered_photos)} photos matching criteria")

    try:
        if args.mode == 'video':
            _process_video(args, s3, filtered_photos)
        else:
            _process_export(args, s3, filtered_photos)

        # Upload output to S3
        content_type = 'video/mp4' if args.mode == 'video' else 'application/zip'
        s3.upload_file(args.output_path, args.s3_output_key, content_type)
        logger.info(f"Output uploaded to s3://{s3.bucket}/{args.s3_output_key}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
    finally:
        # Clean up local output file after upload
        try:
            if os.path.exists(args.output_path):
                os.remove(args.output_path)
                logger.info(f"Cleaned up local output: {args.output_path}")
        except OSError as e:
            logger.warning(f"Failed to cleanup output file: {e}")


def _process_video(args, s3, filtered_photos):
    """Generate timelapse video."""
    generator = TimelapseGenerator(
        output_path=args.output_path,
        fps=args.fps,
        num_workers=args.workers,
        s3_client=s3,
        photo_list=filtered_photos,
    )

    generator.load_images()

    watermark_size = None
    if args.watermark_image:
        size_val = int(300 * args.watermark_size)
        watermark_size = (size_val, size_val)

    generator.create_video_ffmpeg(
        quality=args.quality,
        duration=args.duration,
        show_date=args.datetime,
        text=args.watermark,
        logo_path=args.logo,
        watermark_path=args.watermark_image,
        watermark_size=watermark_size,
        watermark_transparency=args.watermark_transparency,
        batch_size=args.batch_size,
        use_gpu=args.gpu,
    )

    if args.music:
        generator.add_music(
            args.music,
            loop=not args.no_loop,
            volume=args.music_volume,
        )
    else:
        generator.finalize()


def _process_export(args, s3, filtered_photos):
    """Generate photo export ZIP."""
    exporter = PhotoExporter(
        output_path=args.output_path,
        num_workers=args.workers,
        s3_client=s3,
        photo_list=filtered_photos,
    )

    exporter.load_images()
    exporter.export_photos(
        show_date=args.datetime,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
