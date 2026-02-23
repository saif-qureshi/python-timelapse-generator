#!/usr/bin/env python3
"""
Photo Exporter Module
Export photos with optional date/time overlays and effects.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import zipfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import shutil
from contextlib import contextmanager

from image_editor import ImageEditor
from utils import (
    Config, validate_path, validate_numeric_range,
    check_disk_space, ValidationError, SecurityError
)

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class PhotoExporter:
    """Export photos with optional overlays and effects."""

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()

    def cleanup(self):
        """Clean up any temporary resources."""
        if hasattr(self, '_temp_dir') and self._temp_dir:
            try:
                shutil.rmtree(self._temp_dir)
                logger.info(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")

    def __init__(self, output_path: str, num_workers: int = None,
                 s3_client=None, photo_list: list = None):
        """
        Initialize the photo exporter.

        Args:
            output_path: Output zip file path (local temp path)
            num_workers: Number of worker threads (default: CPU count)
            s3_client: S3Client instance for reading photos
            photo_list: List of photo dicts from s3_client.list_photos()
        """
        try:
            output_path_obj = Path(output_path)
            if not output_path_obj.suffix.lower() == '.zip':
                raise ValidationError("Output file must have .zip extension")
            if output_path_obj.parent.exists():
                self.output_path = output_path_obj
            else:
                raise ValidationError(f"Output directory does not exist: {output_path_obj.parent}")

        except (ValidationError, SecurityError) as e:
            logger.error(f"Initialization failed: {e}")
            raise

        self.s3_client = s3_client
        self.photo_list = photo_list or []

        if num_workers:
            self.num_workers = validate_numeric_range(num_workers, 1, Config.MAX_WORKERS, "Workers")
        else:
            self.num_workers = min(os.cpu_count() or 1, Config.MAX_WORKERS)

        self.images = []
        self.image_editor = ImageEditor()
        self._temp_dir = None

    def load_images(self, pattern: str = '*.jpg') -> int:
        """
        Load images from the pre-supplied photo list.
        """
        if not self.photo_list:
            raise ValueError("No images in photo list")

        self.images = self.photo_list
        logger.info(f"Loaded {len(self.images)} images from S3")
        return len(self.images)

    def export_photos(self, show_date: bool = False, batch_size: int = 50):
        """
        Export photos to a zip file with optional date overlays.
        Downloads each image from S3, processes it, writes to temp dir, then zips.
        """
        if not self.images:
            logger.warning("No images to export")
            return

        batch_size = validate_numeric_range(batch_size, Config.MIN_BATCH_SIZE,
                                          Config.MAX_BATCH_SIZE, "Batch size")

        if not check_disk_space(self.output_path.parent, Config.MIN_AVAILABLE_DISK_GB):
            raise RuntimeError(f"Insufficient disk space. Need at least {Config.MIN_AVAILABLE_DISK_GB}GB")

        with tempfile.TemporaryDirectory(prefix="photo_export_") as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Using temporary directory: {temp_path}")

            total_batches = (len(self.images) + batch_size - 1) // batch_size

            logger.info(f"Processing {len(self.images)} images...")

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(self.images))
                    batch_photos = self.images[start_idx:end_idx]

                    future = executor.submit(
                        self._process_batch,
                        batch_photos,
                        temp_path,
                        show_date,
                        batch_idx
                    )
                    futures.append(future)

                completed = 0
                failed_batches = 0
                for future in as_completed(futures):
                    try:
                        future.result()
                        completed += 1
                        logger.info(f"Completed batch {completed}/{total_batches}")
                    except Exception as e:
                        failed_batches += 1
                        logger.error(f"Batch processing failed ({failed_batches} failures so far): {e}")

                if failed_batches == total_batches:
                    raise RuntimeError(f"Export failed: all {total_batches} batches failed")
                if failed_batches > 0:
                    logger.warning(f"Export completed with {failed_batches}/{total_batches} failed batches")

            logger.info(f"Creating zip file: {self.output_path}")
            self._create_zip(temp_path)

        logger.info(f"Export complete: {self.output_path}")

    def _process_batch(self, photo_infos: List[dict], output_dir: Path,
                      show_date: bool, batch_idx: int) -> None:
        """
        Process a batch of images by streaming from S3.
        """
        for photo_info in photo_infos:
            try:
                # Download and decode image from S3 in memory
                img_data = self.s3_client.download_bytes(photo_info['key'])
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                del img_data, img_array

                if img is None:
                    logger.warning(f"Could not decode {photo_info['key']}")
                    continue

                if img.size > Config.MAX_IMAGE_SIZE:
                    logger.warning(f"Image too large, skipping: {photo_info['key']}")
                    continue

                if show_date:
                    img = self.image_editor.add_datetime(img, photo_info=photo_info)

                filename = photo_info['filename']
                output_path = output_dir / filename

                success = cv2.imwrite(str(output_path), img)
                if not success:
                    raise IOError(f"Failed to save image: {output_path}")

                del img

            except cv2.error as e:
                logger.error(f"OpenCV error processing {photo_info.get('key', '?')}: {e}")
            except IOError as e:
                logger.error(f"IO error processing {photo_info.get('key', '?')}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing {photo_info.get('key', '?')}: {e}")

    def _create_zip(self, source_dir: Path) -> None:
        """Create a zip file from processed images."""
        with zipfile.ZipFile(self.output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_file in sorted(source_dir.glob('*')):
                if img_file.is_file():
                    zipf.write(img_file, img_file.name)

        try:
            zip_size_mb = self.output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Zip file size: {zip_size_mb:.2f} MB")
        except OSError as e:
            logger.warning(f"Could not get zip file size: {e}")
