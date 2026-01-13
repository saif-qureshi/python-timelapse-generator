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
import glob
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
    
    def __init__(self, input_dir: str, output_path: str, num_workers: int = None):
        """
        Initialize the photo exporter.
        
        Args:
            input_dir: Directory containing input images
            output_path: Output zip file path
            num_workers: Number of worker threads (default: CPU count)
        """
        try:
            self.input_dir = validate_path(input_dir, must_exist=True)
            if not self.input_dir.is_dir():
                raise ValidationError(f"Input must be a directory: {input_dir}")
                
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
            
        if num_workers:
            self.num_workers = validate_numeric_range(num_workers, 1, Config.MAX_WORKERS, "Workers")
        else:
            self.num_workers = min(os.cpu_count() or 1, Config.MAX_WORKERS)
            
        self.images = []
        self.image_editor = ImageEditor()
        self._temp_dir = None
        
    def load_images(self, pattern: str = '*.jpg') -> int:
        """
        Load image paths from the input directory.
        
        Args:
            pattern: Glob pattern for image files
            
        Returns:
            Number of images found
        """
        if '..' in pattern or '/' in pattern or '\\' in pattern:
            raise SecurityError("Invalid pattern: path separators not allowed")
            
        self.images = sorted(glob.glob(str(self.input_dir / pattern)))
        logger.info(f"Found {len(self.images)} images")
        return len(self.images)
    
    def export_photos(self, show_date: bool = False, batch_size: int = 50):
        """
        Export photos to a zip file with optional date overlays.
        
        Args:
            show_date: Whether to add date/time overlay
            batch_size: Number of images to process per batch
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
                    batch_paths = self.images[start_idx:end_idx]
                    
                    future = executor.submit(
                        self._process_batch,
                        batch_paths,
                        temp_path,
                        show_date,
                        batch_idx
                    )
                    futures.append(future)
                
                completed = 0
                for future in as_completed(futures):
                    try:
                        future.result()
                        completed += 1
                        logger.info(f"Completed batch {completed}/{total_batches}")
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        raise
            
            logger.info(f"Creating zip file: {self.output_path}")
            self._create_zip(temp_path)
            
        logger.info(f"Export complete: {self.output_path}")
    
    def _process_batch(self, image_paths: List[str], output_dir: Path, 
                      show_date: bool, batch_idx: int) -> None:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save processed images
            show_date: Whether to add date overlay
            batch_idx: Batch index for progress tracking
        """
        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not load {img_path}")
                    continue
                
                if img.size > Config.MAX_IMAGE_SIZE:
                    logger.warning(f"Image too large, skipping: {img_path}")
                    continue
                
                if show_date:
                    img = self.image_editor.add_datetime(img, img_path)
                
                filename = Path(img_path).name
                output_path = output_dir / filename
                
                success = cv2.imwrite(str(output_path), img)
                if not success:
                    raise IOError(f"Failed to save image: {output_path}")
                
            except cv2.error as e:
                logger.error(f"OpenCV error processing {img_path}: {e}")
            except IOError as e:
                logger.error(f"IO error processing {img_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing {img_path}: {e}")
    
    
    def _create_zip(self, source_dir: Path) -> None:
        """
        Create a zip file from processed images.
        
        Args:
            source_dir: Directory containing processed images
        """
        with zipfile.ZipFile(self.output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_file in sorted(source_dir.glob('*')):
                if img_file.is_file():
                    zipf.write(img_file, img_file.name)
                    
        try:
            zip_size_mb = self.output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Zip file size: {zip_size_mb:.2f} MB")
        except OSError as e:
            logger.warning(f"Could not get zip file size: {e}")


