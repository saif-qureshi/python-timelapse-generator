#!/usr/bin/env python3
"""
TimelapseGenerator Module
Create timelapse videos with customizable effects and overlays.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List
import glob
from moviepy import VideoFileClip, AudioFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import subprocess
import shlex
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import shutil

from image_editor import ImageEditor
from utils import (
    Config, validate_path, validate_numeric_range,
    check_disk_space, check_memory_available,
    ValidationError, SecurityError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video generation."""
    width: int
    height: int
    fps: int
    quality: str
    bitrate: str
    codec: str = 'libx264'
    preset: str = 'faster'
    crf: int = 23


class TimelapseGenerator:
    """Timelapse video generator with effects and overlay support."""
    
    QUALITY_PRESETS = {
        '720': (1280, 720, '5M', 25),
        'HD': (1920, 1080, '10M', 23),
        '4K': (3840, 2160, '20M', 20)
    }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, '_ffmpeg_process') and self._ffmpeg_process:
            try:
                self._ffmpeg_process.terminate()
                self._ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._ffmpeg_process.kill()
            except Exception:
                pass
                
        if hasattr(self, 'temp_video_path') and self.temp_video_path.exists():
            try:
                os.remove(self.temp_video_path)
            except Exception:
                pass
    
    def __init__(self, input_dir: str, output_path: str, fps: int = 30, 
                 num_workers: int = None):
        """
        Initialize the timelapse generator.
        
        Args:
            input_dir: Directory containing input images
            output_path: Path for output video file
            fps: Frames per second for output video
            num_workers: Number of worker threads (None for auto)
        """
        try:
            self.input_dir = validate_path(input_dir, must_exist=True)
            if not self.input_dir.is_dir():
                raise ValidationError(f"Input must be a directory: {input_dir}")
            
            output_path_obj = Path(output_path)
            if output_path_obj.parent.exists():
                self.output_path = output_path_obj
            else:
                raise ValidationError(f"Output directory does not exist: {output_path_obj.parent}")
            
            self.fps = validate_numeric_range(fps, 1, 120, "FPS")
            
        except (ValidationError, SecurityError) as e:
            logger.error(f"Initialization failed: {e}")
            raise
            
        self.images = []
        self.temp_video_path = self.output_path.with_suffix('.temp.mp4')
        
        if num_workers:
            self.num_workers = validate_numeric_range(num_workers, 1, Config.MAX_WORKERS, "Workers")
        else:
            self.num_workers = min(os.cpu_count() or 1, Config.MAX_WORKERS)
        
        self.image_editor = ImageEditor()
        
        self._ffmpeg_process = None
        
    def load_images(self, pattern: str = '*.jpg') -> List[str]:
        """
        Load and sort images from directory by creation/modification date.
        
        Args:
            pattern: File pattern to match
        """
        if '..' in pattern or '/' in pattern or '\\' in pattern:
            raise SecurityError("Invalid pattern: path separators not allowed")
            
        image_paths = glob.glob(str(self.input_dir / pattern))
        if not image_paths:
            raise ValueError(f"No images found matching {pattern} in {self.input_dir}")
        
        try:
            image_paths.sort(key=lambda x: os.path.getmtime(x))
        except OSError as e:
            logger.warning(f"Error sorting by modification time: {e}")
            image_paths.sort()
        
        self.images = image_paths
        logger.info(f"Found {len(self.images)} images, using {self.num_workers} workers")
        return self.images
    
    def _process_image_batch(self, img_paths: List[str], config: VideoConfig,
                            duration: float, overlay_configs: dict,
                            effects_config: dict) -> List[np.ndarray]:
        """Process a batch of images in parallel."""
        processed = []
        
        total_frames = int(duration * self.fps)
        frames_per_image = max(1, total_frames // len(self.images))
        
        def process_single(img_path):
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            if img.shape[:2] != (config.height, config.width):
                interpolation = cv2.INTER_AREA if img.shape[0] > config.height else cv2.INTER_LANCZOS4
                img = cv2.resize(img, (config.width, config.height), interpolation=interpolation)
            
            img = self.image_editor.apply_effects(img, **effects_config)
            
            if overlay_configs.get('show_date'):
                img = self.image_editor.add_datetime(img, img_path)
            
            if overlay_configs.get('text'):
                img = self.image_editor.add_text(img, overlay_configs['text'])
            
            if overlay_configs.get('logo') is not None:
                img = self.image_editor.add_logo(img, overlay_configs['logo'])
            
            if overlay_configs.get('watermark') is not None:
                img = self.image_editor.add_watermark(img, overlay_configs['watermark'],
                                        overlay_configs['watermark_transparency'])
            
            return img
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_single, path): path
                      for path in img_paths}

            # Timeout for entire batch = per-image timeout * number of images
            batch_timeout = Config.IMAGE_PROCESS_TIMEOUT * len(img_paths)

            try:
                for future in as_completed(futures, timeout=batch_timeout):
                    img_path = futures[future]
                    try:
                        result = future.result(timeout=5)  # Short timeout since future is already complete
                        if result is not None:
                            for _ in range(frames_per_image):
                                processed.append(result)
                    except Exception as e:
                        logger.warning(f"Error processing image {img_path}: {e}, skipping")
            except TimeoutError:
                # Batch timed out - cancel remaining futures and continue
                not_done = [f for f in futures if not f.done()]
                for f in not_done:
                    f.cancel()
                    img_path = futures[f]
                    logger.warning(f"Timeout - cancelled image: {img_path}")
                logger.warning(f"Batch timed out, {len(not_done)} images skipped")

        return processed
    
    def create_video_ffmpeg(self, quality: str = '720', duration: float = 30.0,
                           show_date: bool = False,
                           text: Optional[str] = None,
                           logo_path: Optional[str] = None,
                           watermark_path: Optional[str] = None,
                           watermark_size: Optional[Tuple[int, int]] = None,
                           watermark_transparency: float = 0.3,
                           brightness: float = 0.5,
                           contrast: float = 1.0,
                           saturation: float = 1.0,
                           batch_size: int = 100,
                           use_gpu: bool = False):
        """
        Create video using FFmpeg directly.
        
        Args:
            quality: Video quality (720, HD, 4K)
            duration: Video duration in seconds (30-180)
            show_date: Whether to show date overlay (top-left, format: YYYY-MM-DD HH:MM:SS)
            text: Custom text to overlay (bottom-center)
            logo_path: Path to logo image (top-right at 50x50)
            watermark_path: Path to watermark image (center)
            watermark_size: Watermark size (width, height)
            watermark_transparency: Watermark transparency (0-1)
            brightness: Brightness adjustment (0-1, 0.5 is neutral)
            contrast: Contrast adjustment (0-3, 1.0 is neutral)
            saturation: Saturation adjustment (0-3, 1.0 is neutral)
            batch_size: Number of images to process at once
            use_gpu: Use GPU acceleration if available
        """
        if not self.images:
            raise ValueError("No images loaded. Call load_images() first.")
        
        duration = validate_numeric_range(duration, Config.MIN_VIDEO_DURATION, 
                                        Config.MAX_VIDEO_DURATION, "Duration")
        batch_size = validate_numeric_range(batch_size, Config.MIN_BATCH_SIZE,
                                          Config.MAX_BATCH_SIZE, "Batch size")
        brightness = validate_numeric_range(brightness, 0, 1, "Brightness")
        contrast = validate_numeric_range(contrast, 0, 3, "Contrast")
        saturation = validate_numeric_range(saturation, 0, 3, "Saturation")
        watermark_transparency = validate_numeric_range(watermark_transparency, 0, 1, 
                                                      "Watermark transparency")
        
        if not check_memory_available(Config.MIN_AVAILABLE_MEMORY_MB):
            raise RuntimeError(f"Insufficient memory available. Need at least {Config.MIN_AVAILABLE_MEMORY_MB}MB")
            
        if not check_disk_space(self.output_path.parent, Config.MIN_AVAILABLE_DISK_GB):
            raise RuntimeError(f"Insufficient disk space. Need at least {Config.MIN_AVAILABLE_DISK_GB}GB")
        
        if quality not in self.QUALITY_PRESETS:
            logger.warning(f"Unknown quality preset '{quality}', using '720'")
            quality = '720'
            
        width, height, bitrate, crf = self.QUALITY_PRESETS[quality]
        config = VideoConfig(width, height, self.fps, quality, bitrate, crf=crf)
        
        logo = None
        if logo_path and os.path.exists(logo_path):
            logo = self.image_editor.prepare_logo(logo_path)
        
        watermark = None
        if watermark_path and os.path.exists(watermark_path):
            watermark = self.image_editor.prepare_watermark(watermark_path, watermark_size, watermark_transparency)
        
        overlay_configs = {
            'show_date': show_date,
            'text': text,
            'logo': logo,
            'watermark': watermark,
            'watermark_transparency': watermark_transparency
        }
        
        effects_config = {
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation
        }
        
        codec = 'h264_nvenc' if use_gpu else 'libx264'
        
        ffmpeg_cmd = ['ffmpeg', '-y']
        ffmpeg_cmd.extend(['-f', 'rawvideo'])
        ffmpeg_cmd.extend(['-vcodec', 'rawvideo'])
        ffmpeg_cmd.extend(['-s', f'{width}x{height}'])
        ffmpeg_cmd.extend(['-pix_fmt', 'bgr24'])
        ffmpeg_cmd.extend(['-r', str(self.fps)])
        ffmpeg_cmd.extend(['-i', '-'])
        ffmpeg_cmd.extend(['-c:v', codec])
        ffmpeg_cmd.extend(['-preset', config.preset])
        ffmpeg_cmd.extend(['-crf', str(crf)])
        ffmpeg_cmd.extend(['-pix_fmt', 'yuv420p'])
        ffmpeg_cmd.extend(['-b:v', bitrate])
        ffmpeg_cmd.extend(['-t', str(duration)])
        ffmpeg_cmd.append(str(self.temp_video_path))
        
        logger.info(f"Starting FFmpeg with command: {' '.join(ffmpeg_cmd)}")
        
        try:
            self._ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=Config.FFMPEG_BUFFER_SIZE
            )
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
        except Exception as e:
            raise RuntimeError(f"Failed to start FFmpeg: {e}")
        
        total_batches = (len(self.images) + batch_size - 1) // batch_size
        
        try:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.images))
                batch_paths = self.images[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} "
                           f"({end_idx}/{len(self.images)} images)")
                
                if self._ffmpeg_process.poll() is not None:
                    stderr = self._ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                    raise RuntimeError(f"FFmpeg process died: {stderr}")
                
                processed_batch = self._process_image_batch(batch_paths, config, duration,
                                                          overlay_configs, effects_config)
                
                for img in processed_batch:
                    try:
                        self._ffmpeg_process.stdin.write(img.tobytes())
                    except BrokenPipeError:
                        stderr = self._ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                        raise RuntimeError(f"FFmpeg pipe broken: {stderr}")
                
                del processed_batch
        
        finally:
            if self._ffmpeg_process and self._ffmpeg_process.stdin:
                self._ffmpeg_process.stdin.close()
                
            if self._ffmpeg_process:
                try:
                    self._ffmpeg_process.wait(timeout=Config.FFMPEG_TIMEOUT)
                except subprocess.TimeoutExpired:
                    logger.error("FFmpeg timeout, killing process")
                    self._ffmpeg_process.kill()
                    raise RuntimeError("FFmpeg process timed out")
                
                if self._ffmpeg_process.returncode != 0:
                    stderr = self._ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                    raise RuntimeError(f"FFmpeg failed with code {self._ffmpeg_process.returncode}: {stderr}")
        
        logger.info(f"Video created: {self.temp_video_path}")
        self._ffmpeg_process = None
    
    def add_music(self, music_path: str, loop: bool = True, volume: float = 1.0):
        """
        Add music using FFmpeg directly with proper validation.
        """
        logger.info("Adding music to video...")
        
        try:
            music_path_validated = validate_path(music_path, must_exist=True)
            volume = validate_numeric_range(volume, 0, 2, "Volume")
        except (ValidationError, SecurityError) as e:
            raise ValueError(f"Invalid music parameters: {e}")
        
        if not self.temp_video_path.exists():
            raise RuntimeError("Temporary video file not found. Create video first.")
        
        output_path = str(self.output_path)
        temp_path = str(self.temp_video_path)
        
        cmd = ['ffmpeg', '-y']
        cmd.extend(['-i', temp_path])
        
        if loop:
            cmd.extend(['-stream_loop', '-1'])
            
        cmd.extend(['-i', str(music_path_validated)])
        
        cmd.extend([
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-map', '0:v:0',
            '-map', '1:a:0'
        ])
        
        if volume != 1.0:
            cmd.extend(['-af', f'volume={volume:.2f}'])
        
        cmd.append(output_path)
        
        logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=Config.FFMPEG_TIMEOUT,
                check=False
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FFmpeg timed out after {Config.FFMPEG_TIMEOUT} seconds")
        
        if result.returncode != 0:
            logger.error(f"Error adding music: {result.stderr}")
            self._add_music_fallback(str(music_path_validated), loop, volume)
        else:
            try:
                if Path(temp_path).exists():
                    os.remove(temp_path)
            except OSError as e:
                logger.warning(f"Could not remove temp file: {e}")
            logger.info(f"Final video saved: {output_path}")
    
    def _add_music_fallback(self, music_path: str, loop: bool, volume: float):
        """Fallback to moviepy for audio processing."""
        video = VideoFileClip(str(self.temp_video_path))
        audio = AudioFileClip(music_path)
        
        if volume != 1.0:
            audio = audio.volumex(volume)
        
        if loop and audio.duration < video.duration:
            n_loops = int(video.duration / audio.duration) + 1
            audio = audio.with_fps(audio.fps).looped(n_loops)
        
        audio = audio.subclipped(0, video.duration)
        final_video = video.with_audio(audio)
        
        final_video.write_videofile(
            str(self.output_path),
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp_audio.m4a',
            remove_temp=True,
            threads=self.num_workers
        )
        
        video.close()
        audio.close()
        final_video.close()
        
        if self.temp_video_path.exists():
            os.remove(self.temp_video_path)