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
import tempfile
import queue
import threading
import collections
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List
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

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s:%(name)s:%(message)s')
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

    def __init__(self, output_path: str,
                 num_workers: int = None,
                 s3_client=None, photo_list: list = None):
        """
        Initialize the timelapse generator.

        Args:
            output_path: Path for output video file (local temp path)
            fps: Frames per second for output video
            num_workers: Number of worker threads (None for auto)
            s3_client: S3Client instance for reading photos
            photo_list: List of photo dicts from s3_client.list_photos()
        """
        try:
            output_path_obj = Path(output_path)
            if output_path_obj.parent.exists():
                self.output_path = output_path_obj
            else:
                raise ValidationError(f"Output directory does not exist: {output_path_obj.parent}")

            self.fps = 30  # Default, overridden by create_video_ffmpeg based on duration

        except (ValidationError, SecurityError) as e:
            logger.error(f"Initialization failed: {e}")
            raise

        self.s3_client = s3_client
        self.photo_list = photo_list or []
        self.images = []
        self.temp_video_path = self.output_path.with_suffix('.temp.mp4')

        if num_workers:
            self.num_workers = validate_numeric_range(num_workers, 1, Config.MAX_WORKERS, "Workers")
        else:
            self.num_workers = min(os.cpu_count() or 1, Config.MAX_WORKERS)

        self.image_editor = ImageEditor()

        self._ffmpeg_process = None

    def load_images(self, pattern: str = '*.jpg') -> list:
        """
        Load images from the pre-supplied photo list.
        The list is already filtered and sorted by the wrapper/s3_client.
        """
        if not self.photo_list:
            raise ValueError("No images in photo list")

        self.images = self.photo_list
        logger.info(f"Loaded {len(self.images)} images from S3")
        return self.images

    def _decode_s3_image(self, photo_info: dict) -> Optional[np.ndarray]:
        """Download and decode an image from S3 in memory."""
        img_data = self.s3_client.download_bytes(photo_info['key'])
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        del img_data, img_array
        return img

    def _decode_s3_asset(self, s3_key: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
        """Download and decode a small asset (logo/watermark) from S3."""
        try:
            data = self.s3_client.download_bytes(s3_key)
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, flags)
            del data, arr
            return img
        except Exception as e:
            logger.warning(f"Failed to load asset from S3 '{s3_key}': {e}")
            return None

    def _prefetch_worker(self, image_queue: queue.Queue, stop_event: threading.Event):
        """Background thread that downloads and decodes images ahead of the main loop."""
        for idx, photo_info in enumerate(self.images):
            if stop_event.is_set():
                break
            try:
                img = self._decode_s3_image(photo_info)
            except Exception as e:
                logger.warning(f"Prefetch failed for {photo_info.get('key', '?')}: {e}")
                img = None
            # Use timeout-based put so we can check stop_event if queue is full
            while not stop_event.is_set():
                try:
                    image_queue.put((idx, photo_info, img), timeout=1)
                    break
                except queue.Full:
                    continue
            else:
                break
        # Sentinel to signal end of images (also needs timeout to avoid blocking)
        while not stop_event.is_set():
            try:
                image_queue.put(None, timeout=1)
                break
            except queue.Full:
                continue

    @staticmethod
    def _drain_stderr(pipe, collected: collections.deque):
        """Background thread to drain FFmpeg stderr pipe and collect recent lines."""
        for line in pipe:
            decoded = line.decode('utf-8', errors='replace').rstrip()
            collected.append(decoded)

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
        Create video using FFmpeg with S3 in-memory streaming.

        logo_path, watermark_path: S3 keys for assets
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

        # Pre-download and decode assets from S3 (small files, loaded once)
        logo_img = None
        if logo_path and self.s3_client.exists(logo_path):
            logo_raw = self._decode_s3_asset(logo_path)
            if logo_raw is not None:
                logo_img = self.image_editor.prepare_logo_from_array(logo_raw, cache_key=logo_path)

        watermark_img = None
        if watermark_path and self.s3_client.exists(watermark_path):
            wm_raw = self._decode_s3_asset(watermark_path)
            if wm_raw is not None:
                watermark_img = self.image_editor.prepare_watermark_from_array(
                    wm_raw, watermark_size, watermark_transparency, cache_key=watermark_path
                )

        overlay_configs = {
            'show_date': show_date,
            'text': text,
        }

        effects_config = {
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation
        }

        codec = 'h264_nvenc' if use_gpu else 'libx264'

        # Calculate FPS from actual image count to fit all images in requested duration
        actual_fps = max(1, round(len(self.images) / duration))
        self.fps = actual_fps
        logger.info(f"Adjusted FPS to {actual_fps} to fit {len(self.images)} images in {duration}s")

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

        stderr_lines = collections.deque(maxlen=100)

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

        # Background thread to drain stderr (prevents deadlock since main thread writes to stdin)
        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._ffmpeg_process.stderr, stderr_lines),
            daemon=True
        )
        stderr_thread.start()

        frames_per_image = 1

        # Prefetch images in background thread
        image_queue = queue.Queue(maxsize=10)
        stop_event = threading.Event()
        prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(image_queue, stop_event),
            daemon=True
        )
        prefetch_thread.start()

        try:
            while True:
                # Timeout prevents hanging if prefetch thread dies without sending sentinel
                try:
                    item = image_queue.get(timeout=60)
                except queue.Empty:
                    if not prefetch_thread.is_alive():
                        logger.warning("Prefetch thread died, ending video loop")
                        break
                    continue

                if item is None:
                    break  # Sentinel: all images processed

                idx, photo_info, img = item

                if idx % 100 == 0:
                    logger.info(f"Processing image {idx + 1}/{len(self.images)}")

                if self._ffmpeg_process.poll() is not None:
                    stop_event.set()
                    raise RuntimeError(f"FFmpeg process died with code {self._ffmpeg_process.returncode}")

                try:
                    if img is None:
                        logger.warning(f"Could not decode image: {photo_info['key']}, skipping")
                        continue

                    if img.shape[:2] != (config.height, config.width):
                        interpolation = cv2.INTER_AREA if img.shape[0] > config.height else cv2.INTER_LANCZOS4
                        img = cv2.resize(img, (config.width, config.height), interpolation=interpolation)

                    img = self.image_editor.apply_effects(img, **effects_config)

                    if overlay_configs.get('show_date'):
                        img = self.image_editor.add_datetime(img, photo_info=photo_info)
                    if overlay_configs.get('text'):
                        img = self.image_editor.add_text(img, overlay_configs['text'])
                    if logo_img is not None:
                        img = self.image_editor.add_logo(img, logo_img)
                    if watermark_img is not None:
                        img = self.image_editor.add_watermark(img, watermark_img, watermark_transparency)

                    img_bytes = img.tobytes()
                    for _ in range(frames_per_image):
                        self._ffmpeg_process.stdin.write(img_bytes)

                    del img, img_bytes

                except BrokenPipeError:
                    logger.info("FFmpeg closed pipe (enough frames received)")
                    stop_event.set()
                    break
                except Exception as e:
                    logger.warning(f"Skipping image {photo_info.get('key', '?')}: {e}")

        finally:
            stop_event.set()

            # Drain remaining queue items so prefetch thread can unblock
            while not image_queue.empty():
                try:
                    image_queue.get_nowait()
                except queue.Empty:
                    break
            prefetch_thread.join(timeout=10)

            if self._ffmpeg_process and self._ffmpeg_process.stdin:
                self._ffmpeg_process.stdin.close()

            if self._ffmpeg_process:
                try:
                    self._ffmpeg_process.wait(timeout=Config.FFMPEG_TIMEOUT)
                except subprocess.TimeoutExpired:
                    logger.error("FFmpeg timeout, killing process")
                    self._ffmpeg_process.kill()
                    raise RuntimeError("FFmpeg process timed out")

                stderr_thread.join(timeout=5)

                if self._ffmpeg_process.returncode != 0:
                    tail = list(stderr_lines)[-20:]
                    if tail:
                        logger.error("FFmpeg stderr (last 20 lines):\n" + "\n".join(tail))
                    last_line = tail[-1] if tail else "unknown error"
                    raise RuntimeError(f"FFmpeg failed with code {self._ffmpeg_process.returncode}: {last_line}")
                else:
                    if stderr_lines:
                        logger.debug(f"FFmpeg completed with {len(stderr_lines)} stderr lines (capped at 100)")

        logger.info(f"Video created: {self.temp_video_path}")
        self._ffmpeg_process = None

    def finalize(self):
        """
        Finalize video by renaming temp file to final output.
        Call this when not adding music.
        """
        if self.temp_video_path.exists():
            shutil.move(str(self.temp_video_path), str(self.output_path))
            logger.info(f"Final video saved: {self.output_path}")
        else:
            raise RuntimeError("Temp video file not found. Create video first.")

    def add_music(self, music_s3_key: str, loop: bool = True, volume: float = 1.0):
        """
        Add music to the video. Downloads music from S3 to a temp file
        since FFmpeg requires a seekable local file for audio input.
        """
        logger.info("Adding music to video...")

        volume = validate_numeric_range(volume, 0, 2, "Volume")

        if not self.temp_video_path.exists():
            raise RuntimeError("Temporary video file not found. Create video first.")

        # Download music from S3 to temp file
        music_bytes = self.s3_client.download_bytes(music_s3_key)
        ext = Path(music_s3_key).suffix or '.mp3'
        tmp_music = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        try:
            tmp_music.write(music_bytes)
            tmp_music.close()
            del music_bytes

            output_path = str(self.output_path)
            temp_path = str(self.temp_video_path)

            cmd = ['ffmpeg', '-y']
            cmd.extend(['-i', temp_path])

            if loop:
                cmd.extend(['-stream_loop', '-1'])

            cmd.extend(['-i', tmp_music.name])

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

            logger.info(f"Executing FFmpeg music command: {' '.join(cmd)}")

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
                self._add_music_fallback(tmp_music.name, loop, volume)
            else:
                try:
                    if Path(temp_path).exists():
                        os.remove(temp_path)
                except OSError as e:
                    logger.warning(f"Could not remove temp file: {e}")
                logger.info(f"Final video saved: {output_path}")
        finally:
            try:
                os.unlink(tmp_music.name)
            except OSError:
                pass

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
