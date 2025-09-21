#!/usr/bin/env python3
"""
Timelapse Video Generator
Create timelapse videos with customizable effects and overlays.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime
from pathlib import Path
import argparse
from typing import Tuple, Optional, List
import glob
from moviepy.editor import VideoFileClip, AudioFileClip
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import subprocess
from dataclasses import dataclass

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
    
    # Video quality presets (width, height, bitrate, crf)
    QUALITY_PRESETS = {
        '720': (1280, 720, '5M', 25),
        'HD': (1920, 1080, '10M', 23),
        '4K': (3840, 2160, '20M', 20)
    }
    
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
        self.input_dir = Path(input_dir)
        self.output_path = Path(output_path)
        self.fps = fps
        self.images = []
        self.temp_video_path = self.output_path.with_suffix('.temp.mp4')
        
        # Set number of workers (default to CPU count)
        self.num_workers = num_workers or os.cpu_count()
        
        # Cache for overlays
        self._logo_cache = {}
        self._font_cache = {}
        self._watermark_cache = {}
        
    def load_images(self, pattern: str = '*.jpg') -> List[str]:
        """
        Load and sort images from directory by creation/modification date.
        
        Args:
            pattern: File pattern to match
        """
        image_paths = glob.glob(str(self.input_dir / pattern))
        if not image_paths:
            raise ValueError(f"No images found matching {pattern} in {self.input_dir}")
        
        # Sort by modification time (creation time on most systems)
        image_paths.sort(key=lambda x: os.path.getmtime(x))
        
        self.images = image_paths
        print(f"Found {len(self.images)} images, using {self.num_workers} workers")
        return self.images
    
    def _apply_effects(self, img: np.ndarray, brightness: float = 0.5, 
                      contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
        """
        Apply brightness, contrast and saturation adjustments to image.
        
        Args:
            brightness: 0 to 1 (0.5 is neutral)
            contrast: 0 to 3 (1.0 is neutral)
            saturation: 0 to 3 (1.0 is neutral)
        """
        # Brightness adjustment (-100 to 100, where 0 is neutral)
        brightness_adjust = (brightness - 0.5) * 200
        img = cv2.convertScaleAbs(img, alpha=1, beta=brightness_adjust)
        
        # Contrast adjustment
        if contrast != 1.0:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        
        # Saturation adjustment
        if saturation != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return img
    
    @lru_cache(maxsize=1)
    def _get_font(self, font_size: int):
        """Cache font objects."""
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            return ImageFont.load_default()
    
    def _prepare_logo(self, logo_path: str) -> np.ndarray:
        """Prepare and cache logo overlay with fixed size 50x50."""
        cache_key = logo_path
        if cache_key in self._logo_cache:
            return self._logo_cache[cache_key]
        
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            return None
        
        # Fixed size 50x50
        logo = cv2.resize(logo, (50, 50), interpolation=cv2.INTER_LANCZOS4)
        
        self._logo_cache[cache_key] = logo
        return logo
    
    def _prepare_watermark(self, watermark_path: str, size: Optional[Tuple[int, int]], 
                          transparency: float) -> np.ndarray:
        """Prepare and cache watermark overlay."""
        cache_key = (watermark_path, size, transparency)
        if cache_key in self._watermark_cache:
            return self._watermark_cache[cache_key]
        
        watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
        if watermark is None:
            return None
        
        if size:
            watermark = cv2.resize(watermark, size, interpolation=cv2.INTER_LANCZOS4)
        
        # Store transparency for later use
        self._watermark_transparency = transparency
        
        self._watermark_cache[cache_key] = watermark
        return watermark
    
    def _add_datetime(self, img: np.ndarray, img_path: str,
                     font_size: int = 30,
                     color: Tuple = (255, 255, 255)) -> np.ndarray:
        """Add datetime overlay using OpenCV."""
        try:
            timestamp = os.path.getmtime(img_path)
            dt_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            dt_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = font_size / 30.0
        thickness = max(1, int(font_scale * 2))
        
        text_size = cv2.getTextSize(dt_str, font, font_scale, thickness)[0]
        h, w = img.shape[:2]
        padding = 20
        
        # Always top-left for datetime
        x = padding
        y = padding + text_size[1]
        
        # Add background rectangle
        cv2.rectangle(img, (x - 5, y - text_size[1] - 5),
                     (x + text_size[0] + 5, y + 5),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(img, dt_str, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return img
    
    def _add_text(self, img: np.ndarray, text: str,
                 font_size: int = 40, color: Tuple = (255, 255, 255)) -> np.ndarray:
        """Add custom text overlay."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = font_size / 30.0
        thickness = max(1, int(font_scale * 2))
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        h, w = img.shape[:2]
        padding = 20
        
        # Always bottom-center for custom text
        x = (w - text_size[0]) // 2
        y = h - padding
        
        # Add background rectangle
        cv2.rectangle(img, (x - 5, y - text_size[1] - 5),
                     (x + text_size[0] + 5, y + 5),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return img
    
    def _add_logo(self, img: np.ndarray, logo: np.ndarray) -> np.ndarray:
        """Add logo overlay at fixed position (top-right)."""
        if logo is None:
            return img
        
        logo_h, logo_w = logo.shape[:2]  # Should be 50x50
        img_h, img_w = img.shape[:2]
        padding = 20
        
        # Fixed position: top-right
        x = img_w - logo_w - padding
        y = padding
        
        x = max(0, min(x, img_w - logo_w))
        y = max(0, min(y, img_h - logo_h))
        
        # Alpha blending
        if logo.shape[2] == 4:
            alpha = logo[:, :, 3:4] / 255.0
            img[y:y+logo_h, x:x+logo_w] = \
                (alpha * logo[:, :, :3] + (1 - alpha) * img[y:y+logo_h, x:x+logo_w]).astype(np.uint8)
        else:
            img[y:y+logo_h, x:x+logo_w] = logo[:, :, :3]
        
        return img
    
    def _add_watermark(self, img: np.ndarray, watermark: np.ndarray, 
                      transparency: float = 0.3) -> np.ndarray:
        """Add watermark overlay to center of image."""
        if watermark is None:
            return img
        
        wm_h, wm_w = watermark.shape[:2]
        img_h, img_w = img.shape[:2]
        
        # Center position
        x = (img_w - wm_w) // 2
        y = (img_h - wm_h) // 2
        
        if x < 0 or y < 0:
            return img
        
        # Apply transparency
        roi = img[y:y+wm_h, x:x+wm_w]
        
        if watermark.shape[2] == 4:
            # PNG with alpha channel
            alpha = (watermark[:, :, 3:4] / 255.0) * transparency
            img[y:y+wm_h, x:x+wm_w] = \
                (alpha * watermark[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
        else:
            # JPG without alpha channel
            img[y:y+wm_h, x:x+wm_w] = \
                (transparency * watermark[:, :, :3] + (1 - transparency) * roi).astype(np.uint8)
        
        return img
    
    def _process_image_batch(self, img_paths: List[str], config: VideoConfig,
                            duration: float, overlay_configs: dict,
                            effects_config: dict) -> List[np.ndarray]:
        """Process a batch of images in parallel."""
        processed = []
        
        # Calculate how many times to repeat each image based on duration
        total_frames = int(duration * self.fps)
        frames_per_image = max(1, total_frames // len(self.images))
        
        def process_single(img_path):
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Resize image
            if img.shape[:2] != (config.height, config.width):
                interpolation = cv2.INTER_AREA if img.shape[0] > config.height else cv2.INTER_LANCZOS4
                img = cv2.resize(img, (config.width, config.height), interpolation=interpolation)
            
            # Apply effects
            img = self._apply_effects(img, **effects_config)
            
            # Apply overlays
            if overlay_configs.get('show_date'):
                img = self._add_datetime(img, img_path)
            
            if overlay_configs.get('text'):
                img = self._add_text(img, overlay_configs['text'])
            
            if overlay_configs.get('logo') is not None:
                img = self._add_logo(img, overlay_configs['logo'])
            
            if overlay_configs.get('watermark') is not None:
                img = self._add_watermark(img, overlay_configs['watermark'],
                                        overlay_configs['watermark_transparency'])
            
            return img
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_single, path): path 
                      for path in img_paths}
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    # Repeat frame based on calculated frames per image
                    for _ in range(frames_per_image):
                        processed.append(result)
        
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
        
        # Setup video configuration
        width, height, bitrate, crf = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS['720'])
        config = VideoConfig(width, height, self.fps, quality, bitrate, crf=crf)
        
        # Prepare overlays
        logo = None
        if logo_path and os.path.exists(logo_path):
            logo = self._prepare_logo(logo_path)
        
        watermark = None
        if watermark_path and os.path.exists(watermark_path):
            watermark = self._prepare_watermark(watermark_path, watermark_size, watermark_transparency)
        
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
        
        # Setup FFmpeg command
        codec = 'h264_nvenc' if use_gpu else 'libx264'
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',  # Input from pipe
            '-c:v', codec,
            '-preset', config.preset,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            '-b:v', bitrate,
            '-t', str(duration),  # Set duration
            str(self.temp_video_path)
        ]
        
        # Start FFmpeg process
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Process images in batches
        total_batches = (len(self.images) + batch_size - 1) // batch_size
        
        try:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.images))
                batch_paths = self.images[start_idx:end_idx]
                
                print(f"Processing batch {batch_idx + 1}/{total_batches} "
                      f"({end_idx}/{len(self.images)} images)", end='\r')
                
                # Process batch
                processed_batch = self._process_image_batch(batch_paths, config, duration,
                                                          overlay_configs, effects_config)
                
                # Write to FFmpeg pipe
                for img in processed_batch:
                    process.stdin.write(img.tobytes())
                
                # Clear batch from memory
                del processed_batch
        
        finally:
            process.stdin.close()
            process.wait()
        
        print(f"\nVideo created: {self.temp_video_path}")
    
    def add_music(self, music_path: str, loop: bool = True, volume: float = 1.0):
        """
        Add music using FFmpeg directly.
        """
        print("Adding music to video...")
        
        output_path = str(self.output_path)
        temp_path = str(self.temp_video_path)
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y', '-i', temp_path, '-i', music_path]
        
        if loop:
            cmd.extend(['-stream_loop', '-1'])
        
        cmd.extend([
            '-c:v', 'copy',  # Copy video stream without re-encoding
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',  # Stop when shortest stream ends
            '-map', '0:v:0',
            '-map', '1:a:0'
        ])
        
        if volume != 1.0:
            cmd.extend(['-af', f'volume={volume}'])
        
        cmd.append(output_path)
        
        # Execute FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error adding music: {result.stderr}")
            # Fall back to moviepy
            self._add_music_fallback(music_path, loop, volume)
        else:
            # Remove temporary video
            if Path(temp_path).exists():
                os.remove(temp_path)
            print(f"Final video saved: {output_path}")
    
    def _add_music_fallback(self, music_path: str, loop: bool, volume: float):
        """Fallback to moviepy for audio processing."""
        video = VideoFileClip(str(self.temp_video_path))
        audio = AudioFileClip(music_path)
        
        if volume != 1.0:
            audio = audio.volumex(volume)
        
        if loop and audio.duration < video.duration:
            n_loops = int(video.duration / audio.duration) + 1
            audio = audio.loop(n_loops)
        
        audio = audio.subclip(0, video.duration)
        final_video = video.set_audio(audio)
        
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

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Timelapse video generator')
    
    # Basic arguments
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output', help='Output video file path')
    parser.add_argument('--pattern', default='*.jpg', help='Image file pattern (default: *.jpg)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--workers', type=int, help='Number of worker threads')
    parser.add_argument('--batch-size', type=int, default=100, 
                       help='Images to process per batch')
    parser.add_argument('--gpu', action='store_true', 
                       help='Use GPU acceleration (requires NVENC)')
    
    # Video options
    parser.add_argument('--resolution', choices=['720', 'HD', '4K'], 
                       default='720', help='Video resolution (default: 720)')
    parser.add_argument('--duration', type=float, default=30,
                       choices=[30, 60, 90, 120, 150, 180],
                       help='Video duration in seconds (30-180, default: 30)')
    
    # Image & text options
    parser.add_argument('--show-date', action='store_true', 
                       help='Show date on images')
    
    parser.add_argument('--text', help='Custom text to add')
    
    parser.add_argument('--logo', help='Path to logo image (auto-placed top-right at 50x50)')
    
    parser.add_argument('--watermark', help='Path to watermark image')
    parser.add_argument('--watermark-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='Watermark size')
    parser.add_argument('--watermark-transparency', type=float, default=0.3,
                       help='Watermark transparency (0-1, default: 0.3)')
    
    # Music
    parser.add_argument('--music', help='Path to music file')
    
    # Effect options
    parser.add_argument('--brightness', type=float, default=0.5,
                       help='Brightness (0-1, default: 0.5)')
    parser.add_argument('--contrast', type=float, default=1.0,
                       help='Contrast (0-3, default: 1.0)')
    parser.add_argument('--saturation', type=float, default=1.0,
                       help='Saturation (0-3, default: 1.0)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = TimelapseGenerator(
        args.input_dir, args.output, args.fps, args.workers
    )
    
    # Load images
    generator.load_images(args.pattern)
    
    # Create video
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
    
    # Add music
    if args.music:
        generator.add_music(
            args.music, 
            loop=True,
            volume=1.0
        )
    else:
        if generator.temp_video_path.exists():
            os.rename(generator.temp_video_path, generator.output_path)
            print(f"Final video saved: {generator.output_path}")

if __name__ == '__main__':
    main()