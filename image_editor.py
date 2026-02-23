#!/usr/bin/env python3
"""
ImageEditor Module
Handles image processing operations shared between components.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
from functools import lru_cache
from utils import Config, sanitize_text


class ImageEditor:
    """Handles image editing operations including effects and overlays."""

    def __init__(self):
        """Initialize the image editor with caches."""
        self._logo_cache = {}
        self._font_cache = {}
        self._watermark_cache = {}

    def apply_effects(self, img: np.ndarray, brightness: float = 0.5,
                     contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
        """
        Apply brightness, contrast and saturation adjustments to image.

        Args:
            brightness: 0 to 1 (0.5 is neutral)
            contrast: 0 to 3 (1.0 is neutral)
            saturation: 0 to 3 (1.0 is neutral)
        """
        brightness_adjust = (brightness - 0.5) * 200
        img = cv2.convertScaleAbs(img, alpha=1, beta=brightness_adjust)

        if contrast != 1.0:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

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

    def prepare_logo_from_array(self, logo_img: np.ndarray, cache_key: str = 'default') -> Optional[np.ndarray]:
        """Prepare and cache logo from a pre-decoded numpy array."""
        if cache_key in self._logo_cache:
            return self._logo_cache[cache_key]

        if logo_img is None:
            return None

        logo = cv2.resize(logo_img, Config.DEFAULT_LOGO_SIZE, interpolation=cv2.INTER_LANCZOS4)
        self._logo_cache[cache_key] = logo
        return logo

    def prepare_watermark_from_array(self, watermark_img: np.ndarray,
                                     size: Optional[Tuple[int, int]],
                                     transparency: float,
                                     cache_key: str = 'default') -> Optional[np.ndarray]:
        """Prepare and cache watermark from a pre-decoded numpy array."""
        full_key = (cache_key, size, transparency)
        if full_key in self._watermark_cache:
            return self._watermark_cache[full_key]

        if watermark_img is None:
            return None

        if size:
            watermark_img = cv2.resize(watermark_img, size, interpolation=cv2.INTER_LANCZOS4)

        self._watermark_transparency = transparency
        self._watermark_cache[full_key] = watermark_img
        return watermark_img

    def add_datetime(self, img: np.ndarray, photo_info: dict = None,
                    img_path: str = None,
                    font_size: int = None,
                    color: Tuple = None) -> np.ndarray:
        """
        Add datetime overlay using OpenCV.

        Resolves timestamp in this priority:
        1. photo_info['timestamp'] (datetime object from S3 key parsing)
        2. Parse from photo_info['filename'] stem (YYYYMMDDHHmmss)
        3. os.path.getmtime(img_path) (local file fallback)
        4. current time
        """
        dt_str = None

        # Priority 1: photo_info timestamp
        if photo_info and photo_info.get('timestamp'):
            dt_str = photo_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        # Priority 2: parse from filename
        if dt_str is None and photo_info and photo_info.get('filename'):
            stem = Path(photo_info['filename']).stem
            if len(stem) >= 14 and stem[:14].isdigit():
                try:
                    dt_str = datetime.strptime(stem[:14], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    pass

        # Priority 3: local file mtime (backward compat)
        if dt_str is None and img_path:
            try:
                timestamp = os.path.getmtime(img_path)
                dt_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass

        # Priority 4: fallback
        if dt_str is None:
            dt_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if font_size is None:
            font_size = Config.DEFAULT_FONT_SIZE
        if color is None:
            color = Config.DEFAULT_TEXT_COLOR

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = font_size / 30.0
        thickness = max(1, int(font_scale * 2))

        text_size = cv2.getTextSize(dt_str, font, font_scale, thickness)[0]
        h, w = img.shape[:2]
        padding = Config.TEXT_PADDING

        x = padding
        y = padding + text_size[1]

        pad = Config.TEXT_BACKGROUND_PADDING
        cv2.rectangle(img, (x - pad, y - text_size[1] - pad),
                     (x + text_size[0] + pad, y + pad),
                     (0, 0, 0), -1)

        cv2.putText(img, dt_str, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return img

    def add_text(self, img: np.ndarray, text: str,
                font_size: int = 40, color: Tuple = None) -> np.ndarray:
        """Add custom text overlay."""
        text = sanitize_text(text)
        if not text:
            return img

        if color is None:
            color = Config.DEFAULT_TEXT_COLOR

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = font_size / 30.0
        thickness = max(1, int(font_scale * 2))

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        h, w = img.shape[:2]
        padding = Config.TEXT_PADDING

        x = (w - text_size[0]) // 2
        y = h - padding

        pad = Config.TEXT_BACKGROUND_PADDING
        cv2.rectangle(img, (x - pad, y - text_size[1] - pad),
                     (x + text_size[0] + pad, y + pad),
                     (0, 0, 0), -1)

        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return img

    def add_logo(self, img: np.ndarray, logo: np.ndarray) -> np.ndarray:
        """Add logo overlay at fixed position (top-right)."""
        if logo is None:
            return img

        logo_h, logo_w = logo.shape[:2]
        img_h, img_w = img.shape[:2]
        padding = Config.TEXT_PADDING

        x = img_w - logo_w - padding
        y = padding

        x = max(0, min(x, img_w - logo_w))
        y = max(0, min(y, img_h - logo_h))

        if logo.shape[2] == 4:
            alpha = logo[:, :, 3:4] / 255.0
            img[y:y+logo_h, x:x+logo_w] = \
                (alpha * logo[:, :, :3] + (1 - alpha) * img[y:y+logo_h, x:x+logo_w]).astype(np.uint8)
        else:
            img[y:y+logo_h, x:x+logo_w] = logo[:, :, :3]

        return img

    def add_watermark(self, img: np.ndarray, watermark: np.ndarray,
                     transparency: float = 0.3) -> np.ndarray:
        """Add watermark overlay to center of image."""
        if watermark is None:
            return img

        wm_h, wm_w = watermark.shape[:2]
        img_h, img_w = img.shape[:2]

        x = (img_w - wm_w) // 2
        y = (img_h - wm_h) // 2

        if x < 0 or y < 0:
            return img

        roi = img[y:y+wm_h, x:x+wm_w]

        if watermark.shape[2] == 4:
            alpha = (watermark[:, :, 3:4] / 255.0) * transparency
            img[y:y+wm_h, x:x+wm_w] = \
                (alpha * watermark[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
        else:
            img[y:y+wm_h, x:x+wm_w] = \
                (transparency * watermark[:, :, :3] + (1 - transparency) * roi).astype(np.uint8)

        return img
