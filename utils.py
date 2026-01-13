#!/usr/bin/env python3
"""
Utilities Module
Security, validation, and helper functions.
"""

import re
import os
import shlex
from pathlib import Path
from typing import List, Optional, Union, Generator, Iterator
import tempfile
import psutil

TIMESTAMP_PATTERN = re.compile(r'^\d{14}$')
UNSAFE_FILENAME_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
PATH_SEPARATOR_PATTERN = re.compile(r'[/\\]')


class SecurityError(Exception):
    """Raised when security validation fails."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class Config:
    """Application configuration constants."""
    MAX_IMAGE_SIZE = 100_000_000
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    DEFAULT_LOGO_SIZE = (50, 50)
    DEFAULT_FPS = 30
    DEFAULT_BATCH_SIZE = 100
    MAX_VIDEO_DURATION = 180  # seconds
    MIN_VIDEO_DURATION = 30
    
    MAX_BATCH_SIZE = 1000
    MIN_BATCH_SIZE = 10
    MAX_WORKERS = 32
    
    DEFAULT_FONT_SIZE = 30
    DEFAULT_TEXT_COLOR = (255, 255, 255)
    TEXT_PADDING = 20
    TEXT_BACKGROUND_PADDING = 5
    
    FFMPEG_TIMEOUT = 3600
    FFMPEG_BUFFER_SIZE = 10 * 1024 * 1024
    IMAGE_PROCESS_TIMEOUT = 30  # seconds per image
    
    MIN_AVAILABLE_MEMORY_MB = 500
    MIN_AVAILABLE_DISK_GB = 1


def validate_path(path: Union[str, Path], must_exist: bool = True, 
                 allow_absolute: bool = True) -> Path:
    """
    Validate and sanitize file paths.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        allow_absolute: Whether to allow absolute paths
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
        SecurityError: If path contains security risks
    """
    if not path:
        raise ValidationError("Path cannot be empty")
    
    path_obj = Path(str(path))
    
    parts = path_obj.parts
    if '..' in parts:
        raise SecurityError("Path traversal detected: '..' not allowed")
    
    if '\0' in str(path):
        raise SecurityError("Null bytes in path not allowed")
    
    try:
        if path_obj.is_absolute():
            if not allow_absolute:
                raise SecurityError("Absolute paths not allowed")
            resolved = path_obj.resolve()
        else:
            resolved = Path.cwd() / path_obj
            resolved = resolved.resolve()
    except Exception as e:
        raise ValidationError(f"Invalid path: {e}")
    
    if must_exist and not resolved.exists():
        raise ValidationError(f"Path does not exist: {resolved}")
    
    return resolved


def validate_file_extension(file_path: Path, allowed_extensions: set) -> None:
    """
    Validate file has allowed extension.
    
    Args:
        file_path: Path to file
        allowed_extensions: Set of allowed extensions (with dots)
        
    Raises:
        ValidationError: If extension not allowed
    """
    ext = file_path.suffix.lower()
    if ext not in allowed_extensions:
        raise ValidationError(
            f"File type '{ext}' not allowed. "
            f"Allowed types: {', '.join(sorted(allowed_extensions))}"
        )


def validate_image_path(path: Union[str, Path]) -> Path:
    """
    Validate image file path.
    
    Args:
        path: Path to image
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If not a valid image
    """
    validated = validate_path(path, must_exist=True)
    validate_file_extension(validated, Config.ALLOWED_IMAGE_EXTENSIONS)
    
    size = validated.stat().st_size
    if size > Config.MAX_IMAGE_SIZE:
        raise ValidationError(
            f"Image file too large: {size / 1_000_000:.1f}MB "
            f"(max: {Config.MAX_IMAGE_SIZE / 1_000_000}MB)"
        )
    
    return validated


def sanitize_text(text: str, max_length: int = 200) -> str:
    """
    Sanitize text for overlay.
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    sanitized = ''.join(
        c for c in text 
        if c.isprintable() or c in '\n\t'
    )
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    filename = os.path.basename(filename)
    
    sanitized = UNSAFE_FILENAME_PATTERN.sub('_', filename)
    
    if not sanitized or sanitized == '.':
        sanitized = 'unnamed'
    
    return sanitized


def safe_subprocess_args(args: List[str]) -> List[str]:
    """
    Sanitize subprocess arguments to prevent injection.
    
    Args:
        args: List of arguments
        
    Returns:
        Sanitized arguments
    """
    return [shlex.quote(str(arg)) for arg in args]


def create_secure_temp_dir(prefix: str = "timelapse_") -> Path:
    """
    Create secure temporary directory.
    
    Args:
        prefix: Directory name prefix
        
    Returns:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return Path(temp_dir)


def check_disk_space(path: Union[str, Path], required_gb: float = 1.0) -> bool:
    """
    Check if enough disk space is available.
    
    Args:
        path: Path to check (uses mount point)
        required_gb: Required space in GB
        
    Returns:
        True if enough space available
    """
    path = Path(path)
    stat = os.statvfs(path)
    available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    return available_gb >= required_gb


def check_memory_available(required_mb: int = 500) -> bool:
    """
    Check if enough memory is available.
    
    Args:
        required_mb: Required memory in MB
        
    Returns:
        True if enough memory available
    """
    memory = psutil.virtual_memory()
    available_mb = memory.available / (1024**2)
    return available_mb >= required_mb


def validate_numeric_range(value: Union[int, float], min_val: Union[int, float], 
                         max_val: Union[int, float], name: str) -> Union[int, float]:
    """
    Validate numeric value is within range.
    
    Args:
        value: Value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Parameter name for error message
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value out of range
    """
    if not min_val <= value <= max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )
    return value


def parse_timestamp_filename(filename: str) -> Optional[str]:
    """
    Parse timestamp from filename in YYYYMMDDHHmmss format.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Timestamp string if valid, None otherwise
    """
    base = os.path.splitext(filename)[0]
    
    if TIMESTAMP_PATTERN.match(base):
        return base
    
    return None


def load_images_lazy(image_paths: List[str], chunk_size: int = 10) -> Generator[List[str], None, None]:
    """
    Lazy load image paths in chunks for memory efficiency.
    
    Args:
        image_paths: List of image paths
        chunk_size: Number of images per chunk
        
    Yields:
        Chunks of image paths
    """
    for i in range(0, len(image_paths), chunk_size):
        yield image_paths[i:i + chunk_size]


def filter_valid_images(paths: Iterator[str]) -> Generator[str, None, None]:
    """
    Filter and yield only valid image paths.
    
    Args:
        paths: Iterator of file paths
        
    Yields:
        Valid image paths
    """
    for path in paths:
        try:
            path_obj = Path(path)
            if path_obj.exists() and path_obj.suffix.lower() in Config.ALLOWED_IMAGE_EXTENSIONS:
                yield str(path_obj)
        except Exception:
            continue