# KSA Timelapse Generator

A powerful Python tool for creating timelapse videos from image sequences with customizable effects, overlays, and music.

## Features

- **Timelapse Video Generation**: Create smooth timelapse videos from image sequences
- **Photo Export**: Export photos with date/time overlays as ZIP archives
- **Image Effects**: Adjust brightness, contrast, and saturation
- **Overlays**: Add date/time stamps, custom text, logos, and watermarks
- **Music Support**: Add background music with looping and volume control
- **GPU Acceleration**: Optional NVIDIA GPU support for faster encoding
- **Laravel Integration**: Special wrapper for timestamp-based photo structures

## Installation

### Requirements

- Python 3.8+
- FFmpeg
- Required Python packages:
  ```bash
  pip install opencv-python pillow moviepy numpy
  ```

### Optional

- NVIDIA GPU with NVENC support for hardware acceleration
- Font package for text overlays:
  ```bash
  sudo apt-get install fonts-dejavu  # Ubuntu/Debian
  ```

## Usage

### Basic Timelapse Video

```bash
python main.py /path/to/images output.mp4
```

### With Options

```bash
python main.py /path/to/images output.mp4 \
  --resolution HD \
  --duration 60 \
  --fps 30 \
  --show-date \
  --text "My Timelapse" \
  --logo logo.png \
  --music background.mp3
```

### Export Photos with Date Overlay

```bash
python main.py --mode export /path/to/images photos.zip --show-date
```

### Laravel Integration (Timestamp-based Photos)

```bash
python wrapper.py /path/to/camera/photos output.mp4 \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --start-hour 8 \
  --end-hour 17
```

## Command Line Options

### Common Options

- `input_dir`: Directory containing input images
- `output`: Output file path (video or zip)
- `--mode`: Operation mode: `video` (default) or `export`
- `--pattern`: Image file pattern (default: `*.jpg`)
- `--workers`: Number of worker threads
- `--batch-size`: Images to process per batch (default: 100)

### Video Options

- `--resolution`: Video resolution: `720`, `HD`, `4K` (default: `720`)
- `--duration`: Video duration in seconds: 30-180 (default: 30)
- `--fps`: Frames per second (default: 30)
- `--gpu`: Use GPU acceleration (requires NVENC)

### Overlay Options

- `--show-date`: Show date/time on images
- `--text`: Custom text overlay
- `--logo`: Path to logo image (displayed at 50x50)
- `--watermark`: Path to watermark image
- `--watermark-size`: Watermark size (WIDTH HEIGHT)
- `--watermark-transparency`: Watermark transparency 0-1 (default: 0.3)

### Effect Options

- `--brightness`: Brightness 0-1 (default: 0.5)
- `--contrast`: Contrast 0-3 (default: 1.0)
- `--saturation`: Saturation 0-3 (default: 1.0)

### Music Options

- `--music`: Path to music file

### Laravel Wrapper Options

- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)
- `--start-hour`: Start hour 0-23 (default: 8)
- `--end-hour`: End hour 0-23 (default: 17)

## Architecture

The project is organized into modular components:

- `main.py`: Command-line interface and entry point
- `timelapse_generator.py`: Core video generation functionality
- `photo_exporter.py`: Photo export with overlays
- `image_editor.py`: Shared image processing operations
- `wrapper.py`: Laravel integration for timestamp-based photos

## Examples

### Create a 60-second HD timelapse with date overlay

```bash
python main.py ./photos output.mp4 \
  --resolution HD \
  --duration 60 \
  --show-date
```

### Add logo and background music

```bash
python main.py ./photos output.mp4 \
  --logo company_logo.png \
  --music background_track.mp3 \
  --text "Construction Progress"
```

### Export photos with timestamps

```bash
python main.py --mode export ./photos timestamped_photos.zip --show-date
```

### Process only work hours for a month

```bash
python wrapper.py /var/photos/camera1 january_timelapse.mp4 \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --start-hour 8 \
  --end-hour 17 \
  --resolution 4K
```

## Performance Tips

- Use `--gpu` flag if you have an NVIDIA GPU with NVENC support
- Adjust `--batch-size` based on available memory
- Use `--workers` to set the number of parallel threads

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.