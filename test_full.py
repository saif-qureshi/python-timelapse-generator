#!/usr/bin/env python3
"""
Full test script for 4K video generation with 850 images.
Run: python test_full.py
"""

from timelapse_generator import TimelapseGenerator
import time
import psutil

def get_memory_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

input_dir = '/home/saif-qureshi/projects/ksa-timelapse/storage/app/public/camera_7/camera_7'
output_path = '/tmp/test_video_4k_full.mp4'

print(f'Starting memory: {get_memory_mb():.1f} MB')

gen = TimelapseGenerator(input_dir, output_path, fps=30, num_workers=2)
images = gen.load_images('*.jpg')

# Limit to exactly 850 images
gen.images = images[:850]
print(f'Using {len(gen.images)} images')

print(f'After loading: {get_memory_mb():.1f} MB')
print('Creating 4K video (120s duration, batch_size=100)...')

start = time.time()
gen.create_video_ffmpeg(quality='4K', duration=120, batch_size=100)
elapsed = time.time() - start

print(f'Done in {elapsed:.1f}s')
print(f'Final memory: {get_memory_mb():.1f} MB')
print(f'Output: {output_path}')
