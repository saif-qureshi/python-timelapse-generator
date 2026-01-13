#!/usr/bin/env python3
"""
Test script to verify image processing timeout behavior.
"""

import multiprocessing as mp
import cv2
import time
import sys
from pathlib import Path
from typing import Optional
import numpy as np


def process_image_with_timeout(img_path: str, width: int, height: int, timeout: int = 10) -> Optional[np.ndarray]:
    """
    Process a single image with a hard timeout.
    Uses multiprocessing.Process which CAN be killed.
    """
    result_queue = mp.Queue()

    def worker(path, w, h, queue):
        try:
            img = cv2.imread(path)
            if img is None:
                queue.put(None)
                return

            if img.shape[:2] != (h, w):
                interpolation = cv2.INTER_AREA if img.shape[0] > h else cv2.INTER_LANCZOS4
                img = cv2.resize(img, (w, h), interpolation=interpolation)

            queue.put(img)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            queue.put(None)

    # Create process
    p = mp.Process(target=worker, args=(img_path, width, height, result_queue))
    p.start()

    # Wait with timeout
    p.join(timeout=timeout)

    if p.is_alive():
        # Process is still running - KILL IT
        print(f"TIMEOUT: Killing process for {img_path}")
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()  # Force kill if terminate didn't work
            p.join()
        return None

    # Get result from queue
    if not result_queue.empty():
        return result_queue.get()
    return None


def test_batch_processing(image_dir: str, batch_size: int = 100):
    """Test processing a batch of images with timeout."""

    image_paths = sorted(list(Path(image_dir).glob("*.jpg")))[:batch_size]
    print(f"Testing with {len(image_paths)} images from {image_dir}")

    width, height = 1920, 1080  # HD
    timeout_per_image = 10  # seconds

    processed = []
    skipped = 0

    start_time = time.time()

    for i, img_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {img_path.name}", end=" ... ")

        result = process_image_with_timeout(str(img_path), width, height, timeout_per_image)

        if result is not None:
            processed.append(result)
            print("OK")
        else:
            skipped += 1
            print("SKIPPED")

    elapsed = time.time() - start_time

    print(f"\n--- Results ---")
    print(f"Total images: {len(image_paths)}")
    print(f"Processed: {len(processed)}")
    print(f"Skipped: {skipped}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Avg per image: {elapsed/len(image_paths):.2f}s")


if __name__ == "__main__":
    # Default to the downloaded test photos
    default_dir = "/home/saif-qureshi/projects/ksa-timelapse/storage/app/public/camera_7/camera_7"

    image_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    test_batch_processing(image_dir, batch_size)
