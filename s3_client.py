#!/usr/bin/env python3
"""
S3 Client Module
Handles all S3 interactions for the timelapse generator.
"""
import os
import time
import random
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError

logger = logging.getLogger(__name__)


class S3Client:
    """Manages S3 operations for timelapse generation."""

    def __init__(self, bucket: str, region: str = 'us-east-1',
                 access_key: str = None, secret_key: str = None,
                 endpoint_url: str = None):
        self.bucket = bucket

        kwargs = {'region_name': region}
        if access_key and secret_key:
            kwargs['aws_access_key_id'] = access_key
            kwargs['aws_secret_access_key'] = secret_key
        if endpoint_url:
            kwargs['endpoint_url'] = endpoint_url

        self._client = boto3.client('s3', **kwargs)

    @classmethod
    def from_env(cls) -> 'S3Client':
        """Create S3Client from environment variables."""
        bucket = os.environ.get('S3_BUCKET')
        if not bucket:
            raise ValueError("S3_BUCKET environment variable is required")

        return cls(
            bucket=bucket,
            region=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
            access_key=os.environ.get('AWS_ACCESS_KEY_ID'),
            secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            endpoint_url=os.environ.get('AWS_ENDPOINT_URL'),
        )

    def list_photos(self, prefix: str, extensions: set = None) -> List[dict]:
        """
        List photo objects under a prefix, sorted by timestamp parsed from filename.

        Returns list of dicts: [{'key': 's3-key', 'filename': 'name.jpg', 'timestamp': datetime}]
        """
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png'}

        photos = []
        paginator = self._client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                filename = key.rsplit('/', 1)[-1] if '/' in key else key
                ext = Path(filename).suffix.lower()

                if ext not in extensions:
                    continue

                stem = Path(filename).stem
                timestamp = None
                if len(stem) >= 14 and stem[:14].isdigit():
                    try:
                        timestamp = datetime.strptime(stem[:14], '%Y%m%d%H%M%S')
                    except ValueError:
                        pass

                photos.append({
                    'key': key,
                    'filename': filename,
                    'timestamp': timestamp,
                })

        photos.sort(key=lambda p: (p['timestamp'] or datetime.min, p['key']))
        logger.info(f"Listed {len(photos)} photos under prefix '{prefix}'")
        return photos

    _TRANSIENT_ERRORS = (EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError, ConnectionError, OSError)
    _NON_TRANSIENT_S3_CODES = frozenset({'NoSuchKey', 'AccessDenied', 'InvalidObjectState', 'NoSuchBucket'})

    def download_bytes(self, key: str, max_retries: int = 3) -> bytes:
        """Download an S3 object entirely into memory with retry and backoff."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.get_object(Bucket=self.bucket, Key=key)
                body = response['Body']
                try:
                    return body.read()
                finally:
                    body.close()
            except ClientError as e:
                code = e.response.get('Error', {}).get('Code', '')
                if code in self._NON_TRANSIENT_S3_CODES:
                    raise  # Don't retry permanent errors
                last_error = e
            except self._TRANSIENT_ERRORS as e:
                last_error = e
            except Exception:
                raise  # Unknown errors should not be retried
            if attempt < max_retries:
                delay = (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(f"S3 download failed for '{key}' (attempt {attempt}/{max_retries}): {last_error}. "
                               f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
        raise RuntimeError(f"S3 download failed for '{key}' after {max_retries} attempts: {last_error}")

    def upload_file(self, local_path: str, s3_key: str, content_type: str = None):
        """Upload a local file to S3."""
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        self._client.upload_file(
            local_path, self.bucket, s3_key,
            ExtraArgs=extra_args or None
        )
        logger.info(f"Uploaded {local_path} -> s3://{self.bucket}/{s3_key}")

    def exists(self, key: str) -> bool:
        """Check if an S3 key exists."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


def filter_photos_by_time(photos: List[dict], start_date: str, end_date: str,
                          start_hour: int, end_hour: int) -> List[dict]:
    """
    Filter photo list by date range and hour window.

    Args:
        photos: List of photo dicts from S3Client.list_photos()
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        start_hour: Start hour (0-23)
        end_hour: End hour (0-23)
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
        hour=start_hour, minute=0, second=0
    )
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=end_hour, minute=59, second=59
    )

    filtered = []
    for photo in photos:
        ts = photo.get('timestamp')
        if ts is None:
            continue
        if start_dt <= ts <= end_dt and start_hour <= ts.hour <= end_hour:
            filtered.append(photo)

    logger.info(f"Filtered {len(filtered)} photos from {len(photos)} total "
                f"({start_date} {start_hour}:00 to {end_date} {end_hour}:59)")
    return filtered
