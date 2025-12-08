from minio import Minio
from settings import MINIO_ENDPOINT, MINIO_SECRET_ACCESS_KEY, MINIO_KEY_ID, MINIO_SECURE


import boto3


class S3StorageHandler:
    def __init__(self):
        # Ensure endpoint has protocol
        endpoint = MINIO_ENDPOINT
        if endpoint and not endpoint.startswith(("http://", "https://")):
            # Use https if MINIO_SECURE is True, otherwise http
            protocol = "https://" if MINIO_SECURE else "http://"
            endpoint = protocol + endpoint
        
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=MINIO_KEY_ID,
            aws_secret_access_key=MINIO_SECRET_ACCESS_KEY,
        )

    def parse_s3_url(self, url: str):
        if url.startswith("s3://"):
            # S3 URI format: s3://bucket/key/path
            path = url[len("s3://") :]
            parts = path.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        elif url.startswith(("http://", "https://")):
            # HTTP URL format: http://host:port/bucket/key/path
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path_parts = parsed.path.lstrip("/").split("/", 1)
            bucket = path_parts[0] if len(path_parts) > 0 else ""
            key = path_parts[1] if len(path_parts) > 1 else ""
        else:
            raise ValueError(f"Invalid S3 URL format: {url}. Expected s3://, http://, or https://")

        return bucket, key

    def download(self, url: str, filename: str):
        bucket, key = self.parse_s3_url(url)
        print(bucket)
        print(key)
        print(filename)
        self.s3_client.download_file(bucket, key, filename)

    def upload(self, filename: str, url: str):
        bucket, key = self.parse_s3_url(url)
        out = self.s3_client.upload_file(filename, bucket, key)


def get_minio_client() -> S3StorageHandler:
    return S3StorageHandler()
