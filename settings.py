from dotenv import load_dotenv
import os

load_dotenv(".config.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def read_bool(env_var):
    value = os.getenv(env_var)
    if value is not None:
        return value.lower() in ("true", "1", "yes")
    return False

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_SECURE=read_bool("MINIO_SECURE")
