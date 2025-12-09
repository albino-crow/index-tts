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
MINIO_KEY_ID = os.getenv("MINIO_KEY_ID")
MINIO_SECRET_ACCESS_KEY = os.getenv("MINIO_SECRET_ACCESS_KEY")
MINIO_SECURE = read_bool("MINIO_SECURE")
USE_FLOATING_POINT_16 = read_bool("USE_FLOATING_POINT_16")
