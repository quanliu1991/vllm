import os
import socket
from enum import Enum

import requests
from minio import Minio, S3Error, InvalidResponseError
from minio.tagging import Tags
from urllib3.exceptions import NewConnectionError, MaxRetryError

from vllm.entrypoints.openai.hb_serve.logger import SingleLogger, ContextualLoggerAdapter

logger = SingleLogger.get_logger()
minio_logger = ContextualLoggerAdapter(logger, {'sid': None, 'qid': None, 'aid': None, 'iid': None,
                                                     'rid': None, 'uid': None, 'category': None})
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "10.20.152.74:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "tYqV9s#bt3ADrMy")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "hb-serve-chat-logs")

os.environ["MINIO_ENABLE_LOG"] = MINIO_ENABLE_LOG = os.getenv("MINIO_ENABLE_LOG", "true")

try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    minio_client.list_buckets()
except (NewConnectionError, MaxRetryError, requests.exceptions.ConnectionError, socket.error) as e:
    minio_client = None
    minio_logger.warning(f"MinIO 无法连接（连接被拒绝/网络不可达）：{e}")
except S3Error as e:
    minio_client = None
    minio_logger.warning(f"MinIO 返回 S3 协议错误（如鉴权失败等）: {e}")
except Exception as e:
    minio_client = None
    minio_logger.warning(f"MinIO init except: {e}")

tags = Tags.new_object_tags()
tags["abnormal"] = "exceeding-max-tokens"
# Ensure bucket exists
class MinioObjectTags(Enum):
    exceed_max_tokens = tags


def check_minio_bucket_exists(client: Minio, name: str):
    try:
        if client and not client.bucket_exists(name):
            client.make_bucket(name)
    except Exception as e:
        minio_logger.warning(e)

check_minio_bucket_exists(minio_client, MINIO_BUCKET)
