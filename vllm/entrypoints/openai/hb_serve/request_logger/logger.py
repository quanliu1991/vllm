
import json
import time

from .config import minio_client, MINIO_BUCKET, check_minio_bucket_exists, MinioObjectTags
import asyncio
import io
from datetime import datetime
from minio import Minio
from minio.tagging import Tags

from ..hot_swaps_settings import get_global_swaps
from ..security.log_security import encrypt_for_java
from vllm.entrypoints.openai.hb_serve.logger import SingleLogger, ContextualLoggerAdapter

logger = SingleLogger.get_logger()
minio_logger = ContextualLoggerAdapter(logger, {'sid': None, 'qid': None, 'aid': None, 'iid': None,
                                                     'rid': None, 'uid': None, 'category': None})

class StreamingUploader:
    def __init__(self, minio_client: Minio, bucket: str, request_id: str, tid: str):
        self.client = minio_client
        self.bucket = bucket
        self.buffer = ""
        self.lock = asyncio.Lock()
        self.now = datetime.utcnow().strftime("%Y-%m-%d")
        self.tid = tid
        self.request_id = request_id
        prefix = f"{self.now}/{self.tid}/{self.request_id}"
        self.object_name = f"{prefix}_response.json"
        self.step_number = 0


    async def append_and_upload(self, chunk: str, upload_step: int=50):
        if minio_client:
            self.buffer += chunk
            self.step_number += 1
            json_str = chunk.removeprefix("data: ").strip()

            if self.step_number % upload_step == 0:
                await self._upload(self.object_name)
            elif json_str.startswith("[DONE]"):
                await self._upload(self.object_name)

            try:
                data = json.loads(json_str)
                finish_reason = data["choices"][0]['finish_reason']
            except Exception:
                finish_reason = None
            if finish_reason == "length":
                tags = MinioObjectTags.exceed_max_tokens.value
                bucket = f'{self.bucket}-{tags_to_string(tags)}'
                await self._upload(self.object_name, bucket=bucket, tags=tags)

    async def _upload(self, object_name, bucket: str=None, tags: Tags = None):
        loop = asyncio.get_running_loop()
        content_bytes = self.buffer.encode("utf-8")
        if bucket:
            bucket_name = bucket
        else:
            bucket_name = self.bucket
        check_minio_bucket_exists(self.client, bucket_name)
        # 用 run_in_executor 避免阻塞
        loop.run_in_executor(
            None,
            lambda: self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=io.BytesIO(content_bytes),
                length=len(content_bytes),
                content_type="text/plain",
                tags=tags
            )
        )

def build_curl_command(request, body):
    url = str(request.url)
    headers = request.headers
    token = headers.get("authorization", "Bearer xxx")
    ssl = "-k" if url.startswith("https") else ""

    data = json.dumps(body, ensure_ascii=False)
    if get_global_swaps("LOG_PROMPT_ENCRYPT") == "true":
        data = encrypt_for_java(data)
    curl_lines = [
        f"curl {ssl} -s -X POST '{url}'",
        f"  -H 'Authorization: {token}'",
        f"  -H 'Content-Type: application/json'",
        f"  -d '{data}'"
    ]
    return " \\\n".join(curl_lines)

def tags_to_string(tags: Tags) -> str:
    if not tags:
        return ""
    return ", ".join(f"{key}-{value}" for key, value in tags.items())

def save_request_logs_to_minio(request, body: dict, extra_dict: dict, tags: Tags = None):
    try:
        if minio_client:
            now = datetime.utcnow().strftime("%Y-%m-%d")
            unique_id = extra_dict.get("rid")
            tid = extra_dict.get("tid","tid_none")
            prefix = f"{now}/{tid}/{unique_id}"
            # Save curl
            curl_data = build_curl_command(request, body)
            if get_global_swaps("LOG_PROMPT_ENCRYPT") == "true":
                object_name = f"{prefix}_encrypt_curl.sh"
            else:
                object_name = f"{prefix}_curl.sh"

            def put_object(data, obj_name, bucket_name):
                stream = io.BytesIO(data.encode("utf-8"))
                return minio_client.put_object(
                    bucket_name, obj_name, stream, length=stream.getbuffer().nbytes,
                    tags=tags
                )
            put_object(curl_data, object_name, MINIO_BUCKET)
            if tags:
                new_bucket_name = f"{MINIO_BUCKET}-{tags_to_string(tags)}"
                check_minio_bucket_exists(minio_client, new_bucket_name)
                put_object(curl_data, object_name, new_bucket_name)
    except Exception as e:
        minio_logger.warning(e)



def save_response_logs_to_minio(response: dict, extra_dict, tags: Tags = None):
    try:
        if minio_client:
            now = datetime.utcnow().strftime("%Y-%m-%d")
            unique_id = extra_dict.get("rid", "rid_none")
            tid = extra_dict.get("tid", "tid_none")
            prefix = f"{now}/{tid}/{unique_id}"

            # Save response
            def put_object(data, bucket_name):
                stream = io.BytesIO(json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))
                return minio_client.put_object(
                    bucket_name, f"{prefix}_response.json", stream, length=stream.getbuffer().nbytes,
                    tags=tags
                )

            put_object(response, MINIO_BUCKET)
            if tags:
                new_bucket_name = f"{MINIO_BUCKET}-{tags_to_string(tags)}"
                check_minio_bucket_exists(minio_client, new_bucket_name)
                put_object(response, new_bucket_name)
    except Exception as e:
        minio_logger.warning(e)


def save_guided_logs_to_minio(guided: str, guided_hash: str, mode: str):
    try:
        if minio_client:
            guided_bucket_name = f"guided"
            check_minio_bucket_exists(minio_client, guided_bucket_name)
            guide_dict = {f"guided_{mode}": guided}
            guided = io.BytesIO(json.dumps(guide_dict, indent=2, ensure_ascii=False).encode("utf-8"))
            minio_client.put_object(
                    guided_bucket_name, f"{guided_hash}.txt", guided, length=guided.getbuffer().nbytes
                )
    except Exception as e:
        minio_logger.warning(e)