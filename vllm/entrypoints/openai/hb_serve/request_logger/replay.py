import json
import subprocess
import tempfile

from vllm.entrypoints.openai.hb_serve.request_logger.config import minio_client, MINIO_BUCKET



def list_curl_or_response_objects(data: str, prefix: str = "", limit: int = 10, offset: int = 0,
                                  suffix: str = "curl.sh"):
    objects = minio_client.list_objects(MINIO_BUCKET, prefix=data + "/" + prefix, recursive=True)
    files = sorted(
        [obj.object_name for obj in objects if obj.object_name.endswith(suffix)],
        reverse=True
    )
    return files[offset:offset + limit]


def fetch_and_get_response(object_name: str):
    response = minio_client.get_object(MINIO_BUCKET, object_name)
    response = response.read().decode("utf-8")
    return response


def fetch_and_get_curl(object_name: str):
    curl = minio_client.get_object(MINIO_BUCKET, object_name)
    curl = curl.read().decode("utf-8")
    return curl


def fetch_and_run_curl(object_name: str, host: str = None, port: str = None):
    response = minio_client.get_object(MINIO_BUCKET, object_name)
    curl_script = response.read().decode("utf-8")

    if host:
        curl_script = curl_script.replace("http://localhost:8000", f"http://{host}:{port or 80}")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sh") as tmp_file:
        tmp_file.write(curl_script)
        tmp_file.flush()
        process = subprocess.Popen(
            ["bash", tmp_file.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in iter(process.stdout.readline, ""):
            yield line
        process.stdout.close()
        process.wait()


if __name__ == '__main__':

    objects = list_curl_or_response_objects(data="2025-05-12", prefix="chatcmpl-ba1048b13f", suffix="json")
    for obj in objects:
        response = fetch_and_get_response(obj)
        print(response)
    objects = list_curl_or_response_objects(data="2025-05-12", prefix="chatcmpl-ba1048b13f", suffix="sh")
    for obj in objects:
        curl = fetch_and_get_curl(obj)
        print(curl)
        result = fetch_and_run_curl(obj)
        print(result)
