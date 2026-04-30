#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import socket
import time
from http import HTTPStatus
from typing import List, Dict
from enum import Enum
import shortuuid
from fastapi import responses
import uvicorn

from async_lru import alru_cache
from fastapi import FastAPI
from starlette.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch_npu
import requests
import json
import random
import string
import ssl

app = FastAPI()

device = None
model = None
tokenizer = None

SERVICE_NAME = 'hb-serve-embedding'

INTERNAL_SERVER_ERROR = (500, 'Internal Server Error', 'Server got itself in trouble')
BAD_EMBED_MODEL = 'hb-serve.0071'
BAD_REQUEST = 'hb-serve.0000'



class DataInfo(BaseModel):
    input: list = []


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: str = None
    code: str = None


class EmbedData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list


class EmbedResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"embedding-{shortuuid.random()}")
    object: List[str] = "list"
    data: List[EmbedData]
    model: str

class SimilarityRequest(BaseModel):
    strings: List[str]
    query: str
    top_k: int = 3
    model: str

class SimilarityResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"similarity-{shortuuid.random()}")
    object: str = "dict"
    data: Dict[str, float]
    model: str


def list_to_tuple(function):
    def wrapper(*args):
        args = [tuple(x) if isinstance(x, list) else x for x in args]
        result = function(*args)
        result = tuple(result) if isinstance(result, list) else result
        return result

    return wrapper


async def run_emb_batch(inputs):
    encoded_input = tokenizer(inputs, padding='max_length', return_tensors='pt', max_length=512, truncation=True)
    encoded_input = encoded_input.to(device)
    emb = model(**encoded_input)
    emb = emb[0][:, 0]
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.tolist()


@list_to_tuple
@alru_cache(maxsize=2048)
async def process_text_with_semaphore(texts):
    emb_list = []
    batch_size = args.batch_size
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_index in range(num_batches):
        batch_texts = texts[batch_index * batch_size: (batch_index + 1) * batch_size]
        inputs = [text[:512].strip() for text in batch_texts]
        emb_batch = await run_emb_batch(inputs)
        emb_list = emb_list + emb_batch

    return emb_list

@app.get("/test_bandwidth")
async def test_bandwidth():
    size_in_mb = 50  # 生成 50 MB 的测试数据
    data = b"0" * (size_in_mb * 1024 * 1024)
    start_time = time.time()
    return StreamingResponse(iter([data]), media_type="application/octet-stream")


@app.post("/v1/embeddings", response_model=EmbedResponse, response_class=responses.ORJSONResponse)
async def completions(datainfo: DataInfo):
    try:
        texts = datainfo.input
        if not texts:
            return JSONResponse({"message": "Input is required!", "type": "invalid_request_error", "code": 400},
                                status_code=400)

        emb_list = await process_text_with_semaphore(texts)

        def create_embed_response_json(vectors):
            data_list = []
            for i, vector in enumerate(vectors):
                data = EmbedData(index=i, embedding=vector)
                data_list.append(data)
            embed_rsp = EmbedResponse(data=data_list, model="embedding_v1")
            return embed_rsp

        res = create_embed_response_json(emb_list)
        return res

    except Exception as e:
        return JSONResponse(
            {"message": str(e), "type": "invalid_request_error", "code": 500},
            status_code=500
        )


class HBServeStatus(Enum):
    BAD_REQUEST = 'hb-serve.0000'
    BAD_MODEL = 'hb-serve.0010'
    BAD_MODEL_INIT = 'hb-serve.0011'
    BAD_MODEL_FOR_LORA = 'hb-serve.0012'
    BAD_TOKEN_LENGTH = 'hb-serve.0020'
    BAD_INPUT_PARAMS = 'hb-serve.0030'
    BAD_MESSAGES = 'hb-serve.0031'
    BAD_LORA_TYPE = 'hb-serve.0032'
    BAD_PREFIX_PADDING = 'hb-serve.0033'
    BAD_GUIDED = 'hb-serve.0034'
    BAD_CONNECTION = 'hb-serve.0050'
    BAD_DECRYPT = 'hb-serve.0060'
    BAD_TIMEOUT = 'hb-serve.0070'
    BAD_EMBED_MODEL = 'hb-serve.0071'
    BAD_RERANK_MODEL = 'hb-serve.0072'
    BAD_SIMILARITY_MODEL = 'hb-serve.0073'
    BAD_HEADER = 'hb-serve.0080'


def create_error_response(status_code: HTTPStatus,
                          message: str,
                          code=HBServeStatus.BAD_REQUEST) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error", code=code.value).model_dump(),
                        status_code=status_code.value)


@app.post("/v1/similarity", response_model=SimilarityResponse,
          response_class=responses.ORJSONResponse)
async def similarity(request: SimilarityRequest):

    def remove_duplicates(strings) -> list:
        unique_strings = list(set(strings))
        return unique_strings

    def create_similarity_response_json(result):
        similarity_rsp = SimilarityResponse(data=result, model=request.model)
        return similarity_rsp

    if not request.query or not request.strings:
        rsp = create_similarity_response_json({})
       
        return rsp

    try:
        strings = remove_duplicates(request.strings)
        all_strings = strings + [request.query]

        embeddings = await process_text_with_semaphore(all_strings)

        string_embeddings = embeddings[:-1]
        query_embedding = embeddings[-1]

        # Use torch cosine_similarity instead of sklearn
        cosine_similarities = F.cosine_similarity(
            torch.tensor([query_embedding]),
            torch.tensor(string_embeddings),
            dim=1
        ).numpy()

        top_k_indices = cosine_similarities.argsort()[-request.top_k:][::-1]

        result = {strings[idx]: float(cosine_similarities[idx]) for idx in top_k_indices}

        rsp = create_similarity_response_json(result)

        return rsp

    except Exception as e:

        return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(e), HBServeStatus.BAD_SIMILARITY_MODEL)





def register_service_to_consul(host, port, device):
    consul_host = os.getenv("CONSUL_HOST")
    consul_port = os.getenv("CONSUL_PORT")
    consul_token = os.getenv("CONSUL_TOKEN")

    service_id = f"{SERVICE_NAME}_{host}_{port}"

    url = f"http://{consul_host}:{consul_port}/v1/agent/service/register"
    headers = {
        "X-Consul-Token": consul_token,
        "Content-Type": "application/json"
    }
    service_data = {
        "ID": service_id,
        "Name": SERVICE_NAME,
        "Address": host,
        "Port": port,
        "Meta": {
            "Device": device,
            "model_name": "Embedding-v1",
            "status": "2",
            "version": "v2.1.0"
        },
        "Tags": [SERVICE_NAME, f"{device}"]
    }

    response = requests.put(url, headers=headers, data=json.dumps(service_data))
    if response.status_code == 200:
        print(f"Successfully registered service {SERVICE_NAME} on {device} to Consul.")
    else:
        print(f"Failed to register service {SERVICE_NAME} on {device}. Status Code: {response.status_code}, "
              f"Response: {response.text}")


def launch_model(args):
    global device, model, tokenizer
    device = os.getenv("EMBEDDING_DEVICE", "cpu").lower()
    if device != "cpu":
        device = f"npu:0"
        torch_npu.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)

    host = get_host_ip() if args.host in ['0.0.0.0', 'localhost'] else args.host
    port = args.port
    model_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    if os.getenv("ALLOW_CONSUL","false") == "true":
        register_service_to_consul(host, port, device)

    cert_file = os.getenv("SSL_CERTIFILE", "/home/models/configs/certificate.crt")
    key_file = os.getenv("SSL_KEYFILE", "/home/models/configs/private.key")

    if not os.path.isfile(cert_file) or not os.path.isfile(key_file):
        cert_file = None
        key_file = None
        print("http run")


    uvicorn.run(app=app, host=host, port=port,
                ssl_keyfile=key_file,
                ssl_certfile=cert_file,
                ssl_cert_reqs=int(ssl.CERT_NONE),
                ssl_ca_certs=None)


def get_host_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    return ip


if __name__ == '__main__':
    parser = ArgumentParser(usage='The bge model API server')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='host')
    parser.add_argument('--port', type=int, default=40000, help='port for the bge model on npu-0')
    parser.add_argument('--batch-size', type=int, default=16, help='batch_size')
    parser.add_argument('--model', type=str, default='/home/models/bge-large-zh-v1.5', help='model path')
    args = parser.parse_args()
    launch_model(args)
