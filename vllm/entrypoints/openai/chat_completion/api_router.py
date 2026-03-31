# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.hb_serve.hot_swaps_settings import get_global_swaps
from vllm.entrypoints.openai.hb_serve.logger import request_parse
from vllm.entrypoints.openai.hb_serve.request_logger.config import (
    MINIO_BUCKET,
    minio_client,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.entrypoints.openai.hb_serve.request_logger.logger import StreamingUploader

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


async def upload_stream_chunks_to_minio(
    gen,
    uploader: StreamingUploader,
):
    async for chunk in gen:
        await uploader.append_and_upload(chunk)
        yield chunk


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Chat Completions API"
        )

    # Align with v0.10.1 HBServe: stable id for tracing / MinIO (includes model).
    if request.model:
        request.request_id = (
            f"chatcmpl-{handler._base_request_id(raw_request, request.request_id)}"
            f"--{request.model}"
        )

    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except Exception as e:
        generator = handler.create_error_response(e)

    hb_serve_metrics = getattr(raw_request.app.state, "hb_serve_metrics", None)
    reasoning_label = str(getattr(request, "reasoning", False))
    labels = dict(model_name=request.model, reasoning=reasoning_label)

    if isinstance(generator, ErrorResponse):
        if hb_serve_metrics is not None:
            hb_serve_metrics.log(
                generator.error.hb_code,
                1,
                labels=labels,
            )
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        if hb_serve_metrics is not None:
            hb_serve_metrics.log(labels=labels)
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    # MinIO streaming upload: align with v0.10.1 behavior by chunking in
    # StreamingUploader (default upload_step=50).
    if get_global_swaps("MINIO_ENABLE_LOG") == "true":
        upload_enabled = True
        is_debug_from_request = False
    else:
        upload_enabled = False
        extra_dict, _ = request_parse(request, raw_request)
        is_debug_from_request = extra_dict.get("is_debug_from_request", False)
        upload_enabled = bool(is_debug_from_request)

    if upload_enabled:
        # Serving layer sets this request_metadata on raw_request.state.
        request_id = getattr(raw_request.state.request_metadata, "request_id", None)
        if request_id is None:
            request_id = request.request_id
        extra_dict, _ = request_parse(request, raw_request)
        tid = extra_dict.get("tid", "tid_none")
        uploader = StreamingUploader(minio_client, MINIO_BUCKET, request_id, tid)
        generator = upload_stream_chunks_to_minio(generator, uploader)

    if hb_serve_metrics is not None:
        hb_serve_metrics.log(labels=labels)

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions/render",
    dependencies=[Depends(validate_json_request)],
    response_model=list,
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def render_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Render chat completion request and return conversation and engine
    prompts without generating."""
    handler = chat(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        return base_server.create_error_response(
            message="The model does not support Chat Completions API"
        )

    try:
        result = await handler.render_chat_request(request)
    except Exception as e:
        result = handler.create_error_response(e)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result)


def attach_router(app: FastAPI):
    app.include_router(router)
