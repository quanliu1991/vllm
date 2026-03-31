# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import _thread
import importlib
import inspect
import multiprocessing
import multiprocessing.forkserver as forkserver
import os
import re
import signal
import socket
import tempfile
import warnings
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvloop
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.datastructures import State
from starlette.routing import Mount

from vllm.entrypoints.serve.instrumentator.metrics import PrometheusResponse
from vllm.transformers_utils.runai_utils import is_runai_obj_uri
from vllm.transformers_utils.utils import is_s3

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.server_utils import (
    get_uvicorn_log_config,
    http_exception_handler,
    lifespan,
    log_response,
    validation_exception_handler,
)
from vllm.entrypoints.sagemaker.api_router import sagemaker_standards_bootstrap
from vllm.entrypoints.serve.elastic_ep.middleware import (
    ScalingMiddleware,
)
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.entrypoints.utils import (
    cli_env_setup,
    log_non_default_args,
    log_version_and_model,
    process_lora_modules,
)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tool_parsers import ToolParserManager
from vllm.tracing import instrument, init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import is_valid_ipv6_address
from vllm.utils.system_utils import decorate_logs, set_ulimit
from vllm.v1.metrics.prometheus import get_prometheus_registry
from vllm.version import __version__ as VLLM_VERSION
from vllm.entrypoints.openai.hb_serve.consul_client import ConsulList
from vllm.entrypoints.openai.hb_serve.hb_cli_args import make_hb_arg_parser
from vllm.entrypoints.openai.hb_serve.logger import ContextualLoggerAdapter, SingleLogger, to_serializable, \
    request_parse
from vllm.entrypoints.openai.hb_serve.security.encrypt import Encrypt
from vllm.entrypoints.openai.hb_serve import hot_swaps_settings, const
from vllm.entrypoints.openai.hb_serve.const import get_host_ip

TIMEOUT_KEEP_ALIVE = 5  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
# logger = init_logger("vllm.entrypoints.openai.api_server")
logger = SingleLogger.get_logger()
request_logger = ContextualLoggerAdapter(logger, {'sid': None, 'qid': None, 'aid': None, 'iid': None,
                                                     'rid': None, 'uid': None, 'category': None})


_FALLBACK_SUPPORTED_TASKS: tuple[SupportedTask, ...] = ("generate",)


@asynccontextmanager
async def build_async_engine_client(
    args: Namespace,
    consul_list=None,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool | None = None,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    with Encrypt(args, consul_list=consul_list) as encrypt:
        engine_args = AsyncEngineArgs.from_cli_args(encrypt.args)
        if client_config:
            engine_args._api_process_count = client_config.get("client_count", 1)
            engine_args._api_process_rank = client_config.get("client_index", 0)

        if disable_frontend_multiprocessing is None:
            disable_frontend_multiprocessing = bool(
                args.disable_frontend_multiprocessing)

        async with build_async_engine_client_from_engine_args(
                engine_args,
                usage_context=usage_context,
                disable_frontend_multiprocessing=disable_frontend_multiprocessing,
                client_config=client_config,
        ) as engine:
            yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool = False,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Create the EngineConfig (determines if we can use V1).
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if disable_frontend_multiprocessing:
        logger.warning("V1 is enabled, but got --disable-frontend-multiprocessing.")

    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm: AsyncLLM | None = None

    # Don't mutate the input client_config
    client_config = dict(client_config) if client_config else {}
    client_count = client_config.pop("client_count", 1)
    client_index = client_config.pop("client_index", 0)

    try:
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
            client_addresses=client_config,
            client_count=client_count,
            client_index=client_index,
        )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm
    finally:
        if async_llm:
            async_llm.shutdown()

def mount_metrics(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""

    registry = get_prometheus_registry()

    # `response_class=PrometheusResponse` is needed to return an HTTP response
    # with header "Content-Type: text/plain; version=0.0.4; charset=utf-8"
    # instead of the default "application/json" which is incorrect.
    # See https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/163#issue-1296092364
    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app).expose(app, response_class=PrometheusResponse)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)

def build_app(
    args: Namespace, supported_tasks: tuple["SupportedTask", ...] | None = None
) -> FastAPI:
    if supported_tasks is None:
        warnings.warn(
            "The 'supported_tasks' parameter was not provided to "
            "build_app and will be required in a future version. "
            "Defaulting to ('generate',).",
            DeprecationWarning,
            stacklevel=2,
        )
        supported_tasks = _FALLBACK_SUPPORTED_TASKS

    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan
        )
    elif args.enable_offline_docs:
        app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.state.args = args

    from vllm.entrypoints.serve import register_vllm_serve_api_routers

    register_vllm_serve_api_routers(app)

    from vllm.entrypoints.openai.models.api_router import (
        attach_router as register_models_api_router,
    )

    register_models_api_router(app)

    from vllm.entrypoints.sagemaker.api_router import (
        attach_router as register_sagemaker_api_router,
    )

    register_sagemaker_api_router(app, supported_tasks)

    if "generate" in supported_tasks:
        from vllm.entrypoints.openai.generate.api_router import (
            register_generate_api_routers,
        )

        register_generate_api_routers(app)

        from vllm.entrypoints.serve.disagg.api_router import (
            attach_router as attach_disagg_router,
        )

        attach_disagg_router(app)

        from vllm.entrypoints.serve.rlhf.api_router import (
            attach_router as attach_rlhf_router,
        )

        attach_rlhf_router(app)

        from vllm.entrypoints.serve.elastic_ep.api_router import (
            attach_router as elastic_ep_attach_router,
        )

        elastic_ep_attach_router(app)

    if "transcription" in supported_tasks:
        from vllm.entrypoints.openai.speech_to_text.api_router import (
            attach_router as register_speech_to_text_api_router,
        )

        register_speech_to_text_api_router(app)

    if "realtime" in supported_tasks:
        from vllm.entrypoints.openai.realtime.api_router import (
            attach_router as register_realtime_api_router,
        )

        register_realtime_api_router(app)

    if any(task in POOLING_TASKS for task in supported_tasks):
        from vllm.entrypoints.pooling import register_pooling_api_routers

        register_pooling_api_routers(app, supported_tasks)

    app.root_path = args.root_path
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="/health,/metrics",
            exclude_spans=["send", "receive"],
        )
    except ImportError:
        logger.debug(
            "opentelemetry-instrumentation-fastapi not installed; "
            "skipping FastAPI HTTP tracing"
        )
    mount_metrics(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)

    # Ensure --api-key option from CLI takes precedence over VLLM_API_KEY
    if tokens := [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]:
        from vllm.entrypoints.openai.server_utils import AuthenticationMiddleware

        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    # hb-serve log and replay APIs (used by request_logger/gradio UI)
    from fastapi.responses import JSONResponse, StreamingResponse
    from vllm.entrypoints.openai.hb_serve.request_logger.replay import (
        fetch_and_get_curl,
        fetch_and_get_response,
        fetch_and_run_curl,
        list_curl_or_response_objects,
    )

    @app.get("/log_content")
    async def get_log_content(object_name: str):
        if object_name.endswith(".json"):
            return {
                "type": "response",
                "content": fetch_and_get_response(object_name),
            }
        if object_name.endswith(".sh"):
            return {
                "type": "curl",
                "content": fetch_and_get_curl(object_name),
            }
        return {"type": "unknown", "content": ""}

    @app.get("/logs")
    async def get_logs(
        data: str,
        prefix: str = "",
        page: int = Query(1, ge=1),
        size: int = Query(10, le=100),
    ):
        try:
            offset = (page - 1) * size
            return {
                "curl": list_curl_or_response_objects(
                    data,
                    prefix,
                    limit=size,
                    offset=offset,
                    suffix="curl.sh",
                ),
                "response": list_curl_or_response_objects(
                    data,
                    prefix,
                    limit=size,
                    offset=offset,
                    suffix="json",
                ),
            }
        except Exception as e:
            return {"curl": str(e), "response": str(e)}

    @app.post("/replay")
    async def replay_log(
        object_name: str,
        host: str | None = Query(None),
        port: str | None = Query(None),
        stream: bool = Query(False),
    ):
        try:
            if stream:
                return StreamingResponse(
                    fetch_and_run_curl(object_name, host, port),
                    media_type="text/plain",
                )

            # Non-stream: return the whole replay output as JSON.
            chunks: list[str] = []
            for line in fetch_and_run_curl(object_name, host, port):
                chunks.append(line)
            return JSONResponse(content={"message": "".join(chunks)})
        except Exception as e:
            if stream:
                return StreamingResponse(iter([str(e)]), media_type="text/plain")
            return JSONResponse(content={"message": str(e)})

    @app.get("/manage_config")
    async def show_config(raw_request: Request):
        hb_config = to_serializable(raw_request.app.state.args)
        hb_config.pop("config_format", None)
        hb_config["use_priority"] = hot_swaps_settings.get_global_swaps(
            "USE_PRIORITY"
        )
        return JSONResponse(content=hb_config, status_code=200)

    if args.enable_request_id_headers:
        from vllm.entrypoints.openai.server_utils import XRequestIdMiddleware

        app.add_middleware(XRequestIdMiddleware)

    # Add scaling middleware to check for scaling state
    app.add_middleware(ScalingMiddleware)

    if envs.VLLM_DEBUG_LOG_API_SERVER_RESPONSE:
        logger.warning(
            "CAUTION: Enabling log response in the API Server. "
            "This can include sensitive information and should be "
            "avoided in production."
        )
        app.middleware("http")(log_response)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. Must be a function or a class."
            )

    app = sagemaker_standards_bootstrap(app)
    return app


async def init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
) -> None:
    vllm_config = engine_client.vllm_config
    if supported_tasks is None:
        warnings.warn(
            "The 'supported_tasks' parameter was not provided to "
            "init_app_state and will be required in a future version. "
            "Please pass 'supported_tasks' explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        supported_tasks = _FALLBACK_SUPPORTED_TASKS

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model) for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = vllm_config
    state.args = args
    if args.otlp_traces_endpoint:
        state.trace = init_tracer(
            __name__,
            args.otlp_traces_endpoint)
    resolved_chat_template = load_chat_template(args.chat_template)

    # Merge default_mm_loras into the static lora_modules
    default_mm_loras = (
        vllm_config.lora_config.default_mm_loras
        if vllm_config.lora_config is not None
        else {}
    )
    lora_modules = process_lora_modules(args.lora_modules, default_mm_loras)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        log_error_stack=args.log_error_stack,
    )

    if "generate" in supported_tasks:
        from vllm.entrypoints.openai.generate.api_router import init_generate_state

        await init_generate_state(
            engine_client, state, args, request_logger, supported_tasks
        )

    if "transcription" in supported_tasks:
        from vllm.entrypoints.openai.speech_to_text.api_router import (
            init_transcription_state,
        )

        init_transcription_state(
            engine_client, state, args, request_logger, supported_tasks
        )

    if "realtime" in supported_tasks:
        from vllm.entrypoints.openai.realtime.api_router import init_realtime_state

        init_realtime_state(engine_client, state, args, request_logger, supported_tasks)

    if any(task in POOLING_TASKS for task in supported_tasks):
        from vllm.entrypoints.pooling import init_pooling_state

        init_pooling_state(engine_client, state, args, request_logger, supported_tasks)

    # hb-serve metrics (used by custom hb_serve instrumentation).
    from vllm.entrypoints.openai.hb_serve.metrics import Metrics

    state.hb_serve_metrics = Metrics(["model_name", "reasoning"])
    model_name = (
        args.served_model_name[0]
        if isinstance(args.served_model_name, list)
        else args.served_model_name
        if args.served_model_name is not None
        else args.model
    )
    state.hb_serve_metrics.init_counter(
        dict(model_name=model_name, reasoning="False")
    )

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


def create_server_socket(addr: tuple[str, int]) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)

    return sock


def create_server_unix_socket(path: str) -> socket.socket:
    sock = socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM)
    sock.bind(path)
    return sock


def validate_api_server_args(args):
    valid_tool_parses = ToolParserManager.list_registered()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parsers = ReasoningParserManager.list_registered()
    if (
        reasoning_parser := args.structured_outputs_config.reasoning_parser
    ) and reasoning_parser not in valid_reasoning_parsers:
        raise KeyError(
            f"invalid reasoning parser: {reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parsers)} }})"
        )


@instrument(span_name="API server setup")
def setup_server(args):
    """Validate API server args, set up signal handler, create socket
    ready to serve."""

    log_version_and_model(logger, VLLM_VERSION, args.model)
    log_non_default_args(args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    if args.uds:
        listen_address = f"unix:{args.uds}"
    else:
        addr, port = sock_addr
        is_ssl = args.ssl_keyfile and args.ssl_certfile
        host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr or "0.0.0.0"
        listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"
    return listen_address, sock


async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("APIServer")

    listen_address, sock = setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def run_server_worker(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    # Get uvicorn log config (from file or with endpoint filter)
    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config
    consul_list = ConsulList(args)
    async with build_async_engine_client(
        args,
        client_config=client_config,
        consul_list=consul_list
    ) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        logger.info("Supported tasks: %s", supported_tasks)

        app = build_app(args, supported_tasks)
        await init_app_state(engine_client, app.state, args, supported_tasks)

        logger.info(
            "Starting vLLM API server %d on %s",
            engine_client.vllm_config.parallel_config._api_process_rank,
            listen_address,
        )
        allow_consul = args.allow_consul == 'true'
        if allow_consul:
            _thread.start_new_thread(hot_swaps_settings.check_hot_swaps_serve, ())

        consul_list.update_consul_list("2")
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=args.ssl_ciphers,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    parser = make_hb_arg_parser(parser)
    args = parser.parse_args()
    if args.host in ['0.0.0.0', 'localhost']:
        args.host = get_host_ip()
    validate_parsed_serve_args(args)

    if is_runai_obj_uri(args.model) or is_s3(args.lora_models if args.lora_models else ""):
        args.load_format = "runai_streamer"

    if is_runai_obj_uri(args.model) or is_s3(args.lora_models if args.lora_models else ""):
        os.environ["AWS_REGION"] = "us-east-1"
        os.environ["RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING"] = "0"
        os.environ["RUNAI_STREAMER_NO_BOTO3_SESSION"] = "16"

        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["MINIO_ACCESS_KEY"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["MINIO_SECRET_KEY"]
        os.environ["AWS_ENDPOINT_URL"] = "http://" + os.environ["MINIO_ENDPOINT"]

        if is_s3(args.lora_models if args.lora_models else ""):
            os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"
            os.environ["VLLM_PLUGINS"] = '["s3_adapter_resolver"]'
            os.environ["VLLM_LORA_RESOLVER_CACHE_DIR"] = "/workspace/adapters"
            os.environ["S3_PATH"] = args.lora_models

    uvloop.run(run_server(args))
