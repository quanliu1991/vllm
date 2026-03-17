from __future__ import unicode_literals

import logging
import os
import json
import datetime
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Dict, Any
from collections.abc import Iterable
from vllm.entrypoints.openai.hb_serve.const import SERVE_CHAT, SERVE_NAME
from vllm.entrypoints.openai.hb_serve.security.log_security import encrypt_for_java


class LogType(Enum):
    RESTFUL = 'restful'  # 接口日志
    SYSTEM = 'system'  # 系统日志
    SESSION = 'session'  # 会话日志
    DATABASE = 'database'  # 数据库日志
    SERVICE_DISCOVERY = 'service_discovery'  # 服务发现日志


# 状态
class Statu(Enum):
    SUCCESS = '0'
    FAILED = '1'


class Mode(Enum):
    FILE = 'file'  # 文件输出
    CONSOLE = 'console'  # 控制台输出
    CONSOLE_AND_FILE = 'console_and_file'  # 控制台和日志文件同时输出


# 应用名称配置
APPLICATION_NAME = SERVE_NAME

# 模块名称配置
MODULE = SERVE_CHAT

# 日志输出方式配置
LOG_MODE = Mode.CONSOLE_AND_FILE

# 日志配置
LOGGING_PATH = "./logs"
LOGGING_MAX_BYTES = 20 * 1024 * 1024
LOGGING_BACKUP_COUNT = 10
LOGGING_RETENTION_DAYS = 30


def enum_to_json(enum_instance):
    try:
        return json.dumps(enum_instance)
    except TypeError:
        return json.dumps(enum_instance.value)


def get_logger_level(level):
    if level == "debug":
        return logging.DEBUG
    if level == "error":
        return logging.ERROR
    return logging.INFO


def request_parse(request, raw_request):
    default = 'null'

    if raw_request:
        x_tid = raw_request.headers.get('x-tid', default)
        ext = raw_request.headers.get('ext', default)
        order = raw_request.headers.get('ORDER', 'routine')
        ext_dict = dict(
            item.split(":", 1) for item in ext.strip(";").split(";") if ":" in item
        )
        is_debug_from_request = int(ext_dict.get("debug", "0"))
    else:
        x_tid, is_debug_from_request, ext, order = default, 0, default, 'routine'
    query = request.messages if hasattr(request, 'messages') else default

    parse_result = {'tid': x_tid,
                    'is_debug_from_request':is_debug_from_request,
                    'ext':ext,
                    'category': enum_to_json(LogType.RESTFUL),
                    "order":order}
    return parse_result, query


if not os.path.exists(LOGGING_PATH):
    os.makedirs(LOGGING_PATH)


class JSONFormatter(logging.Formatter):
    def format(self, record):
        shanghai_tz = timezone(timedelta(hours=8))
        local_time = datetime.now(shanghai_tz)
        log_record = {
            "timestamp": local_time.strftime('%Y-%m-%dT%H:%M:%S.%f%z'),
            "application": APPLICATION_NAME,
            "module": MODULE,
            "level": record.levelname,
            "tid": record.tid if hasattr(record, 'tid') else "null",
            "rid": record.rid if hasattr(record, 'rid') else "null",
            "file": f"{record.filename}:{record.lineno}",
            "msg": record.getMessage(),
            "category": record.category  if hasattr(record, 'category') else "null",
            "ext": record.ext  if hasattr(record, 'ext') else "null"
        }

        return json.dumps(log_record, ensure_ascii=False)


class MyRotatingFileHandler(RotatingFileHandler):
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        super(MyRotatingFileHandler, self).__init__(filename, mode, maxBytes, backupCount, encoding, delay)

    def doRollover(self):
        """
        Override doRollover to add timestamp to the filename and handle retention policy.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        current_time = int(time.time())
        ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime(current_time))
        dfn = f"{self.baseFilename}.{ts}"
        if not os.path.exists(dfn) and os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)

        super().doRollover()

        log_files = sorted(
            [f for f in os.listdir(LOGGING_PATH) if f.startswith(APPLICATION_NAME)],
            key=lambda f: os.path.getmtime(os.path.join(LOGGING_PATH, f))
        )

        cutoff_time = time.time() - LOGGING_RETENTION_DAYS * 24 * 60 * 60

        for log_file in log_files:
            log_file_path = os.path.join(LOGGING_PATH, log_file)
            if os.path.getmtime(log_file_path) < cutoff_time or len(log_files) > self.backupCount:
                os.remove(log_file_path)
                log_files.remove(log_file)


class SingleLogger(object):
    __instance = None

    def __init__(self):
        pass

    def __new__(cls, *args, **kwd):
        if SingleLogger.__instance is None:
            SingleLogger.__instance = object.__new__(cls, *args, **kwd)
            SingleLogger.__instance._setup_logger()
        return SingleLogger.__instance

    def _setup_logger(self):
        self.__logger = logging.getLogger(__name__)

        self.__logger.propagate = False  # 禁用传播到父logger

        level = get_logger_level(os.getenv('LOGGING_LEVEL', "debug"))
        self.__logger.setLevel(level)
        formatter = JSONFormatter()

        log_file = os.path.join(LOGGING_PATH, f"{APPLICATION_NAME}.log")
        file_handler = MyRotatingFileHandler(filename=log_file,
                                             maxBytes=LOGGING_MAX_BYTES,
                                             backupCount=LOGGING_BACKUP_COUNT,
                                             encoding='utf-8',
                                             delay=False)
        file_handler.addFilter(logging.Filter(name=__name__))
        file_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.addFilter(logging.Filter(name=__name__))
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(console_handler)

    @staticmethod
    def get_logger():
        if not SingleLogger.__instance:
            SingleLogger()  # 确保实例被创建
        return SingleLogger.__instance.__logger

    @staticmethod
    def set_log_level(level):
        if SingleLogger.__instance:
            # debug/info/warn/error
            if level == "debug":
                SingleLogger.__instance.__logger.setLevel(logging.DEBUG)
            elif level == "info":
                SingleLogger.__instance.__logger.setLevel(logging.INFO)
            elif level == "warn":
                SingleLogger.__instance.__logger.setLevel(logging.WARNING)
            elif level == "error":
                SingleLogger.__instance.__logger.setLevel(logging.ERROR)
            else:
                SingleLogger.__instance.__logger.setLevel(logging.DEBUG)


    def info(self, message):
        SingleLogger.get_logger().info(message)

    def error(self, message):
        SingleLogger.get_logger().error(message)

    def warning(self, message):
        SingleLogger.get_logger().warning(message)

    def debug(self, message):
        SingleLogger.get_logger().debug(message)


class ContextualLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = self.extra.copy()
        if 'extra' in kwargs:
            extra.update(kwargs['extra'])
        kwargs['extra'] = extra
        return msg, kwargs


def request_logs(payload: Dict[str, Any], log_messages=False) -> str:
    from vllm.entrypoints.openai.hb_serve.hot_swaps_settings import get_global_swaps
    """
    将请求参数中 messages 的 content 字段替换为字符长度，
    并返回格式化后的 JSON 字符串。

    :param payload: 原始请求参数字典
    :return: JSON 字符串，已替换 content 内容
    """
    payload.update({"prompt_encrypt": "false"})
    def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """递归地删除所有值为 None 或 False 的字段"""
        return {
            k: clean_dict(v) if isinstance(v, dict) else v
            for k, v in d.items()
            if v not in (None, False)
        }
    if log_messages:
        if get_global_swaps("LOG_PROMPT_ENCRYPT") == "true":
            modified_payload = json.loads(json.dumps(payload))
            for msg in modified_payload.get("messages", []):
                if ("content" in msg and isinstance(msg["content"], str)
                        and msg.get("role","") == "system"):
                    msg["content"] = encrypt_for_java(msg["content"])
            modified_payload.update({"prompt_encrypt": "true"})
        else:
            modified_payload = json.loads(json.dumps(payload))

        return json.dumps(clean_dict(modified_payload), ensure_ascii=False)
    else:
        # 拷贝原始 payload，避免修改原始数据
        modified_payload = json.loads(json.dumps(payload))

        for msg in modified_payload.get("messages", []):
            if "content" in msg and isinstance(msg["content"], str):
                msg["content"] = len(msg["content"])
        # cleaned_payload = clean_dict(modified_payload)
        return json.dumps(modified_payload, ensure_ascii=False)

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return to_serializable(vars(obj))
    elif hasattr(obj, "_asdict"):  # for namedtuples
        return to_serializable(obj._asdict())
    elif hasattr(obj, "model_dump"):
        return to_serializable(obj.model_dump())
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return [to_serializable(t) for t in obj]
    else:
        return obj
