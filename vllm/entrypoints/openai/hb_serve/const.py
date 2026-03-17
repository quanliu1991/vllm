import os
import shortuuid
import socket
from enum import Enum


def get_host_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    return ip


def get_lora_model_dict(path):
    if not path:
        return None
    entries = os.listdir(path)
    entries = [entry for entry in entries if entry.endswith('_lora')]
    return {entry[:-len('_lora')]: os.path.join(path, entry) for entry in entries}


SERVE_NAME = 'hb-serve'
SERVE_CHAT = 'hb-serve-chat'


TIMEOUT_KEEP_ALIVE = 5  # seconds
TIMEOUT_GRACEFUL_SHUTDOWN = 120  # seconds


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
    BAD_GUIDED_TIMEOUT = 'hb-serve.0071'
    BAD_HEADER = 'hb-serve.0080'
    BAD_SERVER = 'hb-serve.0090'


CONFIG_KEY = "configs/prod.yaml"
CONSUL_HOT_SWAPS_CONFIG_FILE = "/workspace/tmp/hot_swaps.yaml"

## TODO remove to configs file
MODELS_NAME_MAP={
    "HengNao-v4": "恒脑安全垂域（快速决策）",
    "HengNao-r1": "恒脑安全垂域（深度思考）",
    "HengNao-v1": "恒脑安全垂域（急速思考）"
}

PRIORITY_DICT = {'exclusive': 0, 'priority': 1, 'routine': 2}
