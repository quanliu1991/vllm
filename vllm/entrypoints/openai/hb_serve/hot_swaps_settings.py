import os
from typing import Any
import yaml

from vllm.entrypoints.openai.hb_serve import const
from vllm.entrypoints.openai.hb_serve.logger import SingleLogger
from apscheduler.schedulers.blocking import BlockingScheduler


logger = SingleLogger.get_logger()

"""
    如果没有注册consul，
    "MINIO_ENABLE_LOG": "true"      # 保存日志到minio
    "LOG_PROMPT_ENCRYPT": "false"   # 提示词加密
    "LOGGER_LEVEL": "debug"         # 日志级别
"""
global_swaps = {
    "LOGGER_LEVEL":  os.getenv('LOGGING_LEVEL', "debug"), # 日志级别
    "USE_PRIORITY": os.getenv('USE_PRIORITY', "true"), # 是否使用优先级队列F
    "MINIO_ENABLE_LOG": os.getenv('MINIO_ENABLE_LOG', "true"),
    "LOG_PROMPT_ENCRYPT": os.getenv('LOG_PROMPT_ENCRYPT', "false")
}

def get_nested_value(data, keys, separator='.', default_value: Any = None):
    """
    从嵌套字典中按路径获取值。

    :param data: 字典，或可迭代的键值对集合。
    :param keys: 路径的键列表，或通过 separator 分隔的字符串。
    :param separator: 键之间的分隔符，默认为点号（.）。
    :param default_value: keys对应值的默认值，如果keys不在data中，设置为默认值。
    :return: 路径对应的值，如果路径不存在则返回 default_value。
    """
    if isinstance(keys, str):
        keys = keys.split(separator)

    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default_value
    return current


def check_hot_swaps():
    try:
        with open(const.CONSUL_HOT_SWAPS_CONFIG_FILE, encoding="utf-8") as f:
            conf = yaml.safe_load(f)
            USE_PRIORITY = get_nested_value(conf, 'use_priority')
            MINIO_ENABLE_LOG = get_nested_value(conf, 'minio_enable_log')
            LOG_PROMPT_ENCRYPT = get_nested_value(conf, 'log_prompt_encrypt')

            if isinstance(USE_PRIORITY, str):
                USE_PRIORITY = USE_PRIORITY == 'true'
            elif isinstance(USE_PRIORITY, bool):
                USE_PRIORITY = USE_PRIORITY
            else:
                logger.warning(f"plases set 'use_priority' type bool or str , now is {type(USE_PRIORITY)}")

            set_global_swaps("LOGGER_LEVEL", get_nested_value(conf, 'logger.level').lower())
            set_global_swaps("USE_PRIORITY", USE_PRIORITY)
            if MINIO_ENABLE_LOG:
                set_global_swaps("MINIO_ENABLE_LOG", MINIO_ENABLE_LOG)
            if LOG_PROMPT_ENCRYPT:
                set_global_swaps("LOG_PROMPT_ENCRYPT", LOG_PROMPT_ENCRYPT)
            SingleLogger.set_log_level(get_global_swaps("LOGGER_LEVEL"))
    except Exception as e:
        logger.error(f"read {const.CONSUL_HOT_SWAPS_CONFIG_FILE} config file fail, cause by:" + str(e))


def check_hot_swaps_serve():
    check_hot_swaps()
    scheduler = BlockingScheduler()
    scheduler.add_job(check_hot_swaps, 'interval', seconds=10, args=[])
    scheduler.start()

def set_global_swaps(key, value):
    global_swaps[key] = value

def get_global_swaps(key):
    value = global_swaps.get(key)
    if value is None:
        logger.warning(f"{key} not in global_swaps")
    return value

