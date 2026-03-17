import os

import consul
import yaml
import time

from . import const
from .logger import SingleLogger



logger = SingleLogger.get_logger()


class ConfigManager:
    def __init__(self, consul_host, consul_port, token, config_key):
        self.consul = consul.Consul(host=consul_host, port=consul_port, token=token)
        self.config_key = config_key
        self.current_config = {}
        self.index = None  # 用于长轮询的index
        os.makedirs(os.path.dirname(const.CONSUL_HOT_SWAPS_CONFIG_FILE), exist_ok=True)

    def update_config(self, new_config):
        # 更新配置
        if new_config != self.current_config:
            self.current_config = new_config
            try:
                # 保存到文件
                if self.current_config:
                    with open(const.CONSUL_HOT_SWAPS_CONFIG_FILE, mode="w+", encoding="utf-8") as f:
                        yaml.dump(yaml.safe_load(new_config), f, default_flow_style=False, encoding="utf-8",
                                  allow_unicode=True)
            except Exception as e:
                logger.warning(f"check consul hot swaps failed, cause by:" + str(e), extra={'category': 'system'})

    def watch_config(self):
        # 长轮询获取配置变化
        while True:
            try:
                index, data = self.consul.kv.get(self.config_key, index=self.index)
                if index != self.index:  # 配置变化时index会更新
                    new_config = data['Value'].decode('utf-8') if data and 'Value' in data else None
                    logger.info(f'Find new config, old index {self.index}, new index {index}')
                    self.update_config(new_config)
                    self.index = index
            except Exception as e:
                logger.warning(f"watching config: {e}")
            time.sleep(10)  # 防止过快轮询


def watch_config_server(consul_host, consul_port, token, config_key):
    _config_Manager = ConfigManager(consul_host, consul_port, token, config_key)
    _config_Manager.watch_config()

