import _thread
import os

import consul
from pydantic import BaseModel
from typing import Optional

from vllm.entrypoints.openai.hb_serve import const
from vllm.entrypoints.openai.hb_serve.const import MODELS_NAME_MAP
from vllm.entrypoints.openai.hb_serve.consul_hot_swaps import watch_config_server
from vllm.entrypoints.openai.hb_serve.hot_swaps_settings import set_global_swaps
from vllm.entrypoints.openai.hb_serve.logger import SingleLogger, ContextualLoggerAdapter, enum_to_json, LogType, Statu
from vllm import __version__ as vllm_version


logger = SingleLogger.get_logger()
contextual_logger = ContextualLoggerAdapter(logger, {'sid': None, 'qid': None, 'aid': None, 'iid': None,
                                                     'rid': None, 'uid': None, 'category': None})
model_name_for_human = os.getenv("MODEL_NAME_FOR_HUMAN", None)


class Meta(BaseModel):
    token_pool_size: Optional[str] = None
    reasoning: Optional[str] = None
    max_model_len: Optional[str] = None
    version: Optional[str] = None
    model_name: Optional[str] = None
    status: Optional[str] = None
    lora: Optional[str] = None
    lora_info: Optional[str] = None
    vllm_version: Optional[str] = None
    model_name_for_human: Optional[str] = None


class ConsulClient(object):
    def __init__(self, host, port, token):
        self.consul_address = f'{host}:{port}'
        self._consul = consul.Consul(host, port, token)
        self.name = const.SERVE_CHAT

    def register_service(self, host, port, tags=None, meta=None):
        try:
            model_name = meta.get("model_name")
        except Exception:
            model_name = "model"
        self.service_id = f"{self.name}_{model_name}_{host}:{port}"
        check = consul.Check.tcp(
            host, port, interval='5s', timeout='10s', deregister='10s')
        tags = tags or []
        # 注册服务
        result = self._consul.agent.service.register(
            name=self.name,
            service_id=self.service_id,
            address=host,
            port=port,
            check=check,
            tags=tags,
            meta=meta)

        if result:
            contextual_logger.info(
                f"Service '{self.name}' registered with Consul '{self.consul_address}', "
                f"status: {Statu.SUCCESS}", extra={'category': enum_to_json(LogType.SERVICE_DISCOVERY)})
        else:
            contextual_logger.error(
                f"Service '{self.name}' registered with Consul '{self.consul_address}' failed,"
                f" status: {Statu.FAILED}", extra={'category': enum_to_json(LogType.SERVICE_DISCOVERY)})

    def deregister(self):
        result = self._consul.agent.service.deregister(self.service_id)
        if result:
            contextual_logger.info(
                f"Service '{self.service_id}' deregistered with Consul '{self.consul_address}'"
                f" success, status: {Statu.SUCCESS}", extra={'category': enum_to_json(LogType.SERVICE_DISCOVERY)})
        else:
            contextual_logger.error(
                f" Service '{self.service_id}' deregistered with Consul '{self.consul_address}' "
                f"failed, status: {Statu.FAILED}", extra={'category': enum_to_json(LogType.SERVICE_DISCOVERY)})

class ConsulList:
    def __init__(self, args):
        lora_info = ""
        self.allow_consul = args.allow_consul == "true"
        if self.allow_consul:
            self.consul_list = []
            self.served_model = [args.model] if args.served_model_name is None else args.served_model_name
            self.allow_lora = args.enable_lora
            self.token_pool_size = str(args.token_pool_size)
            self.max_model_len = str(args.max_model_len)
            self.version = args.version
            self.consul_host_list = args.consul_host.split(',')
            self.consul_port_list = args.consul_port.split(',')
            self.consul_token = args.consul_token
            assert len(self.consul_port_list) == len(self.consul_host_list)
            self.host, self.port = args.host, args.port
            self.reasoning = 'true' if args.reasoning_parser else 'false'

            # 当开启consul时，修改热更新值
            global_swaps_with_consul = {
                "LOGGER_LEVEL": "info",
                "USE_PRIORITY": True,
                "MINIO_ENABLE_LOG": "false",
                "LOG_PROMPT_ENCRYPT": "true"
            }
            for key, value in global_swaps_with_consul.items():
                set_global_swaps(key,value)

            for served_name in self.served_model:
                meta = Meta(token_pool_size=str(args.token_pool_size),
                            max_model_len='0',
                            version=args.version,
                            model_name=served_name,
                            status='-1',
                            reasoning="false" if served_name == "HengNao-v4" else self.reasoning,
                            vllm_version=vllm_version,
                            model_name_for_human=model_name_for_human if model_name_for_human is not None else MODELS_NAME_MAP.get(
                                served_name, served_name),
                            lora="1" if self.allow_lora else "0"
                            ).dict(exclude_unset=True)

                for consul_host, consul_port in zip(self.consul_host_list, self.consul_port_list):
                    consul = ConsulClient(consul_host, consul_port, self.consul_token)
                    # consul.register_service(self.host, self.port, meta=meta)
                    consul.register_service("10.20.152.74", 8300, meta=meta)
                    self.consul_list.append(consul)
                    contextual_logger.info(
                        f"register service {self.host}:{self.port} meta={meta}, status: {Statu.SUCCESS}",
                        extra={'category': enum_to_json(LogType.SERVICE_DISCOVERY)})
                    # 监听热更新yaml
                    _thread.start_new_thread(watch_config_server, (consul_host, consul_port, self.consul_token,
                                                               os.path.join(consul.name, const.CONFIG_KEY)))

    def update_consul_list(self, status):
        if not self.allow_consul:
            return
        for served_name in self.served_model:
            meta = Meta(token_pool_size=self.token_pool_size,
                        max_model_len=self.max_model_len if self.max_model_len else '0',
                        version=self.version,
                        model_name=served_name,
                        reasoning="false" if served_name == "HengNao-v4" else self.reasoning,
                        status=status,
                        vllm_version=vllm_version,
                        model_name_for_human=model_name_for_human if model_name_for_human
                        else MODELS_NAME_MAP.get(served_name, served_name),
                        lora="1" if self.allow_lora else "0"
                        ).dict(exclude_unset=True)
            for c in self.consul_list:
                # c.register_service(self.host, self.port, meta=meta)
                c.register_service("10.20.152.74", 8300, meta=meta)
                contextual_logger.info(
                    f"update consul service {self.host}:{self.port} meta={meta}, status: {Statu.SUCCESS}",
                    extra={'category': enum_to_json(LogType.SERVICE_DISCOVERY)})




if __name__ == '__main__':
    consul_host = "10.20.152.15"  # consul服务器的ip
    consul_port = "8501"  # consul服务器对外的端口
    consul_client = ConsulClient(consul_host, consul_port, None)

    service_name = "ai.model.srv.guanyu"
    service_port = 50003

    meta = {"version": "v1.0.0"}

    consul_client.register_service(service_port, [], meta)

