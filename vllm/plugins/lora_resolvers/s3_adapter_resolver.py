# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import threading

import s3fs

from vllm.entrypoints.openai.hb_serve import const
from vllm.entrypoints.openai.hb_serve.logger import (
    ContextualLoggerAdapter,
    LogType,
    SingleLogger,
    Statu,
    enum_to_json,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry

logger = SingleLogger.get_logger()
contextual_logger = ContextualLoggerAdapter(
    logger,
    {
        "sid": None,
        "qid": None,
        "aid": None,
        "iid": None,
        "rid": None,
        "uid": None,
        "category": None,
    },
)


class S3AdapterResolver(LoRAResolver):
    def __init__(self):
        self.lora_key = "aa675b38-ff66-4848-829b-693146ae790a"
        self.resolver_name = "s3_adapter_resolver"
        self.s3 = s3fs.S3FileSystem(anon=False)
        self.s3_path = os.getenv("S3_PATH")
        self.local_path = os.getenv("VLLM_LORA_RESOLVER_CACHE_DIR")

    async def _async_download_folder(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.s3.get(
                self.s3_path,
                self.local_path,
                recursive=True,
                maxdepth=5,
            ),
        )

    def _is_encrypted_adapter(self):
        try:
            if self.s3_path.endswith(".tar.enc"):
                self.decrypt_lock = threading.Lock()
                self.lora_models = os.path.join(
                    self.local_path, self.s3_path.split("/")[-1]
                )
                contextual_logger.info(
                    f"The adapters are encrypted. status: {Statu.SUCCESS}",
                    extra={"category": enum_to_json(LogType.SERVICE_DISCOVERY)},
                )
                return True
            return False
        except FileNotFoundError:
            return False

    def _decrypt_model(self):
        try:
            self.decrypt_lock.acquire()
            model_folder = self.lora_models.replace(".tar.enc", "")
            self.lora_target_folder = "/tmp/.cache/" + model_folder.split("/")[-1]
            os.makedirs("/tmp/.cache/", exist_ok=True)
            cmd = (
                f"openssl aes-256-cbc -d -in {model_folder}.tar.enc -out "
                f"{self.lora_target_folder}.tar -pass pass:{self.lora_key} "
            )
            status = os.system(cmd)
            assert status == 0, "模型解密错误"
            status = os.system(
                f"tar -xvf {self.lora_target_folder}.tar -C /tmp/.cache/"
            )
            assert status == 0, "解压文件解压出现问题"

            assert len(os.listdir(self.lora_target_folder)) >= 1, (
                "解压出现问题：模型文件个数不匹配"
            )
            self.lora_models = self.lora_target_folder
            contextual_logger.info(
                f"adapters decryption complete. status: {Statu.SUCCESS}",
                extra={"category": enum_to_json(LogType.SERVICE_DISCOVERY)},
            )
            self.local_path = self.lora_target_folder
        except Exception as e:
            code = const.HBServeStatus.BAD_DECRYPT
            contextual_logger.error(
                f"adapters decryption failed, {str(e)}. status: {Statu.SUCCESS}",
                extra={"category": enum_to_json(LogType.SERVICE_DISCOVERY)},
            )
            return ErrorResponse(message=str(e), type="BAD_DECRYPT", code=code.value)
        finally:
            self.decrypt_lock.release()

    async def resolve_lora(
        self,
        base_model_name: str,
        lora_name: str,
    ) -> LoRARequest | None:
        folder_name = (
            f"{lora_name}_lora" if not lora_name.endswith("_lora") else lora_name
        )
        if not self.local_path.startswith("/tmp/"):
            os.makedirs(self.local_path, exist_ok=True)
            try:
                await self._async_download_folder()
            except Exception as e:
                contextual_logger.error(
                    f"Model download failed with the resolve "
                    f"'{self.resolver_name}', {str(e)}."
                    f"status: {Statu.FAILED}",
                    extra={"category": enum_to_json(LogType.SERVICE_DISCOVERY)},
                )
                return None

            if self._is_encrypted_adapter():
                self._decrypt_model()

            adapter_config_file = os.path.join(
                self.local_path, folder_name, "adapter_config.json"
            )
            if not os.path.exists(adapter_config_file):
                contextual_logger.error(
                    f"adapter_config.json missing at {self.local_path}. "
                    f"status: {Statu.FAILED}",
                    extra={"category": enum_to_json(LogType.SERVICE_DISCOVERY)},
                )
                return None

            with open(adapter_config_file, "r") as f:
                adapter_config = json.load(f)
            if adapter_config.get("peft_type") != "LORA":
                contextual_logger.error(
                    f"Not a LORA adapter, skipped. status: {Statu.FAILED}",
                    extra={"category": enum_to_json(LogType.SERVICE_DISCOVERY)},
                )

        adapter_file = os.path.join(self.local_path, folder_name)
        return LoRARequest(
            lora_name=lora_name,
            lora_path=adapter_file,
            lora_int_id=abs(hash(adapter_file + "_" + lora_name)),
        )


def register_s3_adapter_resolver():
    resolver_name = "s3_adapter_resolver"
    try:
        resolver = S3AdapterResolver()
        LoRAResolverRegistry.register_resolver(resolver_name, resolver)
        contextual_logger.info(
            f"The resolver '{resolver_name}' registration successful!"
            f"status: {Statu.SUCCESS}",
            extra={"category": enum_to_json(LogType.SERVICE_DISCOVERY)},
        )
    except Exception as e:
        contextual_logger.error(
            f"The resolver '{resolver_name}' failed to register, {str(e)}!"
            f"status: {Statu.FAILED}",
            extra={"category": enum_to_json(LogType.SERVICE_DISCOVERY)},
        )
