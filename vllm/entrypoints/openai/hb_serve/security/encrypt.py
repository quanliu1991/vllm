import json
import os
import shutil
import threading

import shortuuid

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import create_error_response
from vllm.transformers_utils.utils import is_s3

from vllm.entrypoints.openai.cli_args import LoRAParserAction
from vllm.entrypoints.openai.hb_serve import const
from vllm.entrypoints.openai.hb_serve.consul_client import ConsulList
from vllm.entrypoints.openai.hb_serve.security.config_security import config_encrypt, config_decrypt
from vllm.entrypoints.openai.hb_serve.security.model_security import model_encrpt, model_decrypt
from vllm.entrypoints.openai.hb_serve.logger import SingleLogger, ContextualLoggerAdapter, enum_to_json, LogType, Statu
# from vllm.plugins.lora_resolvers.s3_adapter_resolver import register_s3_adapter_resolver
from vllm.transformers_utils.runai_utils import is_runai_obj_uri

mindie_config_path = os.getenv("MINDIE_CONFIG_PATH","/usr/local/Ascend/mindie/latest/mindie-service/conf")
logger = SingleLogger.get_logger()
contextual_logger = ContextualLoggerAdapter(logger, {'sid': None, 'qid': None, 'aid': None, 'iid': None,
                                                     'rid': None, 'uid': None, 'category': None})

class Encrypt:
    def __init__(self, args, seed=42,  consul_list: ConsulList=None):
        self.args = args
        self.args.llm_model_name = args.model.split("/")[-1]
        self.model_path = self.args.model
        self.seed = seed
        self.temp_model_path = f"/tmp/{shortuuid.random()}"
        self.model_encryption = self.is_encryption()
        self.mindie_config_path = mindie_config_path
        self.mindie_config_mount_path = "/workspace/config.json"
        self.consul_list = consul_list
        self.lora_key = "aa675b38-ff66-4848-829b-693146ae790a"
        self.lora_model_encryption=False

        if self.model_encryption:
            self.decrpty()
            if hasattr(args, 'mindie_port'): # npu
                self.update_mindie_config()
            else:
                self.args.model = self.temp_model_path
        else:
            if hasattr(args, 'mindie_port'):  # npu
                self.move_mindie_config()

        if args.enable_lora:
            self.lora_models: str = args.lora_models
            self.lora_model_encryption = self.is_lora_encryption()

            if self.lora_model_encryption:
                self.decrpty_lora()

            self.lora_modules = self.gen_lora_modules()

            def nullable_str(value):
                if value.lower() in ('none', 'null', ''):
                    return None
                return value
            lora_action = LoRAParserAction(
                dest="lora_modules",  # 目标属性名
                option_strings=["--lora-modules"],  # 参数名
                type=nullable_str,  # 参数类型
                default=None,  # 默认值
                nargs='+',  # 接受多个值
                help="LoRA module configurations"  # 帮助信息
            )
            lora_action(parser=None, namespace=args, values=self.lora_modules)

    def encrypt(self):
        config_encrypt(self.model_path)
        model_encrpt(self.model_path, self.temp_model_path)

    def decrpty(self):
        config_decrypt(self.model_path, self.temp_model_path)
        model_decrypt(self.model_path, self.temp_model_path)
        self.consul_list.update_consul_list("1")

    def update_mindie_config(self):
        assert os.path.isfile(self.mindie_config_mount_path ), f"config.json not in /workspace"
        with open(self.mindie_config_mount_path , "r") as f:
            mindie_config = json.load(f) 
            mindie_config["BackendConfig"]["ModelDeployConfig"]["ModelConfig"][0]["modelWeightPath"] = self.temp_model_path
        with open(os.path.join(self.mindie_config_path,'config.json'), 'w') as file:
            json.dump(mindie_config, file, indent=4)
        os.chown(os.path.join(self.mindie_config_path,'config.json'), 0, 0)
        os.chmod(os.path.join(self.mindie_config_path,'config.json'), 0o600)


    def move_mindie_config(self):
        shutil.copy(self.mindie_config_mount_path , os.path.join(self.mindie_config_path,'config.json'))
        os.chown(os.path.join(self.mindie_config_path,'config.json'), 0, 0)
        os.chmod(os.path.join(self.mindie_config_path,'config.json'), 0o600)

    def is_encryption(self):
        if is_runai_obj_uri(self.model_path):
            return
        for filename in os.listdir(self.model_path):
            if filename.endswith('.json'):
                logger.info(f"model path {self.model_path} is not encrypted.")
                os.environ["MODEL_ENCRYPTION"] = "false"
                return False
        os.environ["MODEL_ENCRYPTION"] = "true"
        return True

    def is_lora_encryption(self):
        if is_s3(self.lora_models):
            # register_s3_adapter_resolver()
            return
        if os.path.isdir(self.lora_models):
            return False
        if self.lora_models.endswith(".tar.enc"):
            self.decrypt_lock = threading.Lock()
            return True

    def decrpty_lora(self):
        try:
            self.decrypt_lock.acquire()
            self.consul_list.update_consul_list('0')
            model_folder = self.lora_models.replace('.tar.enc', '')
            self.lora_target_folder = '/tmp/.cache/' + model_folder.split('/')[-1]
            os.makedirs('/tmp/.cache/', exist_ok=True)
            cmd = f"openssl aes-256-cbc -d -in {model_folder}.tar.enc -out " \
                  f"{self.lora_target_folder}.tar -pass pass:{self.lora_key} "
            status = os.system(cmd)
            assert status == 0, "模型解密错误"
            status = os.system(f"tar -xvf {self.lora_target_folder}.tar -C /tmp/.cache/")
            assert status == 0, "解压文件解压出现问题"

            assert len(os.listdir(self.lora_target_folder)) >= 1, "解压出现问题：模型文件个数不匹配"
            self.lora_models = self.lora_target_folder
            self.consul_list.update_consul_list("1")
            contextual_logger.info(f"answer: end unzip model, status: {Statu.SUCCESS}",
                                   extra={'category': enum_to_json(LogType.SYSTEM)})

        except Exception as e:
            code = const.HBServeStatus.BAD_DECRYPT
            contextual_logger.error(str(e), extra={}, exc_info=True)
            return ErrorResponse(message=str(e), type="BAD_DECRYPT", code=code.value)
        finally:
            self.decrypt_lock.release()



    def gen_lora_modules(self):
        if is_s3(self.lora_models):
            return
        lora_folders = [folder for folder in os.listdir(self.lora_models)
                        if os.path.isdir(os.path.join(self.lora_models, folder))]

        lora_modules_for_vllm =[f"{folder.replace('_lora','')}={os.path.join(self.lora_models, folder)}" for folder in lora_folders]
        return lora_modules_for_vllm


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.temp_model_path):
            shutil.rmtree(self.temp_model_path)
        if hasattr(self.args, 'mindie_port'): # npu
            self.move_mindie_config()
        if self.lora_model_encryption:
            os.system(f'rm -rf {self.lora_target_folder}.tar {self.lora_target_folder}')



