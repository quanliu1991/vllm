import argparse
import importlib
import os
import json

from vllm.entrypoints.openai.hb_serve.security.config import model_map


def config_encrypt(json_folder: str):
    config_file_name = model_map[json_folder.split("/")[-1]]
    config_file_list = []
    with open(os.path.join(f"{config_file_name}.py"), 'w', encoding='utf-8') as config_file:
        for file_name in os.listdir(json_folder):
            if file_name.endswith(".json"):
                with open(os.path.join(json_folder, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if file_name == "model.safetensors.index.json":
                    # model.safetensors.index.json in /workspace/projects/vllm_0.6.2/vllm/vllm/model_
                    # executor/model_loader/loader.py filter any files not found in the index.
                    file_name = "model_safetensors_index.json"
                    continue
                config_file.write(f'{file_name.replace(".json", "")} = {repr(data)}\n')
                config_file_list.append(file_name.replace(".json", ""))
        config_file.write(f'config_file_list = {config_file_list}')


def config_decrypt(model_path: str, output_folder: str):
    config_file_name: str = model_map[model_path.split("/")[-1]]
    os.makedirs(output_folder, exist_ok=True)
    config_module = importlib.import_module(f"vllm.entrypoints.openai.hb_serve.security.{config_file_name}")
    filename_map = {
        'model_safetensors_index': 'model.safetensors.index'
    }
    for config_file in config_module.config_file_list:
        filename = filename_map.get(config_file, config_file)
        filepath = os.path.join(output_folder, f"{filename}.json")
        try:
            value = getattr(config_module, config_file)
            with open(filepath, 'w', encoding='utf-8') as f:
                indent = None if config_file == "vocab" else 2
                json.dump(value, f, ensure_ascii=False, indent=indent)
            os.chmod(filepath, 0o600)
        except Exception as e:
            print(f"Error writing config file {config_file}: {e}")
    # add 返回模型类型，用于解密配置文件选择
    with open(os.path.join(output_folder, "config.json"), "r") as f:
        config = json.load(f)
        model_type = config.get("model_type", None)
        assert model_type is not None, "model config error, have not model_type"
        quantization_config = config.get("quantization_config", None)
        if quantization_config is None:
            quant_method = ""
            bits = ""
        else:
            bits = quantization_config.get("bits", None)
            quant_method = quantization_config.get("quant_method", None)

            assert quant_method is not None, "model config error, have not quant_method in quantization_config"
            assert bits is not None, "model config error, have not bits in quantization_config"
            model_type = model_type + "_" + quant_method + "_" +str(bits)
    # 设置环境变量,用于模型加载
    os.environ["DECRYPT_MODEL_TYPE"] = model_type

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="OpenAI-Compatible RESTful API server.")
    parser.add_argument("--model", type=str, default="/workspace/llm_models/HengNao-r1-32b", help="model path")
    parser.add_argument("--output-model", type=str, default="./temp_model", help="output model path")
    parser.add_argument("--decrypt", action="store_true", help="output model path")
    args = parser.parse_args()
    if args.decrypt:
        config_decrypt(args.model, args.output_model)
    else:
        config_encrypt(args.model)
