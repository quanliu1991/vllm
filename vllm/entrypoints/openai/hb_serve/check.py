import hashlib
import os
import re
import yaml


def modify_model_name(model_name, yaml_data, yaml_file):
    """在YAML文件中修改模型名称"""
    try:
        yaml_data['model_name'] = model_name
        with open(yaml_file, 'w') as file:
            yaml.dump(yaml_data, file, sort_keys=False)
            file.close()
    except Exception as e:
        raise Exception(f"修改{yaml_file}中的模型名称失败：{e}！")


def modify_version(version, yaml_data, yaml_file):
    """在YAML文件中修改版本号"""
    try:
        yaml_data['hb-serve'] = version
        with open(yaml_file, 'w') as file:
            yaml.dump(yaml_data, file, sort_keys=False)
            file.close()
    except Exception as e:
        raise Exception(f"修改{yaml_file}中的版本号失败：{e}！")


def calc_md5(file_path):
    """计算文件的MD5值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compare_lora_md5(lora_model, yaml_data):
    """比较bin文件的MD5值与YAML数据中记录的MD5值是否匹配"""
    loras_data = yaml_data.get('loras', {})
    for lora_name, lora_info in loras_data.items():
        md5 = lora_info.get('md5')
        file_path = os.path.join(lora_model, lora_name, 'adapter_model.bin')
        if not os.path.exists(file_path):
            continue
        actual_md5 = calc_md5(file_path)
        if actual_md5 != md5:
            return False
    return True


def check_version(allow_lora, version, model_name, lora_model):
    """校验版本号并更新模型名称"""
    current_path = os.path.dirname(os.path.realpath(__file__))
    if allow_lora:
        yaml_file = f'/{current_path}/version_lora.yaml'
    else:
        yaml_file = f'/{current_path}/version_chat.yaml'
    try:
        with open(yaml_file, 'r') as file:
            yaml_data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件{yaml_file}未找到！")
    except Exception as e:
        raise Exception(str(e))

    if version == "":
        raise ValueError("版本号不能为空！")

    if model_name == "":
        raise ValueError("model_name不能为空！")

    # modify_model_name(model_name, yaml_data, yaml_file)

    if allow_lora:
        if lora_model == "":
            raise ValueError("lora_model不能为空！")
        if not compare_lora_md5(lora_model, yaml_data):
            raise ValueError("lora版本不匹配！")


def validate_key(key):
    forbidden_characters = r"[;|&$\(\){}<>\n]"
    if re.search(forbidden_characters, key):
        return False
    return True
