import argparse
import os

import re
from safetensors.torch import load_file, save_file

from  vllm.entrypoints.openai.hb_serve.security.utils import generate_unique_orders, get_new_layer_name, extract_numbers
from huggingface_hub import split_torch_state_dict_into_shards



def load_safetensors_from_folder(model_path):
    tensor_dict = {}
    for filename in os.listdir(model_path):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(model_path, filename)
            tensor_data = load_file(file_path)
            tensor_dict.update(tensor_data)
    return tensor_dict


def model_encrpt(model_path, output_path):
    tensor_dict = load_safetensors_from_folder(model_path)
    layer_names = list(tensor_dict.keys())
    layer_ids=[]
    sublayer_number_id_name_map={}
    sub_layer_id = 0
    for layer_name in layer_names:
        layer_id = extract_numbers(layer_name)
        if layer_id is not None:
            layer_ids.append(layer_id)
        if layer_id == 3:
            sublayer_number_id_name_map[sub_layer_id]=layer_name.split(str(layer_id))[-1]
            sub_layer_id += 1
    decrypt_model_type = os.getenv("DECRYPT_MODEL_TYPE")    #"qwen3_5_moe" #os.getenv("DECRYPT_MODEL_TYPE")
    if decrypt_model_type == "qwen3_5_moe":
        sublayer_number = 9
    else:
        sublayer_number = layer_ids.count(0)
    layer_number = max(layer_ids) + 1


    orders = generate_unique_orders(layer_number, num_orders=sublayer_number)

    new_tensor_dict = {}
    for layer_name in layer_names:
        new_layer_name = get_new_layer_name(layer_name, orders)
        new_tensor_dict[new_layer_name] = tensor_dict[layer_name]

    sorted_tensor_dict={}
    for layer_name in sorted(new_tensor_dict):
        sorted_tensor_dict[layer_name] = new_tensor_dict[layer_name]
    state_dict_split = split_torch_state_dict_into_shards(sorted_tensor_dict,filename_pattern='model{suffix}.safetensors',max_shard_size="4GB")
    os.makedirs(output_path, exist_ok=True)
    # Clean the folder from a previous save
    for filename in os.listdir(output_path):
        full_filename = os.path.join(output_path, filename)
        # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
        # in distributed settings to avoid race conditions.
        weights_no_suffix = 'model.safetensors'.replace(".bin", "").replace(".safetensors", "")

        # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
        filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
        reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

        if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors.keys()
                and reg.fullmatch(filename_no_suffix) is not None
        ):
            os.remove(full_filename)
    # Save the model
    filename_to_tensors = state_dict_split.filename_to_tensors.items()

    for shard_file, tensors in filename_to_tensors:
        shard = {tensor: new_tensor_dict[tensor].contiguous() for tensor in tensors}

        save_file(shard, os.path.join(output_path, shard_file), metadata={"format": "pt"})




def model_decrypt(model_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(model_path):
        source_path = os.path.join(model_path, filename)
        target_path = os.path.join(output_path, filename)

        if os.path.isfile(source_path):
            try:
                os.symlink(source_path, target_path)
            except FileExistsError:
                print(f"Symlink already exists for: {target_path}")
            except OSError as e:
                print(f"Error creating symlink for {source_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="OpenAI-Compatible RESTful API server.")
    parser.add_argument("--model", type=str, default="/workspace/llm_models/Qwen3.5-35B-A3B", help="model path")
    parser.add_argument("--output-model", type=str, default="/workspace/llm_models/saved_model/HengNao-3_5-35ba3b-chat", help="output model path")
    parser.add_argument("--decrypt", action="store_true", help="output model path")
    args = parser.parse_args()
    if args.decrypt:
        model_decrypt(args.model, args.output_model)
    else:
        model_encrpt(args.model, args.output_model)

