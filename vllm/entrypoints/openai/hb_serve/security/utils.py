import os
import random
import re

sublayer_map = {
    "qwen2": {
        "input_layernorm.weight": 0,
        "mlp.down_proj.weight": 1,
        "mlp.gate_proj.weight": 2,
        "mlp.up_proj.weight": 3,
        "post_attention_layernorm.weight": 4,
        "self_attn.k_proj.bias": 5,
        "self_attn.k_proj.weight": 6,
        "self_attn.o_proj.weight": 7,
        "self_attn.q_proj.bias": 8,
        "self_attn.q_proj.weight": 9,
        "self_attn.v_proj.bias": 10,
        "self_attn.v_proj.weight": 11
    },
    "qwen2_gptq_4": {'input_layernorm.weight': 0, 'mlp.down_proj.g_idx': 1, 'mlp.down_proj.qweight': 2,
         'mlp.down_proj.qzeros': 3, 'mlp.down_proj.scales': 13, 'mlp.gate_proj.g_idx': 14,
         'mlp.gate_proj.qweight': 6, 'mlp.gate_proj.qzeros': 7, 'mlp.gate_proj.scales': 23,
         'mlp.up_proj.g_idx': 9, 'mlp.up_proj.qweight': 28, 'mlp.up_proj.qzeros': 11, 'mlp.up_proj.scales': 12,
         'post_attention_layernorm.weight': 4, 'self_attn.k_proj.bias': 5, 'self_attn.k_proj.g_idx': 15,
         'self_attn.k_proj.qweight': 16, 'self_attn.k_proj.qzeros': 17, 'self_attn.k_proj.scales': 18,
         'self_attn.o_proj.g_idx': 19, 'self_attn.o_proj.qweight': 20, 'self_attn.o_proj.qzeros': 21,
         'self_attn.o_proj.scales': 22, 'self_attn.q_proj.bias': 8, 'self_attn.q_proj.g_idx': 24,
         'self_attn.q_proj.qweight': 25, 'self_attn.q_proj.qzeros': 26, 'self_attn.q_proj.scales': 27,
         'self_attn.v_proj.bias': 10, 'self_attn.v_proj.g_idx': 29, 'self_attn.v_proj.qweight': 30,
         'self_attn.v_proj.qzeros': 31, 'self_attn.v_proj.scales': 32},
    "qwen3": {
        "input_layernorm.weight": 0,
        "mlp.down_proj.weight": 1,
        "mlp.gate_proj.weight": 2,
        "mlp.up_proj.weight": 3,
        "post_attention_layernorm.weight": 4,
        "self_attn.k_norm.weight": 5,
        "self_attn.k_proj.weight": 6,
        "self_attn.o_proj.weight": 7,
        "self_attn.q_norm.weight": 8,
        "self_attn.q_proj.weight": 9,
        "self_attn.v_proj.weight": 10
    },
    "qwen3_5_moe": {
        "mlp.experts.gate_up_proj": 0,
        "mlp.experts.down_proj": 1,
        "mlp.shared_expert.down_proj.weight": 2,
        "mlp.shared_expert.gate_proj.weight": 3,
        "mlp.shared_expert.up_proj.weight": 4,
        "mlp.shared_expert_gate.weight": 5,
        "mlp.gate.weight": 6,
        "input_layernorm.weight": 7,
        "post_attention_layernorm.weight": 8
    }
}


def extract_numbers(text):
    # 使用正则表达式匹配所有的数字（包括整数和小数）
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    else:
        return None


def get_new_layer_name(layer_name, orders, encrypt=True):
    # if encrypt is True, get encrypt name, else get decrypt name
    decrypt_model_type = os.getenv("DECRYPT_MODEL_TYPE") #"qwen3_5_moe" #os.getenv("DECRYPT_MODEL_TYPE")
    layer_id = extract_numbers(layer_name)
    if layer_id is not None and decrypt_model_type is not None:
        sep = "." + str(layer_id) + "."
        try:
            prefix_name, posix_name = layer_name.split(sep, 1)
        except ValueError:
            prefix_name = layer_name
            posix_name = ""
        try:
            sublayer_id = sublayer_map[decrypt_model_type][posix_name]
        except KeyError:
            sublayer_id = None
        if sublayer_id is not None:
            if encrypt:
                layer_name = prefix_name + "." + str(orders[sublayer_id][layer_id]) + "." + posix_name
            else:
                layer_name = prefix_name + "." + str(orders[sublayer_id].index(layer_id)) + "." + posix_name
    return layer_name


def generate_unique_orders(layer_number, num_orders=33, seed=42):
    orders = []
    input_list = [i for i in range(layer_number)]  # 你的输入列表

    # 固定初始种子
    random.seed(seed)

    for i in range(num_orders):
        shuffled_list = input_list[:]
        random.shuffle(shuffled_list)
        orders.append(shuffled_list)

        # 改变种子以生成不同的顺序
        random.seed(seed + i + 1)  # 每次增加种子值，确保顺序不同

    return orders


def update_hb_layer_name(name, input_len, encrypt=False):
    orders = generate_unique_orders(input_len)
    name = get_new_layer_name(name, orders, encrypt=encrypt)
    return name
