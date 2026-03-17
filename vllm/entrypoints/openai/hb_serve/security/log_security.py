from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import base64

def encrypt_for_java(plaintext: str) -> str:
    # 确保 key 是 16 / 24 / 32 字节（AES-128/192/256）
    key = "aa675b38-ff66-4848-829b-693146ae790a"
    # len(key_bytes) not in (16, 24, 32)
    key = key.replace("-","")
    key_bytes = key.encode('utf-8')

    # AES/ECB/PKCS5Padding 加密
    cipher = AES.new(key_bytes, AES.MODE_ECB)
    padded_text = pad(plaintext.encode('utf-8'), AES.block_size)
    encrypted = cipher.encrypt(padded_text)

    # Base64 编码后返回
    return base64.b64encode(encrypted).decode('utf-8')

if __name__ == '__main__':
    # 示例
    text = "你好"
    ciphertext = encrypt_for_java(text)
    print("加密结果（Base64）:", ciphertext)