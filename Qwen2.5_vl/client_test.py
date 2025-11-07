import base64
import os
from openai import OpenAI

# --- 配置 ---
# 替换为vLLM服务器所在的局域网IP地址
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8000
# vLLM加载的模型路径，必须与启动命令中的--model参数一致
MODEL_PATH = "/home/work301/models/Qwen2.5-VL-7B-Instruct"

# API的基地址
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}/v1"

# 本地图片文件路径
IMAGE_PATH = "/home/work301/vsc_projects/MLLM1/Qwen2.5_vl/test.jpg"
# 要向模型提出的问题
TEXT_PROMPT = "你知道这只网红猫的名字吗？"

# --- 脚本核心 ---

def encode_image_to_base64(image_path):
    """将图片文件编码为Base64字符串"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"错误：找不到图片文件 '{image_path}'")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_inference():
    """执行推理请求"""
    print("--- 开始进行多模态推理 ---")
    
    # 1. 初始化OpenAI客户端，指向本地vLLM服务器
    # 由于是本地服务，API密钥是可选的，可以填任意字符串
    client = OpenAI(
        api_key="not-used",
        base_url=BASE_URL,
    )

    try:
        # 2. 将图片编码为Base64
        print(f"读取并编码图片: {IMAGE_PATH}")
        base64_image = encode_image_to_base64(IMAGE_PATH)
        image_url = f"data:image/jpeg;base64,{base64_image}"

        # 3. 构建符合OpenAI多模态格式的请求消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TEXT_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
        
        print(f"向模型发送请求: \"{TEXT_PROMPT}\"")

        # 4. 发送请求到vLLM服务器
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
            max_tokens=2048,  # 设置期望的最大回复长度
            temperature=0.7,  # 控制生成文本的随机性
        )

        # 5. 打印模型的回复
        print("\n--- 模型回复 ---")
        result = response.choices[0].message.content
        print(result)
        print("------------------\n")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"\n请求过程中发生错误: {e}")
        print("请检查vLLM服务是否正常运行，以及服务器IP和端口是否配置正确。")

if __name__ == "__main__":
    run_inference()
