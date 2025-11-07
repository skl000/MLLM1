#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='/home/work301/models/Qwen2.5-VL-7B-Instruct')
print(f"模型已下载到: {model_dir}")