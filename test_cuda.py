import torch

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# 如果CUDA可用，则进行下一步检查
if cuda_available:
    # 获取GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    # 获取当前GPU设备名称
    current_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Current GPU name: {current_device_name}")

    # 检查cuDNN是否启用
    cudnn_enabled = torch.backends.cudnn.enabled
    print(f"cuDNN enabled: {cudnn_enabled}")

    # 在GPU上执行一个简单的张量运算
    print("\nTesting a simple computation on GPU...")
    try:
        # 创建一个张量并将其移动到GPU
        x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
        # 执行加法操作
        y = x + x
        print("Computation successful!")
        print(f"Input tensor on GPU: {x}")
        print(f"Result tensor on GPU: {y}")
    except Exception as e:
        print(f"An error occurred during GPU computation: {e}")
