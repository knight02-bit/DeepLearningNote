import torch

# 检查CUDA是否可用
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    # 获取CUDA版本
    print("CUDA version:", torch.version.cuda)

    # 获取GPU数量
    print("GPU count:", torch.cuda.device_count())

    # 获取当前GPU索引
    print("Current GPU index:", torch.cuda.current_device())

    # 获取GPU名称
    print("GPU name:", torch.cuda.get_device_name())

    # 获取所有GPU的详细信息
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(i)}")
        # 获取内存信息
        print(f"Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
else:
    print("CUDA is not available")