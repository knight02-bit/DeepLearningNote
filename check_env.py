#!/usr/bin/env python3
"""
深度学习环境版本检查脚本
检查CUDA、PyTorch、TensorFlow等深度学习相关库的版本信息
"""

import sys
import platform
import subprocess

def get_python_version():
    """获取Python版本"""
    return sys.version

def get_cuda_version():
    """获取CUDA版本"""
    try:
        # 尝试通过nvidia-smi获取CUDA版本
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    return cuda_version
        return "未检测到CUDA或nvidia-smi不可用"
    except FileNotFoundError:
        return "nvidia-smi命令未找到"
    except Exception as e:
        return f"检查CUDA版本时出错: {str(e)}"

def get_cudnn_version():
    """获取cuDNN版本"""
    try:
        import torch
        if torch.backends.cudnn.is_available():
            return torch.backends.cudnn.version()
        else:
            return "cuDNN不可用"
    except ImportError:
        return "PyTorch未安装，无法检查cuDNN"
    except Exception as e:
        return f"检查cuDNN版本时出错: {str(e)}"

def get_pytorch_version():
    """获取PyTorch版本和相关信息"""
    try:
        import torch
        info = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else "N/A",
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else "N/A"
        }
        
        if torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name(0)
            info['device_capability'] = torch.cuda.get_device_capability(0)
        
        return info
    except ImportError:
        return "PyTorch未安装"
    except Exception as e:
        return f"检查PyTorch时出错: {str(e)}"

def get_tensorflow_version():
    """获取TensorFlow版本"""
    try:
        import tensorflow as tf
        info = {
            'version': tf.__version__,
            'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
            'gpu_devices': [device.name for device in tf.config.list_physical_devices('GPU')]
        }
        return info
    except ImportError:
        return "TensorFlow未安装"
    except Exception as e:
        return f"检查TensorFlow时出错: {str(e)}"

def get_other_libraries():
    """获取其他常用深度学习库版本"""
    libraries = {}
    
    # 检查常用库
    lib_list = [
        'numpy', 'pandas', 'matplotlib', 'scikit-learn', 
        'transformers', 'datasets', 'accelerate', 'bitsandbytes',
        'opencv-python', 'pillow', 'torchvision', 'torchaudio'
    ]
    
    for lib in lib_list:
        try:
            if lib == 'opencv-python':
                import cv2
                libraries[lib] = cv2.__version__
            elif lib == 'pillow':
                import PIL
                libraries[lib] = PIL.__version__
            elif lib == 'scikit-learn':
                import sklearn
                libraries[lib] = sklearn.__version__
            else:
                module = __import__(lib)
                libraries[lib] = getattr(module, '__version__', '版本信息不可用')
        except ImportError:
            libraries[lib] = "未安装"
        except Exception as e:
            libraries[lib] = f"检查时出错: {str(e)}"
    
    return libraries

def print_separator(title):
    """打印分隔符"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def main():
    print("深度学习环境版本检查")
    print(f"检查时间: {platform.platform()}")
    
    # 系统信息
    print_separator("系统信息")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"Python版本: {get_python_version()}")
    
    # CUDA信息
    print_separator("CUDA信息")
    cuda_version = get_cuda_version()
    print(f"CUDA版本: {cuda_version}")
    cudnn_version = get_cudnn_version()
    print(f"cuDNN版本: {cudnn_version}")
    
    # PyTorch信息
    print_separator("PyTorch信息")
    pytorch_info = get_pytorch_version()
    if isinstance(pytorch_info, dict):
        print(f"PyTorch版本: {pytorch_info['version']}")
        print(f"CUDA可用: {pytorch_info['cuda_available']}")
        print(f"PyTorch CUDA版本: {pytorch_info['cuda_version']}")
        print(f"GPU设备数量: {pytorch_info['device_count']}")
        if pytorch_info['cuda_available']:
            print(f"当前设备: {pytorch_info['current_device']}")
            print(f"设备名称: {pytorch_info.get('device_name', 'N/A')}")
            print(f"设备计算能力: {pytorch_info.get('device_capability', 'N/A')}")
    else:
        print(f"PyTorch: {pytorch_info}")
    
    # TensorFlow信息
    print_separator("TensorFlow信息")
    tf_info = get_tensorflow_version()
    if isinstance(tf_info, dict):
        print(f"TensorFlow版本: {tf_info['version']}")
        print(f"GPU可用: {tf_info['gpu_available']}")
        print(f"GPU设备: {tf_info['gpu_devices']}")
    else:
        print(f"TensorFlow: {tf_info}")
    
    # 其他库信息
    print_separator("其他深度学习相关库")
    other_libs = get_other_libraries()
    for lib, version in other_libs.items():
        print(f"{lib:20}: {version}")
    
    print(f"\n{'='*60}")
    print("环境检查完成!")

if __name__ == "__main__":
    main()