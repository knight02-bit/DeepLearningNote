import sys

if "numpy" in sys.modules:
    fake_numpy = sys.modules["numpy"]
    print("模块对象:", fake_numpy)
    print("模块文件:", getattr(fake_numpy, "__file__", None))
else:
    print("numpy 还没被加载")

print("\n=== 搜索 sys.path 里可能存在的 numpy ===")
import os
for p in sys.path:
    try:
        for name in os.listdir(p):
            if name.lower().startswith("numpy"):
                print("找到可疑文件/文件夹:", os.path.join(p, name))
    except Exception:
        pass
