import sys
import torch
import torchvision
import numpy
import pandas
import matplotlib
import sklearn
import scipy
import PIL
import tqdm
import torchinfo
import torchsummary

print("=" * 50)
print("Python 路径:", sys.executable)
print("Python 版本:", sys.version)
print()

print("PyTorch 版本:", torch.__version__)
print("Torchvision 版本:", torchvision.__version__)
print("NumPy 版本:", numpy.__version__)
print("Pandas 版本:", pandas.__version__)
print("Matplotlib 版本:", matplotlib.__version__)
print("Scikit-learn 版本:", sklearn.__version__)
print("SciPy 版本:", scipy.__version__)
print()

print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 数量:", torch.cuda.device_count())
    print("当前 GPU:", torch.cuda.get_device_name(0))
    print("PyTorch CUDA 版本:", torch.version.cuda)

    # 真正做一次 GPU 运算
    x = torch.randn(2000, 2000, device="cuda")
    y = torch.randn(2000, 2000, device="cuda")
    z = x @ y

    print("运算结果所在设备:", z.device)
    print("GPU 运算测试成功！")
else:
    print("没有检测到 CUDA GPU")

print("=" * 50)
print("环境测试完成")