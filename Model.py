# 导入 PyTorch 主包，用于张量计算、设备检测以及模型运行。
import torch

# 从 SymPy 导入 false；当前文件未使用该名称，因为代码中使用的是 Python 内置的 False。
from sympy import false

# 从 PyTorch 导入神经网络模块，并使用简写名称 nn。
from torch import nn

# 导入 torchsummary 的 summary 函数，用于打印模型各层结构和参数量。
from torchsummary import summary


# 定义一个继承自 nn.Module 的基本残差块。
class Residual(nn.Module):
    # 初始化残差块，接收输入通道数、输出通道数、是否启用 1×1 卷积以及步长。
    def __init__(self, input_channels, output_channels, use_cov1=False, strides=1):
        # 调用父类 nn.Module 的初始化方法，以便正确注册网络层和参数。
        super(Residual, self).__init__()
        # 创建 ReLU 激活函数，用来为网络加入非线性表达能力。
        self.relu = nn.ReLU()
        # 创建第一个 3×3 卷积层；它可以通过 strides 改变特征图尺寸，并将通道数变为 output_channels。
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=strides,
            # 将卷积层的填充值设置为 1，使步长为 1 时特征图的高和宽保持不变。
            padding=1,
        )
        # 创建第二个 3×3 卷积层，保持输出通道数以及特征图的高和宽不变。
        self.conv2 = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )
        # 为第一个卷积层的输出创建批量归一化层。
        self.bn1 = nn.BatchNorm2d(output_channels)
        # 为第二个卷积层的输出创建批量归一化层。
        self.bn2 = nn.BatchNorm2d(output_channels)
        # 判断捷径分支是否需要使用 1×1 卷积调整通道数或特征图尺寸。
        if use_cov1:
            # 创建 1×1 卷积，将输入 x 的通道数和尺寸调整到能够与主分支 y 相加。
            self.cov3 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                # 让捷径分支使用与主分支相同的步长，并且不进行边缘填充。
                stride=strides,
                padding=0,
            )
        # 当输入与输出形状一致时，不需要额外的 1×1 卷积。
        else:
            # 使用 None 表示捷径分支直接保留原始输入 x。
            self.cov3 = None

    # 定义残差块的前向传播过程，x 是输入特征图。
    def forward(self, x):
        # 输入依次经过第一个卷积层、批量归一化层和 ReLU 激活函数。
        y = self.relu(self.bn1(self.conv1(x)))
        # 中间结果再经过第二个卷积层和批量归一化层，此处暂不进行激活。
        y = self.bn2(self.conv2(y))

        # 如果创建了 1×1 卷积，就需要调整捷径分支输入 x 的形状。
        if self.cov3:
            # 让 x 经过 1×1 卷积，使其形状与主分支输出 y 一致。
            x = self.cov3(x)

        # 将主分支 y 与捷径分支 x 相加，再使用 ReLU 激活，形成残差连接。
        y = self.relu(y + x)

        # 返回残差块处理后的特征图。
        return y


# 定义完整的 ResNet18 网络，并继承 nn.Module。
class ResNet18(nn.Module):
    # 初始化网络；Residual 参数用于指定构建网络时采用的残差块类型。
    def __init__(self, Residual):
        # 调用父类 nn.Module 的初始化方法。
        super(ResNet18, self).__init__()
        # 使用 Sequential 定义网络的第一个阶段，各层会按照书写顺序依次执行。
        self.b1 = nn.Sequential(
            # 使用 7×7 卷积接收单通道图像，输出 64 个通道，并通过步长 2 缩小特征图。
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            # 对卷积结果应用 ReLU 激活函数。
            nn.ReLU(),
            # 对 64 个通道的特征进行批量归一化。
            nn.BatchNorm2d(64),
            # 使用 3×3 最大池化并设置步长为 2，进一步缩小特征图。
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 结束第一个阶段 b1 的 Sequential 定义。
        )

        # 定义第二个阶段 b2，其中包含两个输出通道数为 64 的残差块。
        self.b2 = nn.Sequential(
            # 第一个残差块保持 64 个通道和原有尺寸，因此不需要 1×1 卷积。
            Residual(input_channels=64, output_channels=64, use_cov1=False, strides=1),
            # 第二个残差块同样保持 64 个通道和原有尺寸。
            Residual(input_channels=64, output_channels=64, use_cov1=False, strides=1),
            # 结束第二个阶段 b2 的 Sequential 定义。
        )

        # 定义第三个阶段 b3，将通道数从 64 增加到 128。
        self.b3 = nn.Sequential(
            # 通过步长 2 缩小特征图，并用 1×1 卷积同步调整捷径分支的尺寸和通道数。
            Residual(input_channels=64, output_channels=128, use_cov1=True, strides=2),
            # 在 128 个通道上继续提取特征，同时保持特征图尺寸不变。
            Residual(
                input_channels=128, output_channels=128, use_cov1=False, strides=1
            ),
            # 结束第三个阶段 b3 的 Sequential 定义。
        )

        # 定义第四个阶段 b4，将通道数从 128 增加到 256。
        self.b4 = nn.Sequential(
            # 通过步长 2 缩小特征图，并使用 1×1 卷积调整捷径分支。
            Residual(input_channels=128, output_channels=256, use_cov1=True, strides=2),
            # 在 256 个通道上继续提取特征，同时保持特征图尺寸不变。
            Residual(
                input_channels=256, output_channels=256, use_cov1=False, strides=1
            ),
            # 结束第四个阶段 b4 的 Sequential 定义。
        )

        # 定义第五个阶段 b5，将通道数从 256 增加到 512。
        self.b5 = nn.Sequential(
            # 通过步长 2 缩小特征图，并使用 1×1 卷积调整捷径分支。
            Residual(input_channels=256, output_channels=512, use_cov1=True, strides=2),
            # 在 512 个通道上继续提取特征，同时保持特征图尺寸不变。
            Residual(
                input_channels=512, output_channels=512, use_cov1=False, strides=1
            ),
            # 结束第五个阶段 b5 的 Sequential 定义。
        )

        # 定义网络的分类阶段 b6，将卷积特征转换为 10 个类别的输出。
        self.b6 = nn.Sequential(
            # 将每个通道的特征图自适应平均池化为 1×1，不依赖输入图像的具体尺寸。
            nn.AdaptiveAvgPool2d((1, 1)),
            # 先将多维特征展平，再通过全连接层把 512 个特征映射为 10 个类别分数。
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10),
            # 结束分类阶段 b6 的 Sequential 定义。
        )

    # 定义整个 ResNet18 网络的前向传播过程。
    def forward(self, x):
        # 输入首先经过初始卷积、激活、批量归一化和最大池化阶段。
        x = self.b1(x)
        # 特征依次经过包含两个 64 通道残差块的第二阶段。
        x = self.b2(x)
        # 特征经过第三阶段，通道数增加到 128，空间尺寸减小。
        x = self.b3(x)
        # 特征经过第四阶段，通道数增加到 256，空间尺寸继续减小。
        x = self.b4(x)
        # 特征经过第五阶段，通道数增加到 512，空间尺寸继续减小。
        x = self.b5(x)
        # 特征经过池化、展平和全连接层，得到 10 个类别的输出分数。
        x = self.b6(x)
        # 返回网络最终的分类结果。
        return x


# 仅当直接运行当前文件时执行下面的模型测试代码；被其他文件导入时不会执行。
if __name__ == "__main__":
    # 如果 CUDA 可用则选择 GPU，否则选择 CPU 作为模型运行设备。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建 ResNet18 模型，将 Residual 作为残差块类型，并把模型移动到选定设备。
    model = ResNet18(Residual).to(device)
    # 以单通道 224×224 图像为输入，打印模型结构、各层输出形状和参数量。
    print(summary(model, input_size=(1, 224, 224)))
