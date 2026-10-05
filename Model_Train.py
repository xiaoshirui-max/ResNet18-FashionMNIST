# 导入 copy 模块，用于深拷贝并保存模型参数。
import copy

# 导入 time 模块，用于统计训练和验证耗时。
import time

# 导入 PyTorch，用于张量运算、模型训练和设备管理。
import torch

# 从 torchvision 导入 FashionMNIST 数据集类。
from torchvision.datasets import FashionMNIST

# 导入 torchvision 的图像预处理模块。
from torchvision import transforms

# 导入 PyTorch 数据处理工具，并使用 Data 作为简写名称。
import torch.utils.data as Data

# 导入 NumPy 数值计算库，并使用 np 作为简写名称。
import numpy as np

# 导入 Matplotlib 绘图模块，并使用 plt 作为简写名称。
import matplotlib.pyplot as plt

# 从 Model.py 导入残差块和 ResNet18 网络。
from Model import Residual, ResNet18

# 导入 PyTorch 神经网络模块，并使用 nn 作为简写名称。
import torch.nn as nn

# 导入 Pandas，并使用 pd 作为简写名称来组织训练记录。
import pandas as pd


# 定义训练集和验证集的数据处理函数。
def train_val_data_process():
    # 从 ./data 目录读取或下载 FashionMNIST 数据集。
    train_data = FashionMNIST(
        root='./data',
        # 指定读取 FashionMNIST 的官方训练集。
        train=True,
        # 将图像缩放到 224×224，并将其转换为 PyTorch 张量。
        transform=transforms.Compose(
            [transforms.Resize(size=224), transforms.ToTensor()]
        ),
        # 当本地不存在数据集时自动下载。
        download=True,
    )

    # 按照 80% 和 20% 的比例，将官方训练集随机拆分为训练集和验证集。
    train_data, val_data = Data.random_split(
        train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))]
    )
    # 根据拆分后的训练集创建训练数据加载器。
    train_dataloader = Data.DataLoader(
        dataset=train_data,
        # 设置训练时每个批次包含 32 个样本。
        batch_size=32,
        # 每轮训练前随机打乱训练样本顺序。
        shuffle=True,
        # 使用两个子进程并行加载训练数据。
        num_workers=2,
    )

    # 根据拆分后的验证集创建验证数据加载器。
    val_dataloader = Data.DataLoader(
        dataset=val_data,
        # 设置验证时每个批次包含 32 个样本。
        batch_size=32,
        # 在遍历验证集时随机打乱样本顺序。
        shuffle=True,
        # 使用两个子进程并行加载验证数据。
        num_workers=2,
    )

    # 返回训练数据加载器和验证数据加载器。
    return train_dataloader, val_dataloader


# 定义模型训练函数，接收模型、训练/验证数据加载器以及训练轮数。
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    # 按照指定的 num_epochs 轮数循环训练和验证模型。
    for epoch in range(num_epochs):
        # 输出当前轮次编号和最后一轮的编号。
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        # 输出分隔线，让不同训练轮次的日志更清晰。
        print("-" * 10)

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)
        # 逐批遍历验证数据，同时获得批次序号、图像数据和标签。
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        # 输出当前轮次的平均训练损失和训练准确率。
        print(
            "{} train loss:{:.4f} train acc: {:.4f}".format(
                epoch, train_loss_all[-1], train_acc_all[-1]
            )
        )
        # 输出当前轮次的平均验证损失和验证准确率。
        print(
            "{} val loss:{:.4f} val acc: {:.4f}".format(
                epoch, val_loss_all[-1], val_acc_all[-1]
            )
        )

        # 判断当前轮次的验证准确率是否超过此前记录的最高值。
        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        # 将累计耗时转换为分钟和秒并输出。
        print(
            "训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60)
        )

    # 选择最优参数，保存最优参数的模型
    model.load_state_dict(best_model_wts)
    # torch.save(model.load_state_dict(best_model_wts), "C:/Users/86159/Desktop/LeNet/best_model.pth")
    torch.save(best_model_wts, "E:/MyProject/LearnLine/ResNet/ResNet18/best_model.pth")

    # 创建 DataFrame，并将训练轮次作为 epoch 列。
    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            # 将每轮训练损失保存到 train_loss_all 列。
            "train_loss_all": train_loss_all,
            # 将每轮验证损失保存到 val_loss_all 列。
            "val_loss_all": val_loss_all,
            # 将每轮训练准确率保存到 train_acc_all 列。
            "train_acc_all": train_acc_all,
            # 将每轮验证准确率保存到 val_acc_all 列，并结束数据字典和 DataFrame 构造。
            "val_acc_all": val_acc_all,
        }
    )

    # 返回包含每轮损失和准确率的训练过程表格。
    return train_process


# 定义绘图函数，用于显示训练和验证过程中的损失及准确率曲线。
def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    # 在画布中创建一行两列的第一个子图，用于绘制损失曲线。
    plt.subplot(1, 2, 1)
    # 使用红色圆点实线绘制每轮训练损失。
    plt.plot(
        train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss"
    )
    # 使用蓝色方块实线绘制每轮验证损失。
    plt.plot(
        train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss"
    )
    # 显示训练损失和验证损失的图例。
    plt.legend()
    # 将第一个子图的横轴名称设置为 epoch。
    plt.xlabel("epoch")
    # 将第一个子图的纵轴名称设置为 Loss。
    plt.ylabel("Loss")
    # 切换到一行两列的第二个子图，用于绘制准确率曲线。
    plt.subplot(1, 2, 2)
    # 使用红色圆点实线绘制每轮训练准确率。
    plt.plot(
        train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc"
    )
    # 使用蓝色方块实线绘制每轮验证准确率。
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    # 将第二个子图的横轴名称设置为 epoch。
    plt.xlabel("epoch")
    # 将第二个子图的纵轴名称设置为 acc。
    plt.ylabel("acc")
    # 显示训练准确率和验证准确率的图例。
    plt.legend()
    # 显示包含两个子图的绘图窗口。
    plt.show()


# 仅当直接运行当前文件时执行模型创建、数据加载、训练和绘图代码。
if __name__ == '__main__':
    # 加载需要的模型
    ResNet18 = ResNet18(Residual)
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(ResNet18, train_data, val_data, num_epochs=20)
    # 绘制训练与验证的损失曲线和准确率曲线。
    matplot_acc_loss(train_process)
