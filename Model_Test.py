# 导入 PyTorch，用于张量运算、设备判断以及模型参数加载。
import torch

# 导入 PyTorch 的数据处理工具，并使用 Data 作为简写名称。
import torch.utils.data as Data

# 导入 torchvision 的图像预处理模块。
from torchvision import transforms

# 导入 FashionMNIST 数据集类。
from torchvision.datasets import FashionMNIST

# 从 Model.py 导入残差块和 ResNet18 网络。
from Model import Residual, ResNet18


# 定义测试数据处理函数，用来创建并返回测试集的数据加载器。
def test_data_process():
    # 从 ./data 目录读取或下载 FashionMNIST 数据集。
    test_data = FashionMNIST(
        root='./data',
        # 指定读取官方测试集，而不是训练集。
        train=False,
        # 将图像缩放到 224×224，并将其转换为 PyTorch 张量。
        transform=transforms.Compose(
            [transforms.Resize(size=224), transforms.ToTensor()]
        ),
        # 当本地不存在数据集时自动下载。
        download=True,
    )

    # 根据测试数据集创建 DataLoader，以便按批次读取数据。
    test_dataloader = Data.DataLoader(
        dataset=test_data,
        # 每个批次只读取一个测试样本。
        batch_size=1,
        # 在遍历测试集时随机打乱样本顺序。
        shuffle=True,
        # 不创建额外的数据加载子进程，由当前主进程加载数据。
        num_workers=0,
    )
    # 返回构建完成的测试数据加载器。
    return test_dataloader


# 定义模型测试函数，接收待测试模型和测试数据加载器。
def test_model_process(model, test_dataloader):
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 讲模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    # 记录已经参与准确率计算的测试样本总数。
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        # 逐批读取测试图像 test_data_x 和对应标签 test_data_y。
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    # 将最终测试准确率输出到控制台。
    print("测试的准确率为：", test_acc)


# 仅当直接运行当前文件时执行下面的模型测试和逐样本预测代码。
if __name__ == "__main__":
    # 加载模型
    model = ResNet18(Residual)
    # 从 best_model.pth 文件读取并恢复训练好的模型参数。
    model.load_state_dict(torch.load('best_model.pth'))
    # # 利用现有的模型进行模型的测试
    test_dataloader = test_data_process()
    # 使用加载好的模型在整个测试集上计算准确率。
    test_model_process(model, test_dataloader)

    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    # 再次确保模型已经移动到选定的测试设备。
    model = model.to(device)
    # 按照 FashionMNIST 的标签编号顺序定义十个类别名称。
    classes = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot',
    ]
    # 关闭梯度计算，以减少逐样本预测时的内存和计算开销。
    with torch.no_grad():
        # 逐批遍历测试集；当前 batch_size 为 1，所以每次处理一张图像。
        for b_x, b_y in test_dataloader:
            # 将当前测试图像移动到模型所在设备。
            b_x = b_x.to(device)
            # 将当前图像的真实标签移动到同一设备。
            b_y = b_y.to(device)

            # 设置模型为验证模型
            model.eval()
            # 将当前图像输入模型，得到十个类别的预测分数。
            output = model(b_x)
            # 取分数最高的类别编号作为预测标签。
            pre_lab = torch.argmax(output, dim=1)
            # 将只含一个元素的预测标签张量转换为 Python 整数。
            result = pre_lab.item()
            # 将只含一个元素的真实标签张量转换为 Python 整数。
            label = b_y.item()
            # 根据类别编号输出预测类别名称和真实类别名称。
            print("预测值：", classes[result], "------", "真实值：", classes[label])
