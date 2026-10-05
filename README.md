# ResNet18 FashionMNIST

这是一个使用 PyTorch 实现的 ResNet18 图像分类学习项目，训练和测试数据来自 FashionMNIST。

## 项目文件

- `Model.py`：残差块与 ResNet18 网络结构。
- `Model_Train.py`：数据集划分、模型训练、验证、最优权重保存和训练曲线绘制。
- `Model_Test.py`：加载最优权重并在测试集上计算准确率、输出预测结果。
- `Test.py`：PyTorch 与 CUDA 环境检查。
- `best_model.pth`：训练得到的最优模型参数，使用 Git LFS 存储。
- `data/FashionMNIST/raw/`：FashionMNIST 训练集与测试集原始文件，使用 Git LFS 存储。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练模型

```bash
python Model_Train.py
```

仓库已经包含 FashionMNIST 数据集；如果数据文件缺失，`download=True` 仍会自动重新下载。训练结束后，最优参数保存为 `best_model.pth`。

## 测试模型

```bash
python Model_Test.py
```

模型输入为单通道 FashionMNIST 图像，输出对应十个服装类别的分类结果。
