import torch  # 导入PyTorch库
import torchvision  # 导入Torchvision库，用于处理图像数据集
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.utils import data  # 从PyTorch中导入数据加载模块
from torchvision import transforms  # 从Torchvision中导入数据预处理模块

# 评估模型性能的函数，计算平均损失和准确率
def evaluate_model(data_iter, net, device, loss_fn):
    total_loss = 0  # 累加总损失
    correct = 0  # 累加正确预测的数量
    total = 0  # 累加样本总数
    with torch.no_grad():  # 不计算梯度，评估模型时用
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)  # 将数据移动到指定设备（CPU或GPU）
            y_hat = net(X)  # 前向传播，得到预测结果
            l = loss_fn(y_hat, y)  # 计算损失
            total_loss += l.sum()   # 累加损失值，乘以样本数量以获得总损失
            y_pred = torch.argmax(y_hat, axis=1)  # 找到预测值中最大概率的类别
            correct += (y_pred == y).sum().item()  # 计算正确预测的数量
            total += y.size(0)  # 累加样本总数
    avg_loss = total_loss / total  # 计算平均损失
    accuracy = correct / total  # 计算准确率
    return avg_loss, accuracy  # 返回平均损失和准确率

if __name__ == "__main__":
    # 选择设备，优先使用GPU（CUDA或MPS），否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device")  # 打印使用的设备

    # 数据预处理，将图像转换为张量
    trans = transforms.ToTensor()
    # 加载FashionMNIST数据集
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

    batch_size = 256  # 定义批量大小
    # 创建数据加载器，用于批量加载训练数据和测试数据
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False)

    # 定义模型
    # nn.Flatten()将图像从二维展平为一维
    # 线性层，784个输入特征，10个输出类别
    # 初始化模型并移动到设备（CPU或GPU）
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(device)
    num_epochs = 50  # 训练的周期数
    loss_fn = nn.CrossEntropyLoss(reduction='none')  # 损失函数，使用交叉熵损失
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)  # 优化器，使用随机梯度下降法，学习率为0.1

    for epoch in range(num_epochs):  # 循环训练，每个epoch进行一次完整的训练和测试
        net.train()  # 确保网络在训练模式
        total_loss = 0  # 初始化总损失
        total_number = 0  # 初始化样本总数
        for X, y in train_iter:  # 迭代每个批次的训练数据
            X, y = X.to(device), y.to(device)  # 将数据移动到设备
            l = loss_fn(net(X), y)  # 计算损失
            trainer.zero_grad()  # 梯度清零
            l.mean().backward()  # 反向传播计算梯度
            trainer.step()  # 更新参数

            total_loss += l.sum()  # 累加损失值
            total_number += y.size(0)  # 累加样本总数

        train_loss = total_loss / total_number  # 计算训练集上的平均损失
        net.eval()  # 设置网络为评估模式
        test_loss, test_accuracy = evaluate_model(test_iter, net, device, loss_fn)  # 计算测试集上的损失和准确率

        # 打印当前epoch的训练损失、测试损失和测试准确率
        print(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}, Test Accuracy {test_accuracy * 100:.2f}%')
