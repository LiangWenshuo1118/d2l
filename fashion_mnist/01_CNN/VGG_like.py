import torch  # 导入PyTorch库
import torchvision  # 导入Torchvision库，用于处理图像数据集
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.utils import data  # 从PyTorch中导入数据加载模块
from torchvision import transforms  # 从Torchvision中导入数据预处理模块

# 定义一个类似VGG的神经网络模型
class VGG_like(nn.Module):
    def __init__(self):
        super(VGG_like, self).__init__()  # 调用父类的构造函数
        # 特征提取部分，包含多层卷积和池化层
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输入通道为1（灰度图像），输出通道为16，卷积核大小为3x3，padding=1保持尺寸不变
            nn.ReLU(),  # 使用ReLU激活函数
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 再次卷积，输入和输出通道都是16
            nn.ReLU(),  # 使用ReLU激活函数
            nn.MaxPool2d(2, 2),  # 最大池化层，池化窗口大小为2x2，步幅为2，减少特征图尺寸
            nn.Dropout(0.25),  # Dropout层，随机丢弃25%的神经元以防止过拟合

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 卷积层，输入通道为16，输出通道为32
            nn.ReLU(),  # 使用ReLU激活函数
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # 再次卷积，输入和输出通道都是32
            nn.ReLU(),  # 使用ReLU激活函数
            nn.MaxPool2d(2, 2),  # 最大池化层，池化窗口大小为2x2
            nn.Dropout(0.25)  # 再次使用Dropout层，随机丢弃25%的神经元
        )
        # 分类器部分，包含全连接层
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 100),  # 全连接层，输入尺寸为32*7*7（展平后的特征图），输出为100
            nn.ReLU(),  # 使用ReLU激活函数
            nn.Dropout(0.5),  # Dropout层，随机丢弃50%的神经元
            nn.Linear(100, 10)  # 最终输出层，输出大小为10（对应10个类别）
        )

    def forward(self, x):
        x = self.features(x)  # 前向传播，先通过特征提取部分
        x = x.view(x.size(0), -1)  # 将多维张量展平成二维张量 (batch_size, flatten_features)
        x = self.classifier(x)  # 通过分类器部分
        return x  # 返回最终输出

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
    print(f"Using {device} device")

    # 数据预处理，将图像转换为张量
    trans = transforms.ToTensor()
    # 加载FashionMNIST数据集
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

    batch_size = 256
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False)

    # 初始化模型并移动到设备（CPU或GPU）
    net = VGG_like().to(device)
    num_epochs = 50
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 初始化记录结果的列表
    results = []

    for epoch in range(num_epochs):
        net.train()  # 确保网络在训练模式
        total_loss = 0
        total_number = 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)  # 将数据移动到设备
            l = loss_fn(net(X), y)  # 计算损失
            trainer.zero_grad()  # 梯度清零
            l.mean().backward()  # 反向传播计算梯度
            trainer.step()  # 更新参数

            total_loss += l.sum()  # 累加损失值
            total_number += y.size(0)  # 累加样本总数

        train_loss = total_loss / total_number

        net.eval()  # 设置网络为评估模式
        test_loss, test_accuracy = evaluate_model(test_iter, net, device, loss_fn)  # 计算测试集上的损失和准确率

        results.append([epoch + 1, train_loss.item(), test_loss.item(), test_accuracy])

        # 打印当前epoch的训练损失、测试损失和测试准确率
        print(f'Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}, Test Accuracy {test_accuracy * 100:.2f}%')

    # 训练结束后，把数据写入txt文件
    with open('log/VGG_like_droupout.txt', 'w') as f:
        f.write("{:<10}{:<15}{:<15}{:<15}\n".format("Epoch", "Train Loss", "Test Loss", "Test Accuracy"))
        for data in results:
            f.write("{:<10}{:<15.4f}{:<15.4f}{:<15.2f}\n".format(*data))
