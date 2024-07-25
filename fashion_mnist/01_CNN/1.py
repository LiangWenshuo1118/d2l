import torch  # 导入PyTorch库
import torchvision  # 导入Torchvision库，用于处理图像数据集
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.utils import data  # 从PyTorch中导入数据加载模块
from torchvision import transforms  # 从Torchvision中导入数据预处理模块
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, padding=0, stride=strides) # padding默认值是0
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
             X = self.conv3(X)
             Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

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

    # 输入1*28*28 > 输出16 * 28 * 28
    b1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=3), 
    nn.BatchNorm2d(16), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2) # 输入16 * 28 * 28 > 输出16 * 14 * 14
    )
    # 输入16 * 14 * 14 > 输出 16 * 14 * 14
    b2 = nn.Sequential(*resnet_block(16, 16, 2, first_block=True))
    # 输入16 * 14 * 14 > 输出 32 * 7 * 7
    b3 = nn.Sequential(*resnet_block(16, 32, 2))
    net = nn.Sequential(b1, b2,
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(), 
        nn.Linear(32, 10)
        ).to(device)

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
    with open('log/VGG_like_dropout.txt', 'w') as f:
        f.write("{:<10}{:<15}{:<15}{:<15}\n".format("Epoch", "Train Loss", "Test Loss", "Test Accuracy"))
        for data in results:
            f.write("{:<10}{:<15.4f}{:<15.4f}{:<15.2f}\n".format(*data))
