import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=False):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        if use_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return F.relu(self.main_path(x) + self.shortcut(x))

def make_layers(in_channels, out_channels, num_blocks, first_block=False):
    layers = []
    for i in range(num_blocks):
        if i == 0 and not first_block:
            layers.append(ResidualBlock(in_channels, out_channels, stride=2, use_1x1conv=True))
        else:
            layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)

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

    # 使用简化的模型搭建方式
    b1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 16 x 14 x 14
        nn.Dropout(0.25)
    )

    b2 = make_layers(16, 16, 1, first_block=True)  # 输出: 16 x 14 x 14
    b3 = make_layers(16, 32, 1)  # 输出: 32 x 7 x 7
    b4 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(100, 10)
    )

    net = nn.Sequential(b1, b2, b3,b4).to(device)

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
