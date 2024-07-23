import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # 第一个卷积层，输入图像大小为（1, 28, 28）
            nn.Conv2d(
                in_channels=1,     # 输入通道数为1，因为是灰度图像
                out_channels=16,   # 输出通道数为16，表示使用16个卷积核
                kernel_size=5,     # 卷积核大小为5x5
                stride=1,          # 卷积核移动步长为1
                padding=2          # 填充为2，保持图像大小不变
            ),  # 输出图像大小: (16, 28, 28)
            nn.ReLU(),            # 使用ReLU激活函数
            nn.MaxPool2d(2)       # 使用2x2最大池化，输出图像大小减半: (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            # 第二个卷积层，输入图像大小为（16, 14, 14）
            nn.Conv2d(
                in_channels=16,    # 输入通道数为16
                out_channels=32,   # 输出通道数为32，表示使用32个卷积核
                kernel_size=5,     # 卷积核大小为5x5
                stride=1,          # 卷积核移动步长为1
                padding=2          # 填充为2，保持图像大小不变
            ),  # 输出图像大小: (32, 14, 14)
            nn.ReLU(),            # 使用ReLU激活函数
            nn.MaxPool2d(2)       # 使用2x2最大池化，输出图像大小减半: (32, 7, 7)
        )
        # 定义全连接层，输入特征数为32*7*7，输出特征数为10
        self.out = nn.Linear(32 * 7 * 7, 10)  # 进行10分类

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 将多维输入一维化
        output = self.out(x)
        return output

def evaluate_accuracy(data_iter, net):
    correct = 0
    total = 0
    for X, y in data_iter:
        with torch.no_grad():
            y_hat = net(X)
            y_pred = torch.argmax(y_hat, axis=1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)

    net = CNN()

    num_epochs = 50
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        total_loss = 0
        total_number = 0
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()

            total_loss += l.sum()
            total_number += X.shape[0]

        avg_loss = total_loss / total_number
        print(f'epoch {epoch + 1}, average loss {float(avg_loss):f}')

    test_iter = data.DataLoader(mnist_test, batch_size=256, shuffle=False)
    accuracy = evaluate_accuracy(test_iter, net)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
