import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


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
    # 下载数据
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)

    batch_size = 256  # 小批量大小
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)

    # 定义模型
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)

    # 超参数设置
    num_epochs = 10  # 训练周期数

    # 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none')

    # 定义优化器
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练模型
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

        # 计算整个数据集上的平均损失
        avg_loss = total_loss / total_number
        print(f'epoch {epoch + 1}, average loss {float(avg_loss):f}')

    # 创建测试数据加载器
    test_iter = data.DataLoader(mnist_test, batch_size=256, shuffle=False)
    accuracy = evaluate_accuracy(test_iter, net)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
