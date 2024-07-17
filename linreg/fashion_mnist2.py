import torch
import torchvision
from torch.utils import data
from torchvision import transforms

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

def net(X,W,b):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def sgd(params, lr, batch_size):
    with torch.no_grad():  # 停止自动梯度计算，节省计算资源和内存
        for param in params:
            param -= lr * param.grad / batch_size  # 按缩放后的梯度更新参数
            param.grad.zero_()

def evaluate_accuracy(data_iter, net):
    correct = 0
    total = 0
    for X, y in data_iter:
        with torch.no_grad():
            y_hat = net(X, W, b)
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

    # 初始化参数
    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 超参数设置
    lr = 0.1  # 学习率
    num_epochs = 10   # 训练周期数
    batch_size = 256  # 小批量大小

    # 训练模型
    for epoch in range(num_epochs):
        train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True)
        total_loss = 0
        total_number = 0

        for X, y in train_iter:
            # 计算小批量的损失
            l = cross_entropy(net(X, W, b), y)
            l.sum().backward()
            sgd([W, b], lr, batch_size)  # 使用参数的梯度更新参数
            total_loss += l.sum()
            total_number += X.shape[0]

        # 计算整个数据集上的平均损失
        avg_loss = total_loss / total_number
        print(f'epoch {epoch + 1}, average loss {float(avg_loss):f}')

    # 创建测试数据加载器
    test_iter = data.DataLoader(mnist_test, batch_size=256, shuffle=False)
    accuracy = evaluate_accuracy(test_iter, net)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
