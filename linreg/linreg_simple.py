import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的数据集"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

if __name__ == "__main__":
    # 真实的权重和偏置项
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    # 生成数据
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 封装数据
    batch_size = 10
    # 将特征和标签封装成一个数据集
    dataset = TensorDataset(features, labels)
    # DataLoader 接受一个数据集并允许你定义批量大小和是否在每个 epoch 开始时混洗数据
    data_iter = DataLoader(dataset, batch_size, shuffle=True)

    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))

    # 初始化模型参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化器
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    # 训练模型
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)  # 先计算损失：这样才能知道在当前模型参数下，模型的表现如何。
            trainer.zero_grad()  # 再清空梯度：以避免错误的梯度累积。
            l.backward()         # 执行反向传播：计算关于当前损失的梯度。
            trainer.step()       # 最后更新参数：根据计算得到的梯度优化模型参数。
        with torch.no_grad():
            l = loss(net(features), labels)  # 计算全数据集的损失
            print(f'epoch {epoch + 1}, loss {l:f}')
