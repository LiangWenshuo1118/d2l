import torch
from torch import nn
from torch.utils import data

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的数据集"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个PyTorch数据迭代器。此函数接受数据和标签，将它们封装成一个可迭代的 DataLoader，从而允许在训练模型时进行批量处理和可选的数据混洗。

    :param data_arrays: 一个包含特征和标签的元组。例如，(features, labels)。
                        - `features` 是一个张量，包含了输入特征。
                        - `labels` 是一个张量，包含了与特征对应的标签。
    :param batch_size: 指定每个数据批次的大小。这是每次迭代训练时网络将处理的样本数量。
    :param is_train: 一个布尔值，指示是否在迭代时对数据进行随机洗牌。在训练模型时通常设置为 True 以增加随机性和提高模型泛化能力；在评估模型时通常设置为 False。

    :return: 返回一个 DataLoader 对象，它是一个迭代器，能够按照指定的批次大小(batch_size)和是否混洗(shuffle)来批量提供数据。
    """
    # 将传入的特征和标签数组封装成 TensorDataset 对象，它是一个包含张量的数据集，可以用于 DataLoader。
    # *data_arrays 使用星号表达式来解包参数列表，使得函数可以接受任意数量的数据张量作为输入。
    dataset = data.TensorDataset(*data_arrays)

    # 创建 DataLoader 对象，它是 PyTorch 中的一种数据迭代器，用于按照指定的批量大小和是否混洗来加载数据集。
    # DataLoader 的 shuffle 参数控制是否在每个 epoch 开始时对数据进行随机洗牌。
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == "__main__":
    # 真实的权重和偏置项
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    # 生成数据
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 定义模型
    # nn.Sequential: 顺序容器。模块将按照它们在构造函数中传递的顺序添加到其中，每个模块的输出将作为下一个模块的输入。
    # nn.Linear: 应用一个线性变换到输入数据 y = xA^T + b，参数解释:
    # 2 - 输入特征的数量。在这个例子中，每个输入样本是一个包含2个特征的向量。
    # 1 - 输出特征的数量。这意味着对于每个输入向量，模型将输出一个标量值。
    # 这个结构非常适合执行简单的线性回归任务，其中我们试图学习一个从输入特征到一个输出值的映射。
    net = nn.Sequential(nn.Linear(2, 1))

    # 初始化模型参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义批量大小
    batch_size = 10

    # 加载数据
    data_iter = load_array((features, labels), batch_size)
    
    # 定义优化器
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    # 训练模型
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)  # 计算损失
            trainer.zero_grad()  # 清空梯度
            l.backward()         # 计算梯度
            trainer.step()       # 更新参数
        with torch.no_grad():
            l = loss(net(features), labels)  # 计算全数据集的损失
            print(f'epoch {epoch + 1}, loss {l:f}')
