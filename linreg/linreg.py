import torch
import random

# 生成数据集
def synthetic_data(w, b, num_examples):
    """
    生成线性数据集。模型为 y = Xw + b + 噪声。
    :param w: 权重向量，尺寸为 (特征数, 1)
    :param b: 偏置项
    :param num_examples: 生成的样本数量
    :return: 特征矩阵 X 和标签向量 y
    """
    # 生成正态分布的随机特征 X
    # torch.normal(mean, std, size): 生成均值为 mean，标准差为 std 的正态分布随机数，size 指定输出张量的大小
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算标签的理论值（不包含噪声）
    # torch.matmul(tensor1, tensor2): 计算两个张量的矩阵乘法。如果两个张量都是一维的，进行点积操作。
    y = torch.matmul(X, w) + b
    # 添加正态分布噪声以模拟真实场景中的测量误差
    # 噪声的生成同样使用 torch.normal，这里我们添加的噪声均值为0，标准差为0.01
    y += torch.normal(0, 0.01, y.shape)
    # 将 y 的形状从 [num_examples,] 调整为 [num_examples, 1]，这样做是为了后续处理的方便
    return X, y.reshape((-1, 1))

# 读取数据集
def data_iter(batch_size, features, labels):
    """
    生成小批量数据。
    :param batch_size: 每批的样本数，决定了每次迭代返回的数据量
    :param features: 完整的特征数据集，通常是一个二维张量
    :param labels: 完整的标签数据集，与特征集对应
    :yield: 每次迭代返回一批随机的特征和标签
    
    这个函数使用 Python 的生成器机制，每次调用时生成一个数据批次，直到整个数据集被完全遍历。
    """
    num_examples = len(features)  # 计算总样本数
    indices = list(range(num_examples))  # 创建一个索引列表，从0到num_examples-1
    random.shuffle(indices)  # 随机打乱索引，以确保数据的随机性

    # 迭代返回每个批次的数据
    for i in range(0, num_examples, batch_size):  # 从0开始，以batch_size为步长遍历索引
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])  # 获取当前批次的索引
        yield features[batch_indices], labels[batch_indices]  # 使用索引从数据集中取出对应的特征和标签，并返回这一批次的数据

# 定义模型
def linreg(X, w, b):
    """
    线性回归模型。
    :param X: 特征矩阵
    :param w: 权重向量
    :param b: 偏置项
    :return: 线性回归模型的输出
    """
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """
    均方损失。
    :param y_hat: 预测值
    :param y: 真实值
    :return: 均方损失值
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降。
    :param params: 模型参数列表，需要更新的参数，这些参数应已设置requires_grad=True
    :param lr: 学习率，控制梯度下降步骤的更新幅度
    :param batch_size: 小批量样本数量，用于梯度标准化，保持梯度的尺度不受批量大小影响
    """
    with torch.no_grad():  # 停止自动梯度计算，节省计算资源和内存
        for param in params:
            param -= lr * param.grad / batch_size  # 按缩放后的梯度更新参数
            param.grad.zero_()  # 清空累积梯度，为下一次梯度计算做准备

if __name__ == "__main__":
    # 真实的权重和偏置项，用于生成数据
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    # 生成 1000 个数据点
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[0], '\nlabel:', labels[0])

    # 初始化模型参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 超参数设置
    lr = 0.03  # 学习率
    num_epochs = 3  # 训练周期数
    batch_size = 10  # 小批量大小

    # 训练模型
    for epoch in range(num_epochs):
        # 迭代访问小批量数据，每次迭代返回一批特征和对应的标签
        for X, y in data_iter(batch_size, features, labels):
            # 计算小批量的损失
            l = squared_loss(linreg(X, w, b), y)

            # 计算损失函数 L 对模型参数 w 和 b 的梯度，梯度会被存储在参数 w 和 b 的 grad 属性中
            # 1. 首先对损失 l (均方误差) 求导，得到 l 对预测值 y_hat 的导数 (dl/dy_hat)
            # 2. 接着对预测值 y_hat 求导，得到 y_hat 对模型参数 w 和 b 的导数 (dy_hat/dw 和 dy_hat/db)
            # 3. 应用链式法则，将误差从输出层传递到输入层，计算损失 l 对参数 w 和 b 的导数 (dl/dw 和 dl/db)
            l.sum().backward()

            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数

        with torch.no_grad():
            # 计算整个数据集上的损失
            train_l = squared_loss(linreg(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
