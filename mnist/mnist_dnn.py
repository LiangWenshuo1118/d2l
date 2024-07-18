import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 定义网络结构
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 输入层到隐藏层
        self.relu = nn.ReLU()             # 激活函数
        self.fc2 = nn.Linear(512, 10)     # 隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平图像
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 数据加载和转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 下载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# 数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 实例化模型、损失函数和优化器
model = DNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train(model, criterion, optimizer, loader):
    model.train()
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试模型
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 训练和测试循环
for epoch in range(10):  # 训练10个epoch
    train(model, criterion, optimizer, train_loader)
    accuracy = test(model, test_loader)
    print(f'Epoch {epoch+1}, Accuracy: {accuracy:.2f}%')

