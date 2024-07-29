import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dpdata
import matplotlib.pyplot as plt

# 高斯型描述符转换函数
def gaussian_transform(r, eta, Rs):
    return np.exp(-eta * (r - Rs) ** 2)

def gaussian_transform_angle(theta, zeta, Theta_s):
    return np.exp(-zeta * (np.radians(theta) - np.radians(Theta_s)) ** 2)

# 计算所有描述符
def compute_descriptors(coords):
    # 从坐标中提取原子坐标
    O, H1, H2 = coords

    # 计算距离
    d_OH1 = np.linalg.norm(O - H1)
    d_OH2 = np.linalg.norm(O - H2)
    d_H1H2 = np.linalg.norm(H1 - H2)

    # 计算角度
    vec_OH1 = H1 - O
    vec_OH2 = H2 - O
    cos_theta_O = np.dot(vec_OH1, vec_OH2) / (np.linalg.norm(vec_OH1) * np.linalg.norm(vec_OH2))
    theta_HOH_O = np.arccos(cos_theta_O) * (180 / np.pi)  # 氧为中心角度

    vec_H1O = O - H1
    vec_H1H2 = H2 - H1
    cos_theta_H1 = np.dot(vec_H1O, vec_H1H2) / (np.linalg.norm(vec_H1O) * np.linalg.norm(vec_H1H2))
    theta_HOH_H1 = np.arccos(cos_theta_H1) * (180 / np.pi)  # 氢1为中心角度

    vec_H2O = O - H2
    vec_H2H1 = H1 - H2
    cos_theta_H2 = np.dot(vec_H2O, vec_H2H1) / (np.linalg.norm(vec_H2O) * np.linalg.norm(vec_H2H1))
    theta_HOH_H2 = np.arccos(cos_theta_H2) * (180 / np.pi)  # 氢2为中心角度

    eta = 10.0
    Rs_values = [0.5, 1.0, 1.5]
    zeta = 2.0
    Theta_s_values = [90, 120, 180]

    # 计算描述符
    descriptors_O = [gaussian_transform(d_OH1, eta, Rs) for Rs in Rs_values] + \
                    [gaussian_transform(d_OH2, eta, Rs) for Rs in Rs_values] + \
                    [gaussian_transform_angle(theta_HOH_O, zeta, Theta_s) for Theta_s in Theta_s_values]

    descriptors_H1 = [gaussian_transform(d_OH1, eta, Rs) for Rs in Rs_values] + \
                     [gaussian_transform(d_H1H2, eta, Rs) for Rs in Rs_values] + \
                     [gaussian_transform_angle(theta_HOH_H1, zeta, Theta_s) for Theta_s in Theta_s_values]

    descriptors_H2 = [gaussian_transform(d_OH2, eta, Rs) for Rs in Rs_values] + \
                     [gaussian_transform(d_H1H2, eta, Rs) for Rs in Rs_values] + \
                     [gaussian_transform_angle(theta_HOH_H2, zeta, Theta_s) for Theta_s in Theta_s_values]

    return descriptors_O, descriptors_H1, descriptors_H2

# 分子神经网络，包含三个子网络
class MolecularNN(nn.Module):
    def __init__(self, input_size, hidden_size=10):
        super(MolecularNN, self).__init__()
        self.subnet_O = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.subnet_H1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.subnet_H2 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x_O, x_H1, x_H2 = x
        energy_O = self.subnet_O(x_O)
        energy_H1 = self.subnet_H1(x_H1)
        energy_H2 = self.subnet_H2(x_H2)
        total_energy = energy_O + energy_H1 + energy_H2
        return total_energy

def main():
    # 使用 dpdata 读取 OUTCAR 文件
    system = dpdata.LabeledSystem('OUTCAR', fmt='vasp/outcar')

    # 提取坐标和能量
    energies = system['energies']
    coords = system['coords']
    #forces = system['forces']  # 加载受力数据

    descriptors = np.array([compute_descriptors(frame) for frame in coords])

    descriptors_O = [desc[0] for desc in descriptors]
    descriptors_H1 = [desc[1] for desc in descriptors]
    descriptors_H2 = [desc[2] for desc in descriptors]

    # 转换为张量
    descriptors_O = torch.tensor(descriptors_O, dtype=torch.float32)
    descriptors_H1 = torch.tensor(descriptors_H1, dtype=torch.float32)
    descriptors_H2 = torch.tensor(descriptors_H2, dtype=torch.float32)
    energies = torch.tensor(energies[:, None], dtype=torch.float32)

    # 数据划分
    n_samples = len(energies)
    n_train = int(n_samples * 0.8)
    indices = torch.randperm(n_samples)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_descriptors_O = descriptors_O[train_indices]
    train_descriptors_H1 = descriptors_H1[train_indices]
    train_descriptors_H2 = descriptors_H2[train_indices]
    train_energies = energies[train_indices]

    test_descriptors_O = descriptors_O[test_indices]
    test_descriptors_H1 = descriptors_H1[test_indices]
    test_descriptors_H2 = descriptors_H2[test_indices]
    test_energies = energies[test_indices]

    # 网络和优化器
    input_size = descriptors_O.shape[1]
    net = MolecularNN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(400000):
        optimizer.zero_grad()
        outputs = net((train_descriptors_O, train_descriptors_H1, train_descriptors_H2))
        loss = criterion(outputs, train_energies)
        loss.backward()
        optimizer.step()
        if epoch % 40000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # 测试模型
    net.eval()
    with torch.no_grad():
        test_outputs = net((test_descriptors_O, test_descriptors_H1, test_descriptors_H2))
        test_loss = criterion(test_outputs, test_energies)
        print(f'Test Loss: {test_loss.item()}')

        # 将预测和实际值转换为 NumPy 数组用于绘图
        predicted = test_outputs.numpy().flatten()
        actual = test_energies.numpy().flatten()

        # 绘制散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(actual, predicted, alpha=0.5, label='Predicted vs. Actual')
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', label='Ideal')
        plt.xlabel('Actual Energy')
        plt.ylabel('Predicted Energy')
        plt.title('Predicted vs Actual Energies Correlation')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
