import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

    return np.concatenate([descriptors_O, descriptors_H1, descriptors_H2])

# 神经网络定义
class BPNN(nn.Module):
    def __init__(self, input_size):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

def apply_transformation(coords):
    # 平移
    translation_vector = np.array([1.0, 1.0, 1.0])
    translated_coords = coords + translation_vector

    # 旋转 (绕z轴旋转45度)
    angle = np.pi / 4  # 45 degrees
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    rotated_coords = np.dot(coords, rotation_matrix)

    # 置换
    permuted_coords = coords[[1, 0, 2]]  # Swap O and H1

    return translated_coords, rotated_coords, permuted_coords

# 主函数
def main():
    # 原子坐标
    coords = np.array([
        [0.919, -0.032, 0.087],  # 氧原子
        [1.887, -0.030, 0.041],  # 氢原子1
        [0.640, 0.002, -0.840]  # 氢原子2
    ])

    # 获取描述符
    descriptors = compute_descriptors(coords)
    descriptors_tensor = torch.tensor(descriptors, dtype=torch.float32).unsqueeze(0)

    # 网络和优化器
    net = BPNN(len(descriptors))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 假设的训练数据（能量值）
    energy = torch.tensor([[-75.0]])

    # 训练循环
    for epoch in range(5000):
        optimizer.zero_grad()
        output = net(descriptors_tensor)
        loss = criterion(output, energy)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # 测试变换后的坐标
    translated, rotated, permuted = apply_transformation(coords)
    test_coords = [translated, rotated, permuted]
    for test in test_coords:
        test_descriptors = compute_descriptors(test)
        test_descriptors_tensor = torch.tensor(test_descriptors, dtype=torch.float32).unsqueeze(0)
        test_output = net(test_descriptors_tensor)
        print(f'Transformed Output: {test_output.item()}')

if __name__ == "__main__":
    main()
