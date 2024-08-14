import torch
from chempy import Substance
from collections import defaultdict
from torch_geometric.data import Data
from mendeleev import element


def get_elements(system):
    """ 返回体系中所有不同的元素的原子序数 """
    element_atomic_numbers = set()
    for compound in system['system']:
        element_atomic_numbers.update(Substance.from_formula(compound).composition)
    return sorted(element_atomic_numbers)

def get_element_molar_percentages(system):
    """ 返回各个元素的摩尔百分比 """
    element_counts = defaultdict(int)
    total_moles = sum(system['mol_ratio'])

    for compound, ratio in zip(system['system'], system['mol_ratio']):
        for element, count in Substance.from_formula(compound).composition.items():
            element_counts[element] += count * ratio

    total_element_moles = sum(element_counts.values())
    return [(element_counts[element] / total_element_moles) * 100 for element in sorted(element_counts)]

def fetch_element_features(elements):
    """根据原子序数获取每个元素的特征"""
    features = []
    for atomic_number in elements:
        el = element(atomic_number)
        features.append([atomic_number, el.period, el.en_pauling])
    return features

def build_feature_matrix(elements, percentages):
    """构建特征矩阵，包含原子属性和摩尔百分比"""
    element_features = fetch_element_features(elements)
    # 将摩尔百分比添加到特征列表中
    for i, features in enumerate(element_features):
        features.append(percentages[i])
    return torch.tensor(element_features, dtype=torch.float)

def build_adjacency_matrix(elements):
    """构建邻接矩阵，目前仅考虑N-O和S-O的连接"""
    size = len(elements)
    adj_matrix = torch.zeros((size, size))
    element_symbols = [element(e).symbol for e in elements]
    for i, symbol1 in enumerate(element_symbols):
        for j, symbol2 in enumerate(element_symbols):
            if (symbol1 == 'N' and symbol2 == 'O') or (symbol1 == 'S' and symbol2 == 'O'):
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1  # 因为是无向图
    return adj_matrix

def build_graph_data(system_data):
    """从系统数据构建用于图神经网络的数据对象"""
    for key, system in system_data.items():
        elements = get_elements(system)
        percentages = get_element_molar_percentages(system)
        features = build_feature_matrix(elements, percentages)
        adjacency = build_adjacency_matrix(elements)

        # 全局特征和标签
        temperature = torch.tensor([temp[0] for temp in system['temperature_density']], dtype=torch.float)
        labels = torch.tensor([[temp[1], visc[1]] for temp, visc in zip(system['temperature_density'], system['temperature_viscosity'])], dtype=torch.float)

        # 构建PyTorch Geometric数据对象
        data_object = Data(x=features, edge_index=adjacency.nonzero(as_tuple=True), y=labels, global_attr=temperature)
        print(f"{key} Graph Data: {data_object}")



# 提供的熔盐体系数据
data = {
  "system1": {
    "system": ["KNO3", "NaNO3", "NaCl"],
    "mol_ratio": [22, 50, 28],
    "temperature_density": [
      [400, 2.20],
      [420, 2.31],
      [430, 2.40],
      [440, 2.45]
    ],
    "temperature_viscosity": [
      [400, 0.30],
      [420, 0.28],
      [430, 0.26],
      [440, 0.25]
    ]
  },
  "system2": {
    "system": ["KNO3", "NaNO3", "NaCl"],
    "mol_ratio": [22, 55, 23],
    "temperature_density": [
      [400, 2.22],
      [420, 2.33],
      [430, 2.42],
      [440, 2.47]
    ],
    "temperature_viscosity": [
      [400, 0.31],
      [420, 0.29],
      [430, 0.27],
      [440, 0.26]
    ]
  }
}

# 调用函数
for key in data:
    element_atomic_numbers = get_elements(data[key])
    element_molar_percentages = get_element_molar_percentages(data[key])

build_graph_data(data)

