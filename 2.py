from chempy import Substance
from collections import defaultdict

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

# 提供的熔盐体系数据
data = {
    "system1": {
        "system": ["KNO3", "NaNO3", "NaCl"],
        "mol_ratio": [22, 50, 28],
    },
    "system2": {
        "system": ["LiNO3", "NaNO3", "NaCl"],
        "mol_ratio": [22, 50, 28],
    }
}

# 调用函数
for key in data:  # 循环遍历每个体系
    element_atomic_numbers = get_elements(data[key])
    element_molar_percentages = get_element_molar_percentages(data[key])

    print(f"{key} - Element Atomic Numbers: {element_atomic_numbers}, Element Molar Percentages: {element_molar_percentages}")
