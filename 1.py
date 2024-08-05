from mendeleev import element

# 获取元素锂
li = element('Li')

# 打印锂的所有相关属性
print(f"Atomic Number: {li.atomic_number}")
print(f"Period: {li.period}")
print(f"Electronegativity(Pauling): {li.en_pauling}")
print(f"Electron Affinity: {li.electron_affinity}")
print(f"Atomic Volume: {li.atomic_volume}")
print(f"Atomic Weight: {li.atomic_weight}")
print(f"Dipole Polarizability: {li.dipole_polarizability}")
print(f"Fusion Heat: {li.fusion_heat}")
print(f"First Ionization Energy: {li.ionenergies.get(1)}")
print(f"Covalent Radius(Bragg): {li.covalent_radius_bragg}")

Li, K, Cl = element(["Li", "K", "Cl"])
print(Li.name,K.name,Cl.name)


# 定义化合物的字典
compounds = {
    "LiCl": [1, 0, 0, 1, 0],
    "NaCl": [0, 1, 0, 1, 0],
    "KCl":  [0, 0, 0, 1, 1],
    "MgCl2":[0, 0, 1, 2, 0]
}

system = ["LiCl", "NaCl", "MgCl2", "KCl"]
percentages = [5.7, 0, 94.3, 0]  # LiCl 和 MgCl2 的摩尔百分比

# 初始化一个包含五个元素的列表，对应 Li, Na, Mg, Cl, K
weighted_sum = [0] * 5
for compound, percent in zip(system, percentages):
    weighted_sum = [sum(x) for x in zip(weighted_sum, [x * percent for x in compounds[compound]])]

# 计算每个元素的百分比
total_moles = sum(weighted_sum)
percentages_of_elements = [x / total_moles * 100 for x in weighted_sum]

print(percentages_of_elements)