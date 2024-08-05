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
