import numpy as np
import pandas as pd
from mendeleev import element

def create_element_features_matrix():
    elements_list = ["Li", "Na", "Mg", "Cl", "K"]
    element_features_matrix = []
    for e in elements_list:
        elem = element(e)
        properties = [
            elem.atomic_number,
            elem.period,
            elem.en_pauling,
            elem.atomic_volume,
            elem.atomic_weight,
            elem.dipole_polarizability,
            elem.fusion_heat,
            elem.ionenergies.get(1) if 1 in elem.ionenergies else None,
            elem.covalent_radius_bragg
        ]
        element_features_matrix.append(properties)

    # Convert list to numpy array
    element_features_matrix = np.array(element_features_matrix)

    return element_features_matrix

def calculate_features(element_features_matrix, system, percentages):
    compounds = {
        "LiCl": [1, 0, 0, 1, 0],
        "NaCl": [0, 1, 0, 1, 0],
        "KCl":  [0, 0, 0, 1, 1],
        "MgCl2":[0, 0, 1, 2, 0]
    }

    weighted_sum = [0] * 5
    for compound, percent in zip(system, percentages):
        weighted_sum = [sum(x) for x in zip(weighted_sum, [x * percent for x in compounds[compound]])]

    total_moles = sum(weighted_sum)
    percentages_of_elements = [x / total_moles for x in weighted_sum]  # Convert to percentage

    # Convert percentages to a numpy column vector
    percentage_matrix = np.array(percentages_of_elements).reshape(-1, 1)
    #print(percentage_matrix)

    weighted_sum_element_features_matrix = np.dot(percentage_matrix.T, element_features_matrix)

    return weighted_sum_element_features_matrix


if __name__ == '__main__':
    system = ["LiCl", "NaCl", "MgCl2", "KCl"]
    element_features_matrix = create_element_features_matrix()
    percentages_list = [[94.3, 0.0, 5.7, 0.0],
                        [82.0, 0.0, 18.0, 0.0],
                        [69.2, 0.0, 30.8, 0.0],
                        [54.6, 0.0, 45.4, 0.0],
                        [43.6, 0.0, 56.4, 0.0],
                        [22.6, 0.0, 77.4, 0.0],
                        [0.0, 17.7, 82.3, 0.0],
                        [0.0, 37.7, 62.3, 0.0],
                   ]
    temp_dens_pairs_list = [[(1030, 1.494),(1040, 1.485),(1050, 1.476),(1060, 1.468),(1070, 1.459),(1080, 1.450),(1090, 1.441),(1100, 1.432),(1110, 1.423),(1120, 1.414)],
        [(980, 1.55), (990, 1.545), (1000, 1.54), (1010, 1.535), (1020, 1.53),(1030, 1.525),(1040, 1.52), (1050, 1.515), (1060, 1.51), (1070, 1.505), (1080, 1.5)],
        [(1020, 1.577), (1030, 1.572), (1040, 1.566), (1050, 1.561), (1060, 1.555), (1070, 1.55),(1080, 1.544), (1090, 1.539), (1100, 1.533), (1110, 1.528), (1120, 1.522)],
        [(980, 1.637), (990, 1.632), (1000, 1.627), (1010, 1.623), (1020, 1.618), (1030, 1.614),(1040, 1.609), (1050, 1.604), (1060, 1.6), (1070, 1.595), (1080, 1.59)],
        [(980, 1.669), (990, 1.664), (1000, 1.659), (1010, 1.654), (1020, 1.649), (1030, 1.644), (1040, 1.639), (1050, 1.634), (1060, 1.629), (1070, 1.624)],
        [(970, 1.705), (980, 1.701), (990, 1.697), (1000, 1.692), (1010, 1.688), (1020, 1.683),(1030, 1.679)],
        [(1020, 1.698), (1030, 1.693), (1040, 1.689), (1050, 1.684), (1060, 1.680), (1070, 1.675),(1080, 1.671), (1090, 1.667), (1100, 1.662), (1110, 1.658)],
        [(1030, 1.674), (1040, 1.669), (1050, 1.664), (1060, 1.659), (1070, 1.653), (1080, 1.648), (1090, 1.643), (1100, 1.638), (1110, 1.633), (1120, 1.628)]

                            ]

    full_data_matrix = []
    for percentages,temp_dens_pairs in zip(percentages_list,temp_dens_pairs_list):
        new_element_features_matrix = calculate_features(element_features_matrix, system, percentages)
        for temp, density in temp_dens_pairs:
            full_features = np.append(new_element_features_matrix, [temp, density])
            full_data_matrix.append(full_features)

    full_data_matrix = np.array(full_data_matrix)

    df = pd.DataFrame(full_data_matrix)
    df.to_csv('training_dataset.csv', index=False)
