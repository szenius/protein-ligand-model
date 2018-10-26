from utils import ls, load_pickle, dump_pickle, read_lines
import numpy as np
import random
import os
from matplotlib import pyplot as plt

PROTEIN_FILENAME_SUFFIX = '_pro_cg.pdb'
LIGAND_FILENAME_SUFFIX = '_lig_cg.pdb' 

def get_training_data(
    training_data_dir_path = os.path.abspath('./training_data'),
    training_data_pkl_path = os.path.abspath('./training_data_dist.pkl'),
    reprocess=False):
    '''
    Args:
        training_data_dir_path (str):  the training_data directory.
        training_data_pkl_path (str):  the serialized training data file path.
        reprocess (boolean):  whether or not tp ignore serialized training data and reprocessed raw training data.

    Returns: 
        training data (list): [protein data, ligand data, labels]
    '''
    # If reprocessing training data or pkl does not exist
    if reprocess or not os.path.exists(training_data_pkl_path):
        training_data = generate_training_data(training_data_dir_path)
        dump_pickle(training_data_pkl_path, training_data) # save data to disk
        return training_data
    # Else load from previously prepared training data
    else:
        return load_pickle(training_data_pkl_path)

def generate_training_data(training_data_dir_path):
    '''
    Args:
        training_data_dir_path (str): the training_data directory.

    Returns: 
        x_protein (3d np array): training data for protein features.
        x_ligand (3d np array): training data for ligand features.
        y (1d np array): groundtruth labels.
        Where for the same index position in the respective lists, the corresponding triplet is a training example.
    '''
    ############################ Helper functions ############################
    def generate_negative_examples(protein_data, ligand_data, num=1):
        """Generate num negative pairings for each protein"""

        def random_neg_ligand(avoid_index):
            """Returns a random ligand of non-matching index"""
            wrong_index = avoid_index
            while wrong_index == avoid_index:
                wrong_index = random.randint(0, len(ligand_data) - 1)
            return ligand_data[wrong_index]

        x_neg_protein = []
        x_neg_ligand = []
        y_neg = [0] * len(protein_data)
        for i in range(len(protein_data)):
            protein = protein_data[i]
            neg_ligand = random_neg_ligand(i)
            x_neg_protein.append(protein)
            x_neg_ligand.append(neg_ligand)
        return x_neg_protein, x_neg_ligand, y_neg

    def euclidean_distance(v1, v2):
        return np.linalg.norm(np.array(v1) - np.array(v2))

    def generate_distances_for_complex(protein, ligand, max_length):
        distances = []
        for i in range(min(len(protein), len(ligand))):
            distances.append(euclidean_distance(protein[i][:-1], ligand[i][:-1]))
        for i in range(len(distances), max_length):
            distances.append(0)
        return distances
    
    ############################## Function body #############################
    protein_data, ligand_data, max_length = load_data(training_data_dir_path)
    print("Loaded from training data files.")

    # Positive examples
    x_pos_protein = protein_data
    x_pos_ligand = ligand_data
    y_pos = [1] * len(protein_data)

    # Negative examples
    x_neg_protein, x_neg_ligand, y_neg = generate_negative_examples(protein_data, ligand_data)

    # Concat to form training data
    # Note: shuffling is left to the training process
    x_protein = x_pos_protein + x_neg_protein
    x_ligand = x_pos_ligand + x_neg_ligand
    y = y_pos + y_neg
    print("Generated train set for protein and ligand.")

    # Generate distances
    x_distance = []
    for i in range(len(x_protein)):
        print("Generating atom distances for protein-ligand pair", i, "/", len(x_protein))
        x_distance.append(generate_distances_for_complex(x_protein[i], x_ligand[i], max_length))

    return np.array(x_protein), np.array(x_ligand), np.array(x_distance), np.array(y)

def load_data(dir_path):
    '''
    Args:
        dir_path (str): the training_data directory.

    Returns: 
        protein_data (list): [[protein_x_list, protein_y_list, protein_z_list, protein_type_list], ... ].
        ligand_data (list): [[ligand_x_list, ligand_y_list, ligand_z_list, ligand_type_list], ... ].
        Where for the same index position in the respective lists, the corresponding protein and ligand is a receptive pair.
    '''
    ############################ Helper functions ############################
    def get_index(filename):
        return filename.split('_')[0]

    def get_pair(index):
        protein_path = os.path.join(dir_path, index + PROTEIN_FILENAME_SUFFIX)
        ligand_path = os.path.join(dir_path, index + LIGAND_FILENAME_SUFFIX)
        protein = read_complex(protein_path)
        ligand = read_complex(ligand_path)
        return protein, ligand, max(len(protein), len(ligand))

    def read_complex(file_path):
        # Read atom data from file
        content = [x.strip() for x in read_lines(file_path)]

        # Construct complex from atom data
        atoms = []

        for line in content:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            atom_type = line[76:78].strip()
            hydrophobicity = 2 if atom_type == 'C' else 1 # 2 for hydrophobic, 1 for polar
            atoms.append([x, y, z, hydrophobicity])

        return atoms

    ############################## Function body #############################
    protein_data = []
    ligand_data = []
    max_length = 0
    for protein_filename in ls(dir_path, lambda x: x.endswith(PROTEIN_FILENAME_SUFFIX)):
        index = get_index(protein_filename)
        protein, ligand, length = get_pair(index)
        protein_data.append(protein)
        ligand_data.append(ligand)
        max_length = length if length > max_length else max_length
    return protein_data, ligand_data, max_length
