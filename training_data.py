from utils import ls, load_pickle, dump_pickle, read_lines
import numpy as np
import random
import os
import gc

PROTEIN_FILENAME_SUFFIX = '_pro_cg.pdb'
LIGAND_FILENAME_SUFFIX = '_lig_cg.pdb' 

def get_training_data(
    training_data_dir_path = os.path.abspath('./training_data'),
    training_data_pkl_path = os.path.abspath('./training_data.pkl'),
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
    
    def reshape_data(data, max_x, max_y, max_z, num_channels=2):
        result = []
        for i in range(len(data)):
            reshaped = np.zeros(shape=(int(max_x), int(max_y), int(max_z), num_channels))
            complex = data[i]
            for atom in complex:
                x = int(atom[0])
                y = int(atom[1])
                z = int(atom[2])
                channel = int(atom[3])
                reshaped[x][y][z][channel] = 1
            result.append(reshaped)
        return np.array(result)

    ############################## Function body #############################
    protein_data, ligand_data, max_x, max_y, max_z = load_data(training_data_dir_path)

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

    # Reshape into shape=(x, y, z, type)
    max_x = int(max_x * 1000)
    max_y = int(max_y * 1000)
    max_z = int(max_z * 1000)
    print("Reshaping data into shape ({},{},{},{})".format(max_x, max_y, max_z, 2))
    x_protein = reshape_data(x_protein, max_x, max_y, max_z)
    x_ligand = reshape_data(x_ligand, max_x, max_y, max_z)

    return np.array(x_protein), np.array(x_ligand), np.array(y)

def load_data(dir_path):
    '''
    Args:
        dir_path (str): the training_data directory.

    Returns: 
        protein_data (list): [ [[protein_x, protein_y, protein_z, protein_type], ... ] , ... ].
        ligand_data (list): [ [[ligand_x, ligand_y, ligand_z, ligand_type], ... ] , ... ].
        Where for the same index position in the respective lists, the corresponding protein and ligand is a receptive pair.
    '''
    ############################ Helper functions ############################
    def get_index(filename):
        return filename.split('_')[0]

    def get_pair(index):
        protein_path = os.path.join(dir_path, index + PROTEIN_FILENAME_SUFFIX)
        ligand_path = os.path.join(dir_path, index + LIGAND_FILENAME_SUFFIX)
        protein, largest_protein_x, largest_protein_y, largest_protein_z = read_complex(protein_path)
        ligand, largest_ligand_x, largest_ligand_y, largest_ligand_z = read_complex(ligand_path)
        return protein, ligand, max(largest_protein_x, largest_ligand_x), max(largest_protein_y, largest_ligand_y), max(largest_protein_z, largest_ligand_z)

    def read_complex(file_path):
        # Read atom data from file
        content = [x.strip() for x in read_lines(file_path)]

        # Construct complex from atom data
        atoms = []
        max_x = min_x = max_y = min_y = max_z = min_z = 0

        for line in content:
            x = float(line[30:38].strip())
            max_x = max(x, max_x)
            min_x = min(x, min_x)

            y = float(line[38:46].strip())
            max_y = max(y, max_y)
            min_y = min(y, min_y)

            z = float(line[46:54].strip())
            max_z = max(z, max_z)
            min_z = min(z, min_z)

            atom_type = line[76:78].strip()
            hydrophobicity = 1 if atom_type == 'C' else 0 # 1 for hydrophobic, 0 for polar

            atoms.append([x, y, z, hydrophobicity])
        
        for atom in atoms:
            atom[0] += abs(min_x)
            atom[1] += abs(min_y)
            atom[2] += abs(min_z)

        return atoms, max_x + abs(min_x), max_y + abs(min_y), max_z + abs(min_z)

    ############################## Function body #############################
    protein_data = []
    ligand_data = []
    max_x = 0
    max_y = 0
    max_z = 0
    for protein_filename in ls(dir_path, lambda x: x.endswith(PROTEIN_FILENAME_SUFFIX)):
        index = get_index(protein_filename)
        protein, ligand, largest_x, largest_y, largest_z = get_pair(index)
        protein_data.append(protein)
        ligand_data.append(ligand)

        max_x = largest_x if largest_x > max_x else max_x
        max_y = largest_y if largest_y > max_y else max_y
        max_z = largest_z if largest_z > max_z else max_z

    return protein_data, ligand_data, max_x, max_y, max_z
