from utils import ls, load_pickle, dump_pickle, read_lines
import numpy as np
import random
import os
import sys

PROTEIN_FILENAME_SUFFIX = '_pro_cg.pdb'
LIGAND_FILENAME_SUFFIX = '_lig_cg.pdb' 

def get_training_data(
    training_data_dir_path = os.path.abspath('./training_data'),
    training_data_pkl_path = os.path.abspath('./training_data_dist.pkl'),
    add_negative_examples=True,
    reprocess=False,
    size=512):
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
        training_data = generate_training_data(training_data_dir_path, size, add_negative_examples)
        dump_pickle(training_data_pkl_path, training_data) # save data to disk
        return training_data
    # Else load from previously prepared training data
    else:
        return load_pickle(training_data_pkl_path)

def generate_training_data(training_data_dir_path, size, add_negative_examples):
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

        neg_samples_per_protein = 1
        x_neg_protein = []
        x_neg_ligand = []
        y_neg = [0] * len(protein_data) * neg_samples_per_protein
        for i in range(len(protein_data)):
            for i in range(neg_samples_per_protein):
                protein = protein_data[i]
                neg_ligand = random_neg_ligand(i)
                x_neg_protein.append(protein)
                x_neg_ligand.append(neg_ligand)
        return x_neg_protein, x_neg_ligand, y_neg

    def euclidean_distance(v1, v2):
        return np.linalg.norm(np.array(v1) - np.array(v2))
    
    def type_index(type1, type2):
        min_type = min(type1, type2)
        max_type = max(type1, type2)
        if min_type == max_type:
            index = min_type
        else:
            index = min_type + max_type + 2
        return index

    def atom_vector(atom1, atom2=[0,0,0,0]):
        result = [0,0,0,0,0]
        ed = euclidean_distance(atom1[:-1], atom2[:-1])
        result[type_index(atom1[-1], atom2[-1])] = ed
        return result

    def generate_seq_distances(protein, ligand, max_length):
        distances_mlp = []
        distances_lstm = []

        # Get distance vectors between each atom in sequential order
        min_num = min(len(protein), len(ligand))
        for i in range(min_num):
            row = atom_vector(protein[i], ligand[i])
            distances_lstm.append(row)
            distances_mlp.extend(row)
        
        # For uneven lengths of protein and ligand atoms, pad till max of either length.
        for i in range(min_num, len(protein)):
            row = atom_vector(protein[i])
            distances_lstm.append(row)
            distances_mlp.extend(row)
        for i in range(min_num, len(ligand)):
            row = atom_vector(ligand[i])
            distances_lstm.append(row)
            distances_mlp.extend(row)

        # Pad with empty values to max length
        empty_row = [0,0,0,0,0]
        for i in range(len(distances_lstm), max_length):
            distances_lstm.append(empty_row)
            distances_mlp.extend(empty_row)
        return distances_mlp, distances_lstm
    
    def generate_ij_distances(protein, ligand):
        empty_row = [0,0,0,0,0]
        distances_lstm = []
        distances_mlp = []
        for i in range(len(protein)):
            for j in range(len(ligand)):
                row = atom_vector(protein[i], ligand[j])
                distances_lstm.append(row)
                distances_mlp.extend(row)
        return distances_lstm, distances_mlp
    
    ############################## Function body #############################
    protein_data, ligand_data, max_length = load_data(training_data_dir_path, size)
    print("Loaded from training data files.")

    # Positive examples
    x_pos_protein = protein_data
    x_pos_ligand = ligand_data
    y_pos = [1] * len(protein_data)

    # Negative examples
    if add_negative_examples:
        x_neg_protein, x_neg_ligand, y_neg = generate_negative_examples(protein_data, ligand_data)

    # Concat to form training data
    # Note: shuffling is left to the training process
    x_protein = x_pos_protein + x_neg_protein
    x_ligand = x_pos_ligand + x_neg_ligand
    y = y_pos + y_neg
    print("Generated train set for protein and ligand.")

    # Generate sequential distances
    x_seq_dist_lstm = []
    x_seq_dist_mlp = []
    for i in range(len(x_protein)):
        print("Generating seq distances for protein-ligand pair", i + 1, "/", len(x_protein))
        seq_dist_mlp, seq_dist_lstm = generate_seq_distances(x_protein[i], x_ligand[i], max_length)
        x_seq_dist_mlp.append(seq_dist_mlp)
        x_seq_dist_lstm.append(seq_dist_lstm)

    # Generate ij distances
    x_ij_dist = []
    x_ij_dist_rev = []
    x_ij_dist_flattened = []
    max_ij_length = 0
    max_ij_flattened_length = 0
    for i in range(len(x_protein)):
        print("Generating ij distances for protein-ligand pair", i + 1, "/", len(x_protein))
        ij_distance, ij_distance_flattened = generate_ij_distances(x_protein[i], x_ligand[i])
        max_ij_length = max(max_ij_length, len(ij_distance))
        max_ij_flattened_length = max(max_ij_flattened_length, len(ij_distance_flattened))
        x_ij_dist.append(ij_distance)
        x_ij_dist_rev.append(list(reversed(ij_distance)))
        x_ij_dist_flattened.append(ij_distance_flattened)
    for i in range(len(x_ij_dist)):
        for j in range(len(x_ij_dist[i]), max_ij_length):
            x_ij_dist[i].append([0,0,0,0,0])
            x_ij_dist_rev[i].append([0,0,0,0,0])
    for i in range(len(x_ij_dist_flattened)):
        for j in range(len(x_ij_dist_flattened[i]), max_ij_flattened_length):
            x_ij_dist_flattened[i].append(0)

    return np.array(x_seq_dist_lstm), np.array(x_seq_dist_mlp), np.array(x_ij_dist), np.array(x_ij_dist_rev), np.array(x_ij_dist_flattened), np.array(y)

def load_data(dir_path, size):
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
    num_read = 0
    for protein_filename in ls(dir_path, lambda x: x.endswith(PROTEIN_FILENAME_SUFFIX)):
        if size != -1 and num_read >= size:
            break
        index = get_index(protein_filename)
        protein, ligand, length = get_pair(index)
        protein_data.append(protein)
        ligand_data.append(ligand)
        max_length = length if length > max_length else max_length
        num_read += 1
    return protein_data, ligand_data, max_length
