from utils import ls, load_pickle, dump_pickle, read_lines
import numpy as np
import random
import os
import sys

PROTEIN_FILENAME_SUFFIX = '_pro_cg.pdb'
LIGAND_FILENAME_SUFFIX = '_lig_cg.pdb' 

def reshape_data(data, max_x, max_y, max_z, num_channels=2, desc=''):
    data_len = len(data)
    x_len = max_x + 1
    y_len = max_y + 1
    z_len = max_z + 1
    reshaped_data = np.zeros((len(data), x_len, y_len, z_len, num_channels))
    for i in tqdm(range(data_len), desc=desc):
        complex_seq = data[i]
        for atom in complex_seq:
            x = int(atom[0])
            y = int(atom[1])
            z = int(atom[2])
            channel = int(atom[3])
            reshaped_data[i][x][y][z][channel] = 1
    return reshaped_data

def load_data_3d(dir_path):
    '''
    Args:
        dir_path (str): the training_data directory.
        size of training examples in dir_path to load. 0 indicates all

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
        protein, pro_max_x, pro_max_y, pro_max_z = read_complex(protein_path)
        ligand,  lig_max_x, lig_max_y, lig_max_z = read_complex(ligand_path)

        # Resolve coordinate maxes for this pair
        max_x = max(pro_max_x, lig_max_x)
        max_y = max(pro_max_y, lig_max_y)
        max_z = max(pro_max_z, lig_max_z)
        return protein, ligand, max_x, max_y, max_z

    def read_complex(file_path):
        # Construct complex from atom data
        complex_seq = []
        for line in [x.strip() for x in read_lines(file_path)]:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            atom_type = line[76:78].strip()
            hydrophobicity = 1 if atom_type == 'C' else 0 # 1 for hydrophobic, 0 for polar
            complex_seq.append([x, y, z, hydrophobicity])
        
        np_complex_seq = np.asarray(complex_seq)
        xyzh_mins = np.amin(np_complex_seq, axis=0)                 # get all mins
        xyzh_mins[-1] = 0                                           # force min of hydrophobicity to be 0 (just in case)
        np_complex_seq -= xyzh_mins                                 # translate values such that min is now zero
        max_x, max_y, max_z, _ = np.amax(np_complex_seq, axis=0)    # the new maxes for x, y, z
        complex_seq = np_complex_seq.tolist()                       # convert back to python list
        return complex_seq, max_x, max_y, max_z

    ############################## Function body #############################
    protein_data = []
    ligand_data = []
    global_max_x = 0
    global_max_y = 0
    global_max_z = 0
    protein_filenames_list = ls(dir_path, lambda x: x.endswith(PROTEIN_FILENAME_SUFFIX))

    for protein_filename in tqdm(protein_filenames_list, desc='Reading {} pair complexes from {}'.format(end-start if start != -1 and end != -1 else 'all', dir_path)):
        index = get_index(protein_filename)
        protein, ligand, max_x, max_y, max_z = get_pair(index)
        protein_data.append(protein)
        ligand_data.append(ligand)

        # Update global maxes
        global_max_x = max(max_x, global_max_x)
        global_max_y = max(max_y, global_max_y)
        global_max_z = max(max_z, global_max_z)

    return protein_data, ligand_data, global_max_x, global_max_y, global_max_z

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
            splitted_line = line.strip().split('\t')
            x = float(splitted_line[0])
            y = float(splitted_line[1])
            z = float(splitted_line[2])
            atom_type = splitted_line[3].strip()
            hydrophobicity = 2 if atom_type == 'h' else 1 # 2 for hydrophobic, 1 for polar
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
    result = [0,0,0,0,0,0]
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
    empty_row = [0,0,0,0,0,0]
    for i in range(len(distances_lstm), max_length):
        distances_lstm.append(empty_row)
        distances_mlp.extend(empty_row)
    return distances_mlp, distances_lstm

def generate_ij_distances(protein, ligand):
    empty_row = [0,0,0,0,0,0]
    distances_lstm = []
    distances_mlp = []
    for i in range(len(protein)):
        for j in range(len(ligand)):
            row = atom_vector(protein[i], ligand[j])
            distances_lstm.append(row)
            distances_mlp.extend(row)
    return distances_lstm, distances_mlp
