from utils import ls, load_pickle, dump_pickle, read_lines
from tqdm import tqdm
import numpy as np
import random
import os
import gc

PROTEIN_FILENAME_SUFFIX = '_pro_cg.pdb'
LIGAND_FILENAME_SUFFIX = '_lig_cg.pdb' 

def generate_training_data_lists(dir_path = os.path.abspath('./training_data')):
    # Populate positive examples
    x_pos_pro_list = []
    x_pos_lig_list = []
    y_pos_list = []
    for protein_filename in get_protein_filenames(dir_path):
        index = get_index(protein_filename)
        protein_filepath = get_protein_filepath(dir_path, index)
        ligand_filepath = get_ligand_filepath(dir_path, index)
        x_pos_pro_list.append(protein_filepath)
        x_pos_lig_list.append(ligand_filepath)
        y_pos_list.append(1)
    
    # Generate negative examples
    x_neg_pro_list, x_neg_lig_list, y_neg_list = generate_negative_pairings(x_pos_pro_list, x_pos_lig_list)

    # Concatenate positive and negative
    x_pro_list = x_pos_pro_list + x_neg_pro_list
    x_lig_list = x_pos_lig_list + x_neg_lig_list
    y_list = y_pos_list + y_neg_list

    # Cast to numpy
    x_pro_list = np.asarray(x_pro_list).reshape(len(x_pro_list), 1)
    x_lig_list = np.asarray(x_lig_list).reshape(len(x_lig_list), 1)
    y_list = np.asarray(y_list)

    # Shuffle data and return
    x_pro_list, x_lig_list, y_list = shuffle_data(x_pro_list, x_lig_list, y_list)

    x_list = np.concatenate((x_pro_list, x_lig_list), axis=1)
    return x_list, y_list

def load_batch(batch_x):
    batch_size = len(batch_x)
    protein_data = []
    ligand_data = []
    global_max_x = 0
    global_max_y = 0
    global_max_z = 0
    
    for protein_path, ligand_path in batch_x:
        protein, ligand, max_x, max_y, max_z = get_pair(protein_path, ligand_path)
        protein_data.append(protein)
        ligand_data.append(ligand)

        # Update global maxes
        global_max_x = max(max_x, global_max_x)
        global_max_y = max(max_y, global_max_y)
        global_max_z = max(max_z, global_max_z)

    target_shape = (batch_size, int(global_max_x) + 1, int(global_max_y) + 1, int(global_max_z) + 1, 2)
    return format_data(protein_data, target_shape), format_data(ligand_data, target_shape) 

def format_data(data, target_shape):
    reshaped_data = np.zeros(target_shape)
    for i in range(len(data)):
        complex_seq = data[i]
        for atom in complex_seq:
            x = int(atom[0])
            y = int(atom[1])
            z = int(atom[2])
            channel = int(atom[3])
            reshaped_data[i][x][y][z][channel] = 1
    return reshaped_data

def get_pair(protein_path, ligand_path):
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

def get_index(filename):
    return filename.split('_')[0]

def get_protein_filenames(dir_path):
    return ls(dir_path, lambda x: x.endswith(PROTEIN_FILENAME_SUFFIX))

def get_protein_filepath(dir_path, index):
    protein_filename = index + PROTEIN_FILENAME_SUFFIX
    return os.path.abspath(os.path.join(dir_path, protein_filename))

def get_ligand_filepath(dir_path, index):
    ligand_filename = index + LIGAND_FILENAME_SUFFIX
    return os.path.abspath(os.path.join(dir_path, ligand_filename))

def shuffle_data(x_pro_list, x_lig_list, y_list):
    shuffled_index = np.arange(x_pro_list.shape[0])
    np.random.shuffle(shuffled_index)
    shuffled_x_pro_list = x_pro_list[shuffled_index]
    shuffled_x_lig_list = x_lig_list[shuffled_index]
    shuffled_y_list = y_list[shuffled_index]
    return shuffled_x_pro_list, shuffled_x_lig_list, shuffled_y_list

def generate_negative_pairings(x_pos_pro_list, x_pos_lig_list):
    x_neg_pro_list = []
    x_neg_lig_list = []
    y_neg = [0] * len(x_pos_pro_list)
    for i in range(len(x_pos_pro_list)):
        protein_filepath = x_pos_pro_list[i]
        neg_ligand_filepath = random_item(x_pos_lig_list, i)
        x_neg_pro_list.append(protein_filepath)
        x_neg_lig_list.append(neg_ligand_filepath)
    return x_neg_pro_list, x_neg_lig_list, y_neg

def random_item(lst, avoid_index):
    """Returns a random item from list that isn't in avoid_index position"""
    rand_index = avoid_index
    while rand_index == avoid_index:
        rand_index = random.randint(0, len(lst) - 1)
    return lst[rand_index]