from utils import ls, load_pickle, dump_pickle, read_lines
from tqdm import tqdm
import numpy as np
import random
import os
import gc

PROTEIN_FILENAME_SUFFIX = '_pro_cg.pdb'
LIGAND_FILENAME_SUFFIX = '_lig_cg.pdb' 

def generate_ij_distances(protein, ligand):
    empty_row = [0,0,0,0,0]
    distances_mlp = []
    for i in range(len(protein)):
        for j in range(len(ligand)):
            row = atom_vector(protein[i], ligand[j])
            distances_mlp.extend(row)
    return distances_mlp

def euclidean_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

def type_index(type1, type2):
    min_type = min(type1, type2)
    max_type = max(type1, type2)
    if min_type == max_type:
        index = min_type
    else:
        index = min_type + max_type + 2
    return int(index) - 1

def atom_vector(atom1, atom2=[0,0,0,0]):
    result = [0,0,0,0,0]
    ed = euclidean_distance(atom1[:-1], atom2[:-1])
    result[type_index(atom1[-1], atom2[-1])] = ed
    return result

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

def load_batch(batch_x, max_dims=10000):
    batch_size = len(batch_x)
    protein_data = []
    ligand_data = []
    
    for protein_path, ligand_path in batch_x:
        protein, ligand, max_x, max_y, max_z = get_pair(protein_path, ligand_path, max_dims)
        protein_data.append(protein)
        ligand_data.append(ligand)

    # Reshape data
    flattened = []
    for i in range(len(protein_data)):
        ij_distance_flattened = generate_ij_distances(protein_data[i], ligand_data[i])
        flattened.append(ij_distance_flattened)
    for i in range(len(flattened)):
        for j in range(len(flattened[i]), max_dims):
            flattened[i].append(0)
        flattened[i] = flattened[i][:max_dims]
    
    return np.array(flattened)

def downsample_complex(complex, max_x, max_y, max_z, max_dims):
    for atom in complex:
        atom[0] = atom[0] / max_x * max_dims[0] if max_x != 0 else 0
        atom[1] = atom[1] / max_y * max_dims[1] if max_y != 0 else 0
        atom[2] = atom[2] / max_z * max_dims[2] if max_z != 0 else 0
    return complex

def get_pair(protein_path, ligand_path, max_dims):
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