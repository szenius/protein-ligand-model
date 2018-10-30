from utils import ls, load_pickle, dump_pickle, read_lines
import numpy as np
import random
import os
import sys

PROTEIN_FILENAME_SUFFIX = '_pro_cg.pdb'
LIGAND_FILENAME_SUFFIX = '_lig_cg.pdb' 

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
