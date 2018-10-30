from utils import ls, load_pickle, dump_pickle, read_lines
from tqdm import tqdm
import numpy as np
import random
import os
import gc

PROTEIN_FILENAME_SUFFIX = '_pro_cg.pdb'
LIGAND_FILENAME_SUFFIX = '_lig_cg.pdb' 

def get_training_data(
    training_data_dir_path = os.path.abspath('./training_data'),
    training_data_pkl_path_format = './training_data_{}.pkl',
    start=-1, end=-1, 
    size = 0, reprocess=False, save_training_data=True):
    '''
    Args:
        training_data_dir_path (str):  the training_data directory.
        training_data_pkl_path_format (str): string format for the serialized training data file path.
        size (int): size of training examples in training_data_dir_path to generate data from. 0 indicates all
        reprocess (boolean):  whether or not tp ignore serialized training data and reprocessed raw training data.

    Returns: 
        training data (list): [protein data, ligand data, labels]
    '''
    training_data_pkl_path = training_data_pkl_path_format.format(size if size > 0 else 'all')
    # If reprocessing training data or pkl does not exist
    if reprocess or not os.path.exists(training_data_pkl_path):
        training_data = generate_training_data(training_data_dir_path, size=size, start=start, end=end)
        if save_training_data:
            dump_pickle(training_data_pkl_path, training_data) # save data to disk
        return training_data
    # Else load from previously prepared training data
    else:
        return load_pickle(training_data_pkl_path)

def generate_training_data(training_data_dir_path, size, start, end):
    '''
    Args:
        training_data_dir_path (str): the training_data directory.
        size (int): size of training examples in training_data_dir_path to generate data from. 0 indicates all

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

    ############################## Function body #############################
    protein_data, ligand_data, max_x, max_y, max_z = load_data(training_data_dir_path, size, start, end)

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
    max_x = int(max_x)
    max_y = int(max_y)
    max_z = int(max_z)
    print("Reshaping data into shape ({},{},{},{})".format(max_x, max_y, max_z, 2))
    x_protein = reshape_data(x_protein, max_x, max_y, max_z, desc='Reshaping proteins')
    x_ligand = reshape_data(x_ligand, max_x, max_y, max_z, desc='Reshaping ligands')

    return np.array(x_protein), np.array(x_ligand), np.array(y)

def load_data(dir_path, size, start, end):
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
    # if size > 0:
    #     protein_filenames_list = protein_filenames_list[0:size]

    if start != -1 and end != -1:
        protein_filenames_list = protein_filenames_list[start:end]

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
