import os
import numpy as np
from models import get_model, mlp
from utils import ls, load_pickle, dump_pickle, read_lines

###### CHANGE THESE BEFORE RUNNING ######
TESTING_DATA_PATH = './testing_data' 
WEIGHTS_FILENAME = 'Dual-stream 3D Convolution Neural Network_weights.h5'

PROTEIN_FILENAME_SUFFIX = '_pro_cg.pdb'
LIGAND_FILENAME_SUFFIX = '_lig_cg.pdb'

def get_index(filename):
    return filename.split('_')[0]

def get_filenames(dir_path, suffix):
    return ls(dir_path, lambda x: x.endswith(suffix))

def generate_testing_data_lists(dir_path = os.path.abspath(TESTING_DATA_PATH)):
    x_pro_list = get_filenames(dir_path, PROTEIN_FILENAME_SUFFIX)
    x_lig_list = get_filenames(dir_path, LIGAND_FILENAME_SUFFIX)
        
    return x_pro_list, x_lig_list

def load_conv():
    model = get_model('Dual-stream 3D Convolution Neural Network')(protein_data_shape=(None, None, None, 2), ligand_data_shape=(None, None, None, 2))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])
    model.load_weights(WEIGHTS_FILENAME) # TODO:
    return model

def load_batch(batch_x, max_dims=(10, 10, 10)):
    batch_size = len(batch_x)
    protein_data = []
    ligand_data = []
    global_max_x = 0
    global_max_y = 0
    global_max_z = 0
    
    for protein_path, ligand_path in batch_x:
        protein, ligand, max_x, max_y, max_z = get_pair(protein_path, ligand_path, downsample=True, max_dims=max_dims)
        protein_data.append(protein)
        ligand_data.append(ligand)

        # Update global maxes
        global_max_x = max(max_x, global_max_x)
        global_max_y = max(max_y, global_max_y)
        global_max_z = max(max_z, global_max_z)
    
    if max_dims is None:
        target_shape = (batch_size, int(global_max_x), int(global_max_y), int(global_max_z), 2)
    else:
        target_shape = (batch_size, max_dims[0], max_dims[1], max_dims[2], 2)
    return format_data(protein_data, target_shape), format_data(ligand_data, target_shape) 

def get_pair(protein_path, ligand_path, downsample=False, types_1based=False, max_dims=(100,100,100)):
    protein, pro_max_x, pro_max_y, pro_max_z = read_complex(protein_path, types_1based)
    ligand,  lig_max_x, lig_max_y, lig_max_z = read_complex(ligand_path, types_1based)

    # Resolve coordinate maxes for this pair
    max_x = max(pro_max_x, lig_max_x)
    max_y = max(pro_max_y, lig_max_y)
    max_z = max(pro_max_z, lig_max_z)

    # Downsample
    if downsample is True:
        protein = downsample_complex(protein, max_x, max_y, max_z, max_dims)
        ligand = downsample_complex(ligand, max_x, max_y, max_z, max_dims)

    return protein, ligand, max_x, max_y, max_z

def read_complex(file_path, types_1based):
    # Read atom data from file
    content = [x.strip() for x in read_lines(file_path)]

    # Construct complex from atom data
    complex_seq = []
    for line in content:
        splitted_line = line.strip().split('\t')
        x = float(splitted_line[0])
        y = float(splitted_line[1])
        z = float(splitted_line[2])
        atom_type = splitted_line[3].strip()
        hydrophobicity = 2 if atom_type == 'h' else 1 # 2 for hydrophobic, 1 for polar
        if types_1based is False: hydrophobicity -= 1
        complex_seq.append([x, y, z, hydrophobicity])

    np_complex_seq = np.asarray(complex_seq)
    xyzh_mins = np.amin(np_complex_seq, axis=0)                 # get all mins
    xyzh_mins[-1] = 0                                           # force min of hydrophobicity to be 0 (just in case)
    np_complex_seq -= xyzh_mins                                 # translate values such that min is now zero
    max_x, max_y, max_z, _ = np.amax(np_complex_seq, axis=0)    # the new maxes for x, y, z
    complex_seq = np_complex_seq.tolist()                       # convert back to python list
    return complex_seq, max_x, max_y, max_z

def format_data(data, target_shape):
    reshaped_data = np.zeros(target_shape)
    for i in range(len(data)):
        complex_seq = data[i]
        for atom in complex_seq:
            x = int(atom[0]) - 1
            y = int(atom[1]) - 1
            z = int(atom[2]) - 1
            channel = int(atom[3])
            reshaped_data[i][x][y][z][channel] = 1
    return reshaped_data

def downsample_complex(complex, max_x, max_y, max_z, max_dims):
    for atom in complex:
        atom[0] = atom[0] / max_x * max_dims[0] if max_x != 0 else 0
        atom[1] = atom[1] / max_y * max_dims[1] if max_y != 0 else 0
        atom[2] = atom[2] / max_z * max_dims[2] if max_z != 0 else 0
    return complex