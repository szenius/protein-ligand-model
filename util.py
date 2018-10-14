import numpy as np
import csv
import os

def format_abspath(file_path):
    return os.path.join(os.path.abspath(file_path), '') # automatically adds trailing / for directory

def write_csv(rows, output_path, header=None, delimiter=','):
  """Writes out rows to csv file given output path"""
  with open(output_path, 'w') as csvfile:
    out_writer = csv.writer(csvfile, delimiter=delimiter)
    if header:
      out_writer.writerow(header)
    for row in rows:
      out_writer.writerow(row)

def read_lines(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def format_training_data(pairs, rows=1000):
    '''
    Key Arguments:
        pairs -- protein-ligand pairs in the format
            [
                [
                    [protein_x_list, protein_y_list, protein_z_list, protein_type_list], 
                    [ligand_x_list, ligand_y_list, ligand_z_list, ligand_type_list]
                ],
                ...
            ]
        rows -- use this to fix the first dimension for the return array
    
    Returns:
        array of shape: num_pairs x 2 x rows x 4
            axis1: dim = number of protein-ligand pairs, each is a protein-ligand pair
            axis2: dim = 2, each is either protein or ligand for the pair
            axis3: dim = number of atoms, each is an atom for the protein/ligand
            axis4: dim = 4, each represent x, y, z, type for the atom of the protein/ligand
    '''
    result = []
    for pair in pairs:
        protein = reshape_2darray(pair[0], rows)
        ligand = reshape_2darray(pair[1], rows)
        result.append([protein, ligand])
    return result

def reshape_2darray(data, target_rows):
    # TODO: right now we pad with zeroes or cut off the extra rows.
    # TODO: we should consider ROI pooling
    if len(data) > target_rows:
        return data[0:target_rows]
    while len(data) < target_rows:
        data.append([0 for x in range(len(data[0]))])
    return data

def load_training_data(dir_path, data_size=3000):
    '''
    TODO: generate negative training examples
    
    Key Arguments:
        dir_path -- directory to the training_data directory dir_path. This is a path String.
        
    Returns: 
        pairs -- Protein-Ligand pairs in the following format
            [
                [
                    [protein_x_list, protein_y_list, protein_z_list, protein_type_list], 
                    [ligand_x_list, ligand_y_list, ligand_z_list, ligand_type_list]
                ],
                ...
            ]
        labels -- 0 or 1 for not a pair, or pair
    '''

    dir_path = format_abspath(dir_path)

    pairs = []
    labels = []
    for id in range(1, data_size + 1):
        id_string = pad_left(str(id), '0', 4)
        pfilename = dir_path + id_string + '_pro_cg.pdb'
        lfilename = dir_path + id_string + '_lig_cg.pdb'
        pairs.append([read_complex(pfilename), read_complex(lfilename)])
        labels.append(1)
    return pairs, labels
        
def read_complex(file_path):
    # Read atom data from file
    content = [x.strip() for x in read_lines(file_path)]
    
    # Construct molecule from atom data
    x_list = []
    y_list = []
    z_list = []
    type_list = []

    for line in content:
        x_list.append(float(line[30:38].strip()))
        y_list.append(float(line[38:46].strip()))
        z_list.append(float(line[46:54].strip()))

        atom_type = line[76:78].strip()
        hydrophobicity = 1 if atom_type == 'C' else 0 # 1 for hydrophobic, 0 for polar
        type_list.append(hydrophobicity)

    return [x_list, y_list, z_list, type_list]

def pad_left(string, pad_char, target_width):
    return pad_char * (target_width - len(string)) + string