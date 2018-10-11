import numpy as np
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
        4d array of size num_pairs x 2 x rows x 4
            axis1: number of protein-ligand pairs
            axis2: protein, ligand
            axis3: rows
            axis4: columns represent x, y, z, type for each atom
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
    if dir_path[-1] != '/':
        dir_path += '/'

    pairs = []
    labels = []
    for id in range(1, data_size + 1):
        id_string = pad_with_zeroes(id, 4)
        pfilename = dir_path + id_string + '_pro_cg.pdb'
        lfilename = dir_path + id_string + '_lig_cg.pdb'
        pairs.append([read_complex(pfilename), read_complex(lfilename)])
        labels.append(1)
    return pairs, labels
        
def read_complex(filename):
    # Read atom data from file
    with open(filename, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    # Construct Molecule from atom data
    x_list = []
    y_list = []
    z_list = []
    type_list = []

    for line in content:
        x_list.append(float(line[30:38].strip()))
        y_list.append(float(line[38:46].strip()))
        z_list.append(float(line[46:54].strip()))

        type = line[76:78].strip()
        if type == 'C':
            type_list.append(0) # hydrophobic
        else:
            type_list.append(1) # polar
    return [x_list, y_list, z_list, type_list]

def pad_with_zeroes(num, target_size):
    result = str(num)
    while len(result) < target_size:
        result = '0' + result
    return result