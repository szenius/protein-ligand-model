def data_reader(dir_path, data_size=3000):
    '''
    Given the directory to the training_data directory dir_path, read in all protein-ligand pairs and
    return them in the format 
    [
        [
            [protein_x_list, protein_y_list, protein_z_list, protein_type_list], 
            [ligand_x_list, ligand_y_list, ligand_z_list, ligand_type_list]
        ],
        ...
    ]
    '''
    if dir_path[-1] != '/':
        dir_path += '/'

    pairs = []
    for id in range(1, data_size + 1):
        id_string = pad_with_zeroes(id, 4)
        pfilename = dir_path + id_string + '_pro_cg.pdb'
        lfilename = dir_path + id_string + '_lig_cg.pdb'
        pair = [read_complex(pfilename), read_complex(lfilename)]
        pairs.append(pair)
    return pairs
        
def read_complex(filename):
    # Read atom data from file
    with open(filename, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    # Construct Complex from atom data
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
    return x_list, y_list, z_list, type_list

def pad_with_zeroes(num, target_size):
    result = str(num)
    while len(result) < target_size:
        result = '0' + result
    return result