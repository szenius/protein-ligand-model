def read_pdb(filename):
	
	with open(filename, 'r') as file:
		strline_L = file.readlines()
		# print(strline_L)

	X_list = list()
	Y_list = list()
	Z_list = list()
	atomtype_list = list()
	for strline in strline_L:
		# removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
		stripped_line = strline.strip()
		# print(stripped_line)

		splitted_line = stripped_line.split('\t')
		
		X_list.append(float(splitted_line[0]))
		Y_list.append(float(splitted_line[1]))
		Z_list.append(float(splitted_line[2]))
		atomtype_list.append(str(splitted_line[3]))

	return X_list, Y_list, Z_list, atomtype_list


X_list, Y_list, Z_list, atomtype_list=read_pdb("testing_data/0001_lig_cg.pdb")
print(X_list)
print(Y_list)
print(Z_list)
print(atomtype_list)
