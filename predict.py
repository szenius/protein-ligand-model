import sys
from predict_util import *
import os
import numpy as np

MLP = 'mlp'
CONV = 'conv'
MLP_MAX_LENGTH = 10000

OUTPUT_FILENAME = "test_predictions.txt"
HEADER = '\t'.join(["pro_id", "lig1_id", "lig2_id", "lig3_id", "lig4_id", "lig5_id", "lig6_id", "lig7_id", "lig8_id", "lig9_id", "lig10_id", ])

mode = sys.argv[1] # 'mlp', 'conv'

model = load_mlp() if mode == MLP else load_conv()
x_pro_list, x_lig_list = generate_testing_data_lists()

# Generate predictions
predictions = []
output_data = np.zeros((len(x_pro_list), 11))
for i in range(len(x_pro_list)):
    pro_filename = x_pro_list[i]
    prediction_row = []
    pro_filename_full = os.path.abspath(os.path.join(TESTING_DATA_PATH, pro_filename))
    print("Generating predictions for {}".format(pro_filename))

    for lig_filename in x_lig_list:
        lig_filename_full = os.path.abspath(os.path.join(TESTING_DATA_PATH, lig_filename))
        x_list = np.concatenate(([[pro_filename_full]], [[lig_filename_full]]), axis=1)

        # Preprocessing
        if mode == MLP:
            x = load_batch_dist(x_list)
        else:
            x_protein, x_ligand = load_batch(x_list)
            x = {'protein_input': x_protein, 'ligand_input': x_ligand}

        # Predict
        result = model.predict(x)
        prediction_row.append(result[0][0])

    # Add top ten predictions to output_data
    prediction_row = np.array(prediction_row)
    np.argsort(prediction_row)
    np.flipud(prediction_row)
    output_data[i] = prediction_row[:11]

# Save to file
np.savetxt(OUTPUT_FILENAME, output_data, delimiter="\t")

# Write header
with open(OUTPUT_FILENAME, 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(HEADER.rstrip('\r\n') + '\n' + content)
