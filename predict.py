import sys
from predict_utils import *
import os
import numpy as np

OUTPUT_FILENAME = "test_predictions.txt"
HEADER = '\t'.join(["pro_id", "lig1_id", "lig2_id", "lig3_id", "lig4_id", "lig5_id", "lig6_id", "lig7_id", "lig8_id", "lig9_id", "lig10_id", ])

# Load model and data
model = load_conv()
x_pro_list, x_lig_list = generate_testing_data_lists()

# Generate predictions
predictions = []
output_data = np.zeros((len(x_pro_list), 10), dtype=int)
for i in range(len(x_pro_list)):
    pro_filename = x_pro_list[i]
    prediction_row = []
    pro_filename_full = os.path.abspath(os.path.join(TESTING_DATA_PATH, pro_filename))
    print("Generating predictions for {}".format(pro_filename))

    for lig_filename in x_lig_list:
        lig_filename_full = os.path.abspath(os.path.join(TESTING_DATA_PATH, lig_filename))
        x_list = np.concatenate(([[pro_filename_full]], [[lig_filename_full]]), axis=1)

        # Preprocessing
        x_protein, x_ligand = load_batch(x_list)
        x = {'protein_input': x_protein, 'ligand_input': x_ligand}

        # Predict
        result = model.predict(x)
        prediction_row.append(result[0][0])

    # Add top ten predictions to output_data
    prediction_row = np.array(prediction_row)
    prediction_row = np.argsort(prediction_row)
    prediction_row = np.flipud(prediction_row)
    output_data[i] = prediction_row[:10]
    print(output_data[i])

# Save to file
np.savetxt(OUTPUT_FILENAME, output_data.astype(int), delimiter="\t", fmt='%i')

# Write header
with open(OUTPUT_FILENAME, 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(HEADER.rstrip('\r\n') + '\n' + content)
