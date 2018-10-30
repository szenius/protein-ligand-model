import sys
from keras.models import load_model
import numpy as np
from predict_util import *
from models import get_model

MLP = 'mlp'
LSTM = 'lstm'
CONV = 'conv'

TEST_DATA_PATH = os.path.abspath('./testing_data')
OUTPUT_FILENAME = "test_predictions.txt"

HEADER = '\t'.join(["pro_id", "lig1_id", "lig2_id", "lig3_id", "lig4_id", "lig5_id", "lig6_id", "lig7_id", "lig8_id", "lig9_id", "lig10_id", ])

mode = sys.argv[1] # 'mlp', 'lstm', 'conv'

# Load model
if mode == MLP:
    model = load_model('train_dist_mlp_10_32_337824.h5')
elif mode == LSTM:
    model = load_model('train_dist_lstm_10_32.h5')
elif mode == CONV:
    model = get_model('Dual-stream 3D Convolution Neural Network')
    model.load_weights('###.h5') # TODO:
else:
    print("Invalid mode {}".format(mode))
print("Loaded {} model.".format(mode))

# Read in all test data
if mode == CONV:
    protein_data, ligand_data, max_x, max_y, max_z = load_data_3d(TEST_DATA_PATH)
else:
    protein_data, ligand_data, max_length = load_data(TEST_DATA_PATH, -1)
print("Loaded {} sets from testing data files.".format(len(protein_data)))

# Generate predictions
predictions = []
output_data = np.zeros((len(protein_data), 11))
for i in range(len(protein_data)):
    prediction_row = []
    x_protein = protein_data[i]
    print("Generating predictions for protein {}".format(i))

    for j in range(len(ligand_data)):
        x_ligand = ligand_data[j]

        if mode == MLP:
            # Preprocessing
            not_required, x = generate_ij_distances(x_protein, x_ligand)
            for i in range(len(x), 337824):
                x.append(0)
            x = np.array([x])

            # Predict
            result = model.predict(x)
            prediction_row.append(result[0][0])

        elif mode == LSTM:
            not_required, x = generate_seq_distances(x_protein, x_ligand, 4615)
            # LSTM TODO:

        else:
            # Preprocessing
            x_protein = reshape_data(x_protein, max_x, max_y, max_z)
            x_ligand = reshape_data(x_ligand, max_x, max_y, max_z)

            # Predict
            x = {'protein_input': x_protein, 'ligand_input': x_ligand}
            result = model.predict(x)
            prediction_row.append(result[0][0])
    # Add top ten predictions to output_data
    prediction_row = list(reversed(sorted(prediction_row, key=float)))
    output_data[i] = prediction_row[:11]

# Save to file
np.savetxt(OUTPUT_FILENAME, output_data, delimiter="\t")

# Write header
with open(OUTPUT_FILENAME, 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write(HEADER.rstrip('\r\n') + '\n' + content)
