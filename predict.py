import sys
from keras.models import load_model
import numpy as np
from predict_util import *

MLP = 'mlp'
LSTM = 'lstm'
CONV = 'conv'

TEST_DATA_PATH = os.path.abspath('./testing_data')

mode = sys.argv[1] # 'mlp', 'lstm', 'conv'

# Load model
if mode == MLP:
    model = load_model('train_dist_mlp_10_32_337824.h5')
elif mode == LSTM:
    model = load_model('train_dist_lstm_10_32.h5')
else:
    # CONV TODO:
    sys.exit()
print("Loaded {} model.".format(mode))

# Read in all test data
protein_data, ligand_data, max_length = load_data(TEST_DATA_PATH, -1)
print("Loaded {} sets from testing data files.".format(len(protein_data)))

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
            #CONV TODO:
            sys.exit()
    # Add top ten predictions to output_data
    prediction_row = list(reversed(sorted(prediction_row, key=float)))
    output_data[i] = prediction_row[:11]

# Save to file
np.savetxt("foo.csv", output_data, delimiter=",")