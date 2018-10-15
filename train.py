from training_data import get_training_data
from tensorflow import set_random_seed
from keras import optimizers, losses
from models import dual_stream_cnn
import numpy as np
import os

np.random.seed(0)
set_random_seed(0)

def main():
    x_protein, x_ligand, y = get_training_data()
    print(len(x_protein), len(x_protein[0]), len(x_protein[0][0]))
    model = dual_stream_cnn()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit(x=[x_protein, x_ligand], y=y, epochs=1, verbose=1)
    print(history.history['loss'], history.history['acc'])

if __name__ == '__main__': main()