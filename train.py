from utils import plot_performance, get_example_shape
from training_data import get_training_data
from tensorflow import set_random_seed
from keras import optimizers, losses
from keras.utils import plot_model
from models import get_model
import numpy as np
import os

np.random.seed(0)
set_random_seed(0)

def main():
    epochs = 10
    batch_size = 32
    x_protein, x_ligand, y = get_training_data(size=128, save_training_data=True)
    x = {'protein_input': x_protein, 'ligand_input': x_ligand}
    model_name = 'Dual-stream 3D Convolution Neural Network'
    model = get_model(model_name)(protein_data_shape=(None, None, None, 2), ligand_data_shape=(None, None, None, 2))
    plot_model(model, to_file='./{}.png'.format(model_name))
    print('Model Summary:')
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit(x=x, y=y, epochs=epochs, verbose=1, batch_size=batch_size).history
    model.save_weights('./{}_weights.h5'.format(model_name))
    plot_performance(history, model_name, epochs, batch_size)

if __name__ == '__main__': main()