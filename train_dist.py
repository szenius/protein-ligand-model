from training_data_dist import get_training_data
from tensorflow import set_random_seed
from keras import optimizers, losses
from models import mlp, lstm
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(0)
set_random_seed(0)

# CHANGE THIS VARIABLE TO SWITCH BETWEEN MLP AND LSTM
mode = 'mlp' # 'mlp' or 'lstm'

def plot(data, labels, colours, xlabel, ylabel, title, filename):
    plt.figure()
    for i in range(len(data)):
        plt.plot(data[i], label=labels[i], c=colours[i])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def main():
    x_protein, x_ligand, x_distance, y = get_training_data()

    # Get model
    if mode == 'mlp':
        model = mlp(x_distance.shape[1])
    else:
        model = lstm(x_distance.shape[1])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    # Fit model
    epochs = 10
    batch_size = 32
    history = model.fit(x=x_distance, y=y, epochs=epochs, verbose=1, batch_size=batch_size)
    plot([history.history['loss'], history.history['acc']], ['loss', 'acc'], ['b', 'r'],\
        'epoch', '', mode.upper() + " Training", '_'.join(['train', 'dist', mode, epochs, batch_size, '.png']))

if __name__ == '__main__': main()