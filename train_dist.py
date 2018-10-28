from training_data_dist import get_training_data
from tensorflow import set_random_seed
from keras import optimizers, losses
from models import mlp, lstm
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

np.random.seed(0)
set_random_seed(0)

mode = sys.argv[1] # 'lstm' or 'mlp'
data = sys.argv[2] if len(sys.argv) > 2 else 'seq1d' # 'ij' or 'seq1d' or 'seq2d'

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
    x_seq_dist_2d, x_seq_dist_1d, x_ij_dist, y = get_training_data()

    # Get model
    if mode == 'mlp':
        if data == 'ij':
            x = x_ij_dist
            model = mlp(x.shape[1])
        else: 
            x = x_seq_dist_1d
            model = mlp(x.shape[1])
    elif mode == 'lstm':
        if data == 'ij':
            x = x_ij_dist
            model = lstm(x.shape[1])
        else:
            x = x_seq_dist_2d
            model = lstm(x.shape[1])
    else:
        print("Invalid mode. Please use 'mlp' or 'lstm'.")
        sys.exit()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    # Fit model
    batch_size = 32
    epochs = 10
    history = model.fit(x=x, y=y, epochs=epochs, verbose=1, batch_size=batch_size)
    loss = history.history['loss']
    acc = history.history['acc']

    filename_prefix = '_'.join(['train', 'dist', mode, str(epochs), str(batch_size)])

    # Plot loss vs accuracy
    plot([loss, acc], ['loss', 'acc'], ['b', 'r'], 'epoch', '', mode.upper()\
    + " Training", filename_prefix + '.png')

    # Save model
    model.save(filename_prefix + '.h5')

if __name__ == '__main__': main()