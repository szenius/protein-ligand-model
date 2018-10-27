from training_data import get_training_data
from tensorflow import set_random_seed
from keras import optimizers, losses
from models import dual_stream_cnn
import numpy as np
import os

np.random.seed(0)
set_random_seed(0)

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
    x_protein, x_ligand, y = get_training_data()
    epochs = 10
    batch_size = 1
    mode = 'conv3d'
    model = dual_stream_cnn()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit(x={'protein_input': x_protein, 'ligand_input': x_ligand}, y=y, epochs=epochs, verbose=1, batch_size=batch_size)
        # Plot loss vs accuracy
    plot([history.history['loss'], history.history['acc']], ['loss', 'acc'], ['b', 'r'], 'epoch', '', mode.upper()\
    + " Training", '_'.join(['train', 'dist', mode, epochs, batch_size, '.png']))


if __name__ == '__main__': main()