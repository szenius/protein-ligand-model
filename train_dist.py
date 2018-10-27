from training_data_dist import get_training_data
from tensorflow import set_random_seed
from keras import optimizers, losses
from models import mlp, lstm, single_stream_cnn
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(0)
set_random_seed(0)

# CHANGE THIS VARIABLE TO SWITCH BETWEEN MLP, LSTM, CONV2D
mode = 'conv2d' # 'mlp' or 'lstm' or 'conv2d'

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
    x_seq_dist, x_ij_dist, y = get_training_data()

    # Get model
    if mode == 'mlp':
        model = mlp(x_distance.shape[1])
    elif mode == 'lstm':
        model = lstm(x_distance.shape[1])
    else:            
        model = single_stream_cnn()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    # Fit model
    if mode == 'conv2d':
        epochs = 1
        batch_size = 1
        loss = []
        acc = []
        for i in range(len(x_ij_dist)):
            history = model.fit(x=np.array([x_ij_dist[i]]), y=y[i], epochs=epochs, verbose=1, batch_size=batch_size)
            loss.append(history.history['loss'])
            acc.append(history.history['acc'])
    else:
        epochs = 10
        batch_size = 32
        history = model.fit(x=x_seq_dist, y=y, epochs=epochs, verbose=1, batch_size=batch_size)
        loss = history.history['loss']
        acc = history.history['acc']

    # Plot loss vs accuracy
    plot([loss, acc], ['loss', 'acc'], ['b', 'r'], 'epoch', '', mode.upper()\
    + " Training", '_'.join(['train', 'dist', mode, epochs, batch_size, '.png']))

if __name__ == '__main__': main()