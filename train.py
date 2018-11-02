from utils import plot_performance, get_example_shape, dump_pickle
from train_utils import generate_training_data_lists
from train_sequence import TrainSequence
from tensorflow import set_random_seed
from keras import optimizers, losses
from keras.utils import plot_model
from models import get_model
import numpy as np
import os
from keras import backend as K

np.random.seed(0)
set_random_seed(0)

def main():
    epochs = 20
    batch_size = 128
    x_list, y_list = generate_training_data_lists()
    steps = len(x_list) / batch_size
    train_sequence = TrainSequence(x_list, y_list, batch_size)
    model_name = 'Dual-stream 3D Convolution Neural Network'
    model = get_model(model_name)(protein_data_shape=(None, None, None, 2), ligand_data_shape=(None, None, None, 2))

    plot_model(model, to_file='./{}.png'.format(model_name))
    print('Model Summary:')
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])
    history = model.fit_generator(train_sequence, epochs=epochs, steps_per_epoch=steps, verbose=1).history
    dump_pickle('./history.pkl', history)
    model.save_weights('./{}_weights.h5'.format(model_name))
    plot_performance(history, model_name, epochs, batch_size)

if __name__ == '__main__': main()