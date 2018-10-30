from utils import plot_performance, get_example_shape, dump_pickle
from train_utils import generate_training_data_lists
from train_sequence import TrainSequence
from tensorflow import set_random_seed
from keras import optimizers, losses
from keras.utils import plot_model
from models import get_model
import numpy as np
import os

np.random.seed(0)
set_random_seed(0)

def main():
<<<<<<< HEAD
    epochs = 1
    batch_size = 32
    val_split = 0.33
    num_batches = int((3000 * (1 - val_split)) / batch_size)

=======
    epochs = 10
    batch_size = 128
<<<<<<< HEAD
    x_protein, x_ligand, y = get_training_data(size=128, save_training_data=True)
    x = {'protein_input': x_protein, 'ligand_input': x_ligand}
=======
    x_list, y_list = generate_training_data_lists()
    train_sequence = TrainSequence(x_list, y_list, batch_size)
>>>>>>> 92a621b711649341538d8725b251439ee4666714
>>>>>>> conv3d
    model_name = 'Dual-stream 3D Convolution Neural Network'
    model = get_model(model_name)(protein_data_shape=(None, None, None, 2), ligand_data_shape=(None, None, None, 2))
    plot_model(model, to_file='./{}.png'.format(model_name))
    print('Model Summary:')
    model.summary()
<<<<<<< HEAD

    loss = []
    acc = []
    for i in range(num_batches):
        start = i*batch_size
        end = start + batch_size
        x_protein, x_ligand, y = get_training_data(start=i*batch_size, end=end, save_training_data=False)
        x = {'protein_input': x_protein, 'ligand_input': x_ligand}
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        history = model.fit(x=x, y=y, epochs=epochs, verbose=1, batch_size=batch_size).history
        loss.extend(history['loss'])
        acc.extend(history['acc'])
=======
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit_generator(train_sequence, epochs=epochs, verbose=1).history
    dump_pickle('./history.pkl', history)
>>>>>>> conv3d
    model.save_weights('./{}_weights.h5'.format(model_name))
    history = {
        'loss': loss,
        'acc': acc
    }
    plot_performance(history, model_name, epochs, batch_size)

if __name__ == '__main__': main()