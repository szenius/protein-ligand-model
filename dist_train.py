from tensorflow import set_random_seed
from keras import optimizers, losses
from models import mlp
import numpy as np
from keras.utils import plot_model
from utils import plot_performance, dump_pickle
from dist_train_utils import generate_training_data_lists
from dist_train_sequence import TrainSequence

np.random.seed(0)
set_random_seed(0)

mode = 'mlp'
data = 'ij'
batch_size = 16
epochs = 10
model_name = 'Baseline 5x20 MLP'

def main():
    # Prepare data
    x_list, y_list = generate_training_data_lists()
    steps = len(x_list) / batch_size
    train_sequence = TrainSequence(x_list, y_list, batch_size)

    # Prepare model
    model = mlp(10000)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

    # Plot model    
    plot_model(model, to_file='./{}.png'.format(model_name))

    # Fit model
    history = model.fit_generator(train_sequence, epochs=epochs, steps_per_epoch=steps, verbose=1).history

    # Plot loss vs accuracy
    plot_performance(history, model_name, epochs, batch_size)

    # Write loss and acc to file
    dump_pickle('./history.pkl', history)

    # Save model
    model.save_weights('./{}_weights.h5'.format(model_name))

if __name__ == '__main__': main()