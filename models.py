import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten, LeakyReLU

def cnn(input_shape, class_num=1):
    """
    Keyword Arguments:
        input_shape {tuple} -- shape of input data. Should be max_num_atoms x 4 x 2.
            4: x, y, z, type
            2: protein, ligand.
            Pad empty rows with zeroes.
        class_num {int} -- number of classes. Should be 1 for binary classification.
    
    Returns:
        model -- keras.models.Model() object
    """

    im_input = Input(shape=input_shape)

    for i in range(3):
        t = Conv2D(32, (3, 3), padding='valid')(im_input)
        t = Activation('relu')(t)
        t = MaxPool2D(pool_size=(2,2), padding='valid')(t)
        t = Dropout(0.5)(t)

    t = Flatten()(t)
    t = Dense(512)(t)
    t = Dense(class_num)(t)

    output = Activation('sigmoid')(t)

    model = Model(input=im_input, output=output)
    
    return model
