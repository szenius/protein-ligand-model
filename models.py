import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Conv1D, Activation
from keras.layers import MaxPool1D, MaxPool1D, Dropout, Flatten, LeakyReLU, concatenate, Reshape, GlobalMaxPooling1D

def cnn(input_shape, class_num=1):
    """
    Args:
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
        t = Conv1D(32, (3, 3), padding='valid')(im_input)
        t = Activation('relu')(t)
        t = MaxPool1D(pool_size=(2,2), padding='valid')(t)
        t = Dropout(0.5)(t)

    t = Flatten()(t)
    t = Dense(512)(t)
    t = Dense(class_num)(t)

    output = Activation('sigmoid')(t)

    model = Model(input=im_input, output=output)
    
    return model

def dual_stream_cnn(protein_data_shape=(None,4), ligand_data_shape=(None,4), class_num=1):
    # Note: None in shape denotes variable size.
    # inspired by https://arxiv.org/pdf/1801.10193.pdf, https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty374/4994792
    # hidden neurons 1024; 1024; 512
    
    def protein_network(t):
        t = Conv1D(filters=32, kernel_size=2, padding='valid')(t)
        t = Activation('relu')(t)
        t = MaxPool1D(pool_size=2, padding='valid')(t)

        # t = Conv1D(filters=64, kernel_size=2, padding='valid')(t)
        # t = Activation('relu')(t)
        # t = MaxPool1D(pool_size=2, padding='valid')(t)

        # t = Conv1D(filters=96, kernel_size=2, padding='valid')(t)
        # t = Activation('relu')(t)
        # t = MaxPool1D(pool_size=2, padding='valid')(t)
        return t

    def ligand_network(t):
        t = Conv1D(filters=32, kernel_size=2, padding='valid')(t)
        t = Activation('relu')(t)
        t = MaxPool1D(pool_size=2, padding='valid')(t)

        # t = Conv1D(filters=64, kernel_size=2, padding='valid')(t)
        # t = Activation('relu')(t)
        # t = MaxPool1D(pool_size=2, padding='valid')(t)
        
        # t = Conv1D(filters=96, kernel_size=2, padding='valid')(t)
        # t = Activation('relu')(t)
        # t = MaxPool1D(pool_size=2, padding='valid')(t)
        return t

    protein_input = Input(shape=protein_data_shape, name='protein_input')
    ligand_input = Input(shape=ligand_data_shape, name='ligand_input')
    protein_stream = protein_network(protein_input)
    ligand_stream = ligand_network(ligand_input)

    t = concatenate([protein_stream, ligand_stream], axis=-1)
    t = GlobalMaxPooling1D()(t)
    # t = Reshape(target_shape=(1024,1024))(t) # TODO: explore ROI pooling. 1024 is arbitrary
    # t = Flatten()(t) 

    t = Dense(1024)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(1024)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(512)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(class_num)(t)

    return Model(inputs=[protein_input, ligand_input], outputs=t)