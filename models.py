import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, Activation, MaxPool1D, Dropout, concatenate, GlobalMaxPooling1D
from keras.layers import Conv2D, MaxPool2D, GlobalMaxPooling2D, LeakyReLU, LSTM, Embedding, Flatten

def lstm(length, class_num=1):
    '''
    Adapted from a random LSTM model for binary classification I found
    Source: https://gist.github.com/urigoren/b7cd138903fe86ec027e715d493451b4
    TODO: find better architecture???
    '''
    model = Sequential()
    model.add(LSTM(units=256, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=256, activation='tanh', recurrent_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))
    return model

def mlp(num_input, class_num=1):
    '''
    TODO: better architecture
    '''
    input = Input((num_input,))

    for i in range(5):
        t = Dense(20)(input)
        t = LeakyReLU()(t)

    t = Dense(class_num, activation='softmax')(t)

    return Model(inputs=input, outputs=t)

def dual_stream_cnn(protein_data_shape=(None,4), ligand_data_shape=(None,4), class_num=1):
    # Note: None in shape denotes variable size
    # inspired by https://arxiv.org/pdf/1801.10193.pdf, https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty374/4994792
    # hidden neurons 1024; 1024; 512
    
    def protein_network(t):
        t = Conv1D(filters=32, kernel_size=4, padding='valid')(t)
        t = Activation('relu')(t)
        t = MaxPool1D(pool_size=2, padding='valid')(t)

        t = Conv1D(filters=64, kernel_size=6, padding='valid')(t)
        t = Activation('relu')(t)
        t = MaxPool1D(pool_size=2, padding='valid')(t)

        t = Conv1D(filters=96, kernel_size=8, padding='valid')(t)
        t = Activation('relu')(t)
        t = MaxPool1D(pool_size=2, padding='valid')(t)
        return t

    def ligand_network(t):
        t = Conv1D(filters=32, kernel_size=4, padding='valid')(t)
        t = Activation('relu')(t)
        t = MaxPool1D(pool_size=2, padding='valid')(t)

        t = Conv1D(filters=64, kernel_size=6, padding='valid')(t)
        t = Activation('relu')(t)        
        t = MaxPool1D(pool_size=2, padding='valid')(t)

        t = Conv1D(filters=96, kernel_size=8, padding='valid')(t)
        t = Activation('relu')(t)
        t = MaxPool1D(pool_size=2, padding='valid')(t)
        return t

    protein_input = Input(shape=protein_data_shape, name='protein_input')
    ligand_input = Input(shape=ligand_data_shape, name='ligand_input')
    protein_stream = protein_network(protein_input)
    ligand_stream = ligand_network(ligand_input)

    t = concatenate([protein_stream, ligand_stream], axis=1)
    t = GlobalMaxPooling1D()(t) # TODO: explore ROI pooling

    t = Dense(1024)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(1024)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(512)(t)
    t = Activation('relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(class_num, activation='softmax')(t)

    return Model(inputs=[protein_input, ligand_input], outputs=t)