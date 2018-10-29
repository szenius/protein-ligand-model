import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv3D, Activation, MaxPooling3D, Dropout, Flatten, concatenate, GlobalMaxPooling3D
from keras.layers import LeakyReLU, LSTM, Embedding, Flatten

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

def get_model(model_name):
    if model_name == 'Dual-stream 3D Convolution Neural Network':
        return dual_stream_C3DNN
    else:
        return None

def dual_stream_C3DNN(protein_data_shape, ligand_data_shape, class_num=1):
    # Note: None in shape denotes variable size
    # inspired by https://arxiv.org/pdf/1801.10193.pdf, https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty374/4994792
    # hidden neurons 1024; 1024; 512
    def Conv3D_layer(filters, kernel_size):
        return Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid',
        data_format='channels_last', dilation_rate=(1, 1, 1), activation='relu', use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    def MaxPooling3D_Layer(pool_size):
        return MaxPooling3D(pool_size=pool_size, strides=None, padding='valid', data_format='channels_last')

    def FC_layer(units, activation='relu'):
        return Dense(units=units, activation=activation, use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    def protein_network(t):
        t = Conv3D_layer(filters=32, kernel_size=4)(t)
        t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        t = Conv3D_layer(filters=64, kernel_size=6)(t)
        t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        t = Conv3D_layer(filters=96, kernel_size=8)(t)
        t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        t = GlobalMaxPooling3D(data_format='channels_last')(t)
        return t

    def ligand_network(t):
        t = Conv3D_layer(filters=32, kernel_size=4)(t)
        t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        t = Conv3D_layer(filters=64, kernel_size=6)(t)
        t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        t = Conv3D_layer(filters=96, kernel_size=8)(t)
        t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        t = GlobalMaxPooling3D(data_format='channels_last')(t)
        return t

    protein_input = Input(shape=protein_data_shape, name='protein_input')
    ligand_input = Input(shape=ligand_data_shape, name='ligand_input')
    protein_stream = protein_network(protein_input)
    ligand_stream = ligand_network(ligand_input)

    t = concatenate([protein_stream, ligand_stream], axis=1)

    t = FC_layer(1024)(t)
    t = Dropout(0.5)(t)

    t = FC_layer(1024)(t)
    t = Dropout(0.5)(t)

    t = FC_layer(512)(t)
    t = Dropout(0.5)(t)

    t = FC_layer(class_num, activation='softmax')(t)

    return Model(inputs=[protein_input, ligand_input], outputs=t)