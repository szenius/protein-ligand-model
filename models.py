import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv3D, Activation, MaxPooling3D, Dropout, Flatten, concatenate, GlobalMaxPooling3D
from keras.layers import LeakyReLU, LSTM, Embedding, Flatten

def mlp(input_dim, class_num=1):
    input = Input((input_dim,))

    for i in range(5):
        t = Dense(256)(input)
        t = LeakyReLU()(t)

    t = Dense(class_num, activation='sigmoid')(t)

    return Model(inputs=input, outputs=t)

def get_model(model_name):
    if model_name == 'Dual-stream 3D Convolution Neural Network':
        return dual_stream_C3DNN
    else:
        return None

def dual_stream_C3DNN(protein_data_shape, ligand_data_shape, class_num=1):
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
        t = Conv3D_layer(filters=16, kernel_size=4)(t)
        t = Conv3D_layer(filters=32, kernel_size=6)(t)
        t = GlobalMaxPooling3D(data_format='channels_last')(t)
        return t

    def ligand_network(t):
        t = Conv3D_layer(filters=8, kernel_size=2)(t)
        t = Conv3D_layer(filters=16, kernel_size=4)(t)
        t = GlobalMaxPooling3D(data_format='channels_last')(t)
        return t

    protein_input = Input(shape=protein_data_shape, name='protein_input')
    ligand_input = Input(shape=ligand_data_shape, name='ligand_input')
    protein_stream = protein_network(protein_input)
    ligand_stream = ligand_network(ligand_input)

    t = concatenate([protein_stream, ligand_stream], axis=1)

    t = FC_layer(64)(t)
    t = FC_layer(16)(t)
    t = FC_layer(class_num, activation='sigmoid')(t)

    return Model(inputs=[protein_input, ligand_input], outputs=t)