import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, Activation, MaxPooling3D, Dropout, Flatten, concatenate, GlobalMaxPooling3D

def get_model(model_name):
    if model_name == 'Dual-stream 3D Convolution Neural Network':
        return dual_stream_C3DNN
    elif model_name == 'test':
        return test
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

def test(protein_data_shape, ligand_data_shape, class_num=1):
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
        # t = Conv3D_layer(filters=32, kernel_size=4)(t)
        # t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        # t = Conv3D_layer(filters=64, kernel_size=6)(t)
        # t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        # t = Conv3D_layer(filters=96, kernel_size=8)(t)
        # t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        t = GlobalMaxPooling3D(data_format='channels_last')(t)
        return t

    def ligand_network(t):
        # t = Conv3D_layer(filters=32, kernel_size=4)(t)
        # t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        # t = Conv3D_layer(filters=64, kernel_size=6)(t)
        # t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        # t = Conv3D_layer(filters=96, kernel_size=8)(t)
        # t = MaxPooling3D_Layer(pool_size=(2,2,2))(t)
        t = GlobalMaxPooling3D(data_format='channels_last')(t)
        return t

    protein_input = Input(shape=protein_data_shape, name='protein_input')
    ligand_input = Input(shape=ligand_data_shape, name='ligand_input')
    protein_stream = protein_network(protein_input)
    ligand_stream = ligand_network(ligand_input)

    t = concatenate([protein_stream, ligand_stream], axis=1)

    t = FC_layer(1024)(t)
    # t = Dropout(0.5)(t)

    t = FC_layer(1024)(t)
    # t = Dropout(0.5)(t)

    t = FC_layer(512)(t)
    # t = Dropout(0.5)(t)

    t = FC_layer(class_num, activation='softmax')(t)

    return Model(inputs=[protein_input, ligand_input], outputs=t)