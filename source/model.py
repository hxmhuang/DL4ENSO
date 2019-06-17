# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import image
##from tensorflow.keras.utils.data_utils import get_file
##from tensorflow.keras.applications.imagenet_utils import preprocess_input
##from tensorflow.keras.layers.wrappers import TimeDistributed
from tensorflow.keras.initializers import glorot_uniform



def conv2d_lstm(input_shape= (10,60,60,2), output_shape = 3,filters_list=[16,8], return_seq=False):
    '''
    Input: A tensor, shape:(batch_size, time_length, width, height, channels)
    Output: A tensor,shape:(batch_size, 1) or (batch_size, time_length)
    input_shape : (time_length, width, height, channels)
    output_shape: a real number or a sequence for each input
    filters_list: number of filters for each conv2d_lstm layer
    return_seq: The return_sequences for lst
    '''
    
    X_input = Input(input_shape)
    X = ConvLSTM2D(filters=32, kernel_size=(3, 3),padding='same',return_sequences=True)(X_input)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
        
    for i in range(len(filters_list)):
        #gpu_name = '/gpu:'+ str(i%2)
        gpu_name = '/gpu:1'
        with tf.device(gpu_name):
            X = ConvLSTM2D(filters=filters_list[i], kernel_size=(3, 3),padding='same', return_sequences=True)(X)
            X = Activation('relu')(X)
            X = BatchNormalization()(X)
    
    if return_seq:
       # X = TimeDistributed(Conv2D(filters=1, kernel_size=(3, 3),activation='relu', data_format = 'channels_last'))(X)
        Output = ConvLSTM2D(filters=output_shape, kernel_size=(1, 1),padding='same',return_sequences=True, name="extra_output")(X)
        X = TimeDistributed(Flatten())(Output)
        X = TimeDistributed(Dense(1), name='main_output')(X)
    else:
        Output = Conv3D(filters=2, kernel_size=(3, 3, 3),activation='relu', data_format = 'channels_last', name="extra_output")(X)
        X = Flatten()(Output)
        X = Dense(1, name='main_output')(X)

    model = Model(inputs=X_input, outputs=[X, Output])
    return model

def lstm_1d(input_shape = (10,8540), output_shape = 1, units_list = [2000,1000,500],return_seq = False):
    '''
    Input: A tensor, shape:(batch_size, time_length, number_features)
    Output: A tensor,shape:(batch_size, 1) or (batch_size, time_length)
    input_shape : (time_length, number_features)
    output_shape: a real number or a sequence for each input
    units_list: number of units for each lstm layer
    return_seq: The return_sequences for lst
    '''
    
    X_input = Input(input_shape)
    X = LSTM(4000, return_sequences=True)(X_input)
    for units in units_list:
        X = LSTM(units, return_sequences=True)(X)
    
    if(return_seq):
        X = TimeDistributed(Dense(1)(X))
    else:
        X = Dense(1)(X)
        
    model = Model(inputs= X_input, outputs=X)
    return model



def CNN(input_shape=(60,60,1), output_shape=3, filters_list=[32,16,8], data_format='channels_last'):
    '''
    Input: A tensor, shape:(batch_size, width, height, channels)
    Output: A tensor,shape:(batch_size, 1) or (batch_size, time_length)
    input_shape : (time_length, number_features)
    output_shape: a real number or a sequence for each input
    units_list: number of units for each lstm layer
    return_seq: The return_sequences for lst
    '''

    X_input = Input(input_shape)
    X = Conv2D(64, kernel_size=(3,3), strides=(1, 1), 
               padding='same', data_format=data_format)(X_input)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)

    for filters in filters_list:
        X = Conv2D(filters=filters, kernel_size=(3,3), strides=(1, 1), padding='same', data_format=data_format)(X)
        X = Activation('relu')(X)
        X = BatchNormalization()(X)
    
    Output = Conv2D(filters=output_shape, kernel_size=(3,3), strides=(1, 1), padding='same', data_format=data_format, name='extra_output')(X)
    #Output = Activation('sigmoid')(Output)

    X = AveragePooling2D(pool_size=(2, 2))(Output)
    X =  Conv2D(filters=1024, kernel_size=(180,30), strides=(1, 1), padding='valid', data_format=data_format)(X)


    X = Flatten()(X)
    X = Dense(1, name='main_output')(X)
    model = Model(inputs=X_input, outputs=[X,Output])
    return model

def ST_CNN(input_shape, extra_input_shape=None, output_shape=1, filters_list=[128,256,512], data_format='channels_last'):
    '''
    Input: A tensor, shape:(batch_size, width, height, channels)
    Output: A tensor,shape:(batch_size, 1) or (batch_size, time_length)
    input_shape : (time_length, number_features)
    output_shape: a real number or a sequence for each input
    units_list: number of units for each lstm layer
    return_seq: The return_sequences for lst
    '''

    X_input = Input(input_shape)
    X = Conv2D(filters=256, kernel_size=(5,5), strides=(1, 1), padding='valid', dilation_rate=(3, 3), data_format=data_format)(X_input)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)


    for filters in filters_list:
        X = Conv2D(filters=filters, kernel_size=(5,5), strides=(1, 1), padding='valid', dilation_rate=(3, 3), data_format=data_format)(X)
        X = Activation('relu')(X)
        X = BatchNormalization()(X)

    X = Conv2D(filters=128, kernel_size=(1,1), strides=(1, 1), padding='valid', dilation_rate=(3, 3), data_format=data_format)(X)
    X = Activation('relu')(X)
    X = BatchNormalization()(X)

    X = AveragePooling2D(pool_size=(2, 2))(X)

    X = Flatten()(X)
    if extra_input_shape is not None:
        X_extra_input = Input(extra_input_shape)
        X = Concatenate()([X, X_extra_input])
    X = Dense(1024, activation='relu')(X)
    X = Dense(1024, activation='relu')(X)
    X = Dense(1)(X)
    if extra_input_shape is not None:
        model = Model(inputs = [X_input, X_extra_input], outputs = X)
    else: 
        model = Model(inputs=X_input, outputs=X)
    return model

def simple_VGG(input_shape, extra_input_shape=None, output_shape=1, filters_list=[128,64,32], data_format='channels_last'):
    '''
    Input: A tensor, shape:(batch_size, width, height, channels)
    Output: A tensor,shape:(batch_size, 1) or (batch_size, time_length)
    input_shape : (time_length, number_features)
    output_shape: a real number or a sequence for each input
    units_list: number of units for each lstm layer
    return_seq: The return_sequences for lst
    '''
    
    X_input = Input(input_shape)
    #Stage 1
    X = Conv2D(64, (3, 3), activation='relu', padding='same')(X_input)
    X = BatchNormalization()(X)
    X  = Conv2D(64, (3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = AveragePooling2D(pool_size=(2, 2))(X)
    #Stage 2
    X = Conv2D(128, (3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X  = Conv2D(128, (3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = AveragePooling2D(pool_size=(2, 2))(X)
    #Stage 3
    X = Conv2D(256, (3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X  = Conv2D(256, (3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = AveragePooling2D(pool_size=(2, 2))(X)
    #Stage 4
    X = Conv2D(512, (3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X  = Conv2D(512, (3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)
    X = AveragePooling2D(pool_size=(2, 2))(X)
    
    X  = Conv2D(256, (1, 1), activation='relu', padding='same')(X)
    X = BatchNormalization()(X)

    X = Flatten()(X)
    if extra_input_shape is not None:
        X_extra_input = Input(extra_input_shape)
        X = Concatenate()([X, X_extra_input])

    X = Dense(4096, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(4096, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(1)(X)
    if extra_input_shape is not None:
        model = Model(inputs = [X_input, X_extra_input], outputs = X)
    else: 
        model = Model(inputs=X_input, outputs=X)
    return model



def identity_block(X, f, filters, stage, block):

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', name = conv_name_base+'2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s = 2):

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = tf.keras.layers.Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)


    # Second component of main path (≈3 lines)
    X = tf.keras.layers.Conv2D(F2, (f,f), strides=(1,1), name = conv_name_base + '2b', padding='same' ,kernel_initializer= glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name = bn_name_base +'2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = tf.keras.layers.Conv2D(F3, (1,1), strides=(1,1), name=conv_name_base+'2c', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = tf.keras.layers.Conv2D(F3, (1,1), strides=(s,s), name=conv_name_base+'1', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def ResNet(input_shape, extra_input_shape=None):
    X_input = Input(input_shape)

    X = ZeroPadding2D((2, 2))(X_input)

    # Stage 1
    X = Conv2D(64, (5, 5), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    #X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    #X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters =  [512, 512, 512], stage = 5, block='a', s=2)
    X = identity_block(X, 3,  [512, 512, 512], stage=5, block='b')
    #X = identity_block(X, 3,  [512, 512, 512], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, name="avg_pool")(X)

    X = Flatten()(X)
    #X_extra_input = Input(extra_input_shape)
    #X = Concatenate(name='concat')([X, X_extra_input])

    X = Dense(1024, activation='relu')(X)
    X = Dropout(0.3)(X)
    X = Dense(1)(X)
    model = Model(inputs = X_input, outputs = X)
    return model


def Small_ResNet(input_shape, extra_input_shape=None):
    X_input = tf.keras.layers.Input(input_shape)

    X = tf.keras.layers.ZeroPadding2D((2, 2))(X_input)

    # Stage 1
    X = tf.keras.layers.Conv2D(64, (5, 5), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform())(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s=1)
    ### X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    ### X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s=2)
    ### X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    ### X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    ##X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s=2)
    ### X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    ### X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    ### X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    ##X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    ##X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters =  [512, 512, 512], stage = 5, block='a', s=2)
    ### X = identity_block(X, 3,  [512, 512, 512], stage=5, block='b')
    #X = identity_block(X, 3,  [512, 512, 512], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, name='avg_pool')(X)

    X = tf.keras.layers.Flatten()(X)
    #X_extra_input = Input(extra_input_shape)
    #X = Concatenate(name='concat')([X, X_extra_input])

    X = tf.keras.layers.Dense(256, activation='relu')(X)
    #X = tf.keras.layers.Dense(1024, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.3)(X)
    X = tf.keras.layers.Dense(1)(X)
    model = tf.keras.Model(inputs = X_input, outputs = X)
    return model


def Small_ResNet_name(input_shape, extra_input_shape=None):
    X_input = Input(input_shape)

    X = ZeroPadding2D((2, 2),name = 'L1')(X_input)

    # Stage 1
    X = Conv2D(64, (5, 5), strides = (2, 2), name = 'L2', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = 'L3')(X)
    X = Activation('relu', name = 'L4')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), name = 'L5')(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s=1, name = 'L6')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s=2, name = 'L7')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s=2, name = 'L8')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters =  [512, 512, 512], stage = 5, block='a', s=2, name = 'L9')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, name='L10')(X)

    X = Flatten(name = 'L11')(X)
    X = Dense(1024, activation='relu', name = 'L12')(X)
    X = Dropout(0.3, name = 'L13')(X)
    X = Dense(1, name = 'L14')(X)
    model = Model(inputs = X_input, outputs = X)
    return model
                                                                                 
