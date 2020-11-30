    # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import backend as K

MODEL_NUMBER = 1
seeds = [ 5461 , 123 , 445 , 2500, 1111 ]
np.random.seed( seeds[MODEL_NUMBER-1] )

## DNN training parameter
valid_rate = 0.1
epoch_num = 100
batch_size = 64

## self-defined loss function to magnify the loss during training
def sse(y_true, y_pred):
    """ sum of sqaured errors. """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    
    return K.log( K.sum( K.square( y_pred - y_true ), axis=-1 ) )


## Randomly split validation data from training data bcz "split" is before "shuffle" in "model.fit"
def cut_valid( data_X, data_Y, valid_num ):
    
    X = np.copy( data_X )
    Y = np.copy( data_Y )
    
    # Randomly rearrange the train data
    train_num = len( X )
    
    for i in range( train_num ):
        
        dice = np.random.randint( 0, train_num - 1 )
        X[ [ i, dice ] ] = X[ [ dice, i ] ]
        Y[ [ i, dice ] ] = Y[ [ dice, i ] ]

    X_valid = X[ ( train_num - valid_num ):, : ]
    X_train = X[ :( train_num - valid_num ), : ]
    Y_valid = Y[ ( train_num - valid_num ):, : ] 
    Y_train = Y[ :( train_num - valid_num ), : ]
    
    print('validation data built')
    return X_train , Y_train , X_valid , Y_valid


## construct DNN model
def get_DNN_Model( feat_dim, label_dim, problem_type ):
    
    if( problem_type == "regression" ):
        loss_func = sse
    elif( problem_type == "classification" ):
        loss_func = 'binary_crossentropy'
    
    model = Sequential()
    model.add( Dense( 256,
                      activation='relu',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4),
                      input_dim = feat_dim
                    ))
    model.add( Dropout(0.1) )
    model.add( BatchNormalization() )    
    model.add( Dense( 256,
                      activation='relu',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4)
                    ))
    model.add( Dropout(0.1) )
    model.add( BatchNormalization() )   
    model.add( Dense( label_dim,
                      activation='sigmoid',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4)
                    ))
    model.summary()
                 
#    adam = Adam( lr=0.001, decay=1e-6, clipvalue=0.5 )
    model.compile( loss=loss_func,
                   optimizer='adam',
                 )    
    print('DNN model built')
    return model


## plot the DNN model training process
def PlotTrainingProcess( hist ):
    
    plt.plot( hist.history['loss'] )
    plt.plot( hist.history['val_loss'] )
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


## train the DNN model
def train_DNN_model( model, X_train, Y_train, X_valid, Y_valid, model_path, to_load_model ):
        
    if( to_load_model==False ):        
        model.summary()   
        
        ## some callbacks
        #    earlystopping = EarlyStopping(monitor='val_acc', patience = 10, verbose=1, mode='max')
        checkpoint = ModelCheckpoint( filepath=( model_path + "_w_best.hdf5" ),
                                      verbose=0,
                                      save_best_only=True,
                                      save_weights_only=True,
                                      monitor='loss',
                                      mode='min'
                                    )
        hist = model.fit( X_train , Y_train,
                          validation_data=( X_valid, Y_valid ), 
                          epochs=epoch_num, batch_size=batch_size,
                          verbose=0,
                          shuffle=True,
        #                      callbacks=[ earlystopping, checkpoint ],
                          callbacks=[ checkpoint ]
                        )
        PlotTrainingProcess( hist )
        model.load_weights( model_path + "_w_best.hdf5" )
        model.save( model_path + '.h5' )
        DNN_model = model
        
    else:
        DNN_model = load_model( model_path + '.h5' )
        hist = None
        
    loss = DNN_model.evaluate( X_valid, Y_valid )
    print("\nValidation loss: " , loss )
    
    print('\nDNN model training and test data label prediction finished.')
    return DNN_model, hist

        