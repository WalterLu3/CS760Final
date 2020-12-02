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
def get_DNN_Model( feat_dim, label_dim ):
    
    model = Sequential()
    model.add( Dense( 1024,
                      activation='relu',
                      kernel_initializer='normal',
                      kernel_regularizer=regularizers.l2(1e-4),
                      input_dim = feat_dim
                    ))
    model.add( Dropout(0.1) )
    model.add( BatchNormalization() )
    
    model.add( Dense( 1024,
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
    model.compile( loss=sse,
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
                                      monitor='val_loss',
                                      mode='min'
                                    )
        hist = model.fit( X_train , Y_train,
                          validation_data=( X_valid, Y_valid ), 
                          epochs=epoch_num, batch_size=batch_size,
                          verbose=1,
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
        hist = []
        
    loss = DNN_model.evaluate( X_valid, Y_valid )
    print("\nValidation loss: " , loss )
    
    print('\nDNN model training and test data label prediction finished.')
    return DNN_model, hist
    
########################
###       Main       ###
########################


raw_label_data = pd.read_csv( "label_data.csv" )

id_table = np.array( raw_label_data["Admin_Code"].values )
cnfrm_case_ttl = np.array( raw_label_data["Total_Confirmed"].values )
state_name_ttl = raw_label_data["State"].values
state_cnty_name_ttl = raw_label_data["County"].values
cnty_id_ttl = raw_label_data["Admin_Code"].values

## check data num
data_num = 0
while(1):
    if( id_table[data_num] < 1e8 ):
        data_num = data_num + 1
    else:
        break

cnty_id, counts = np.unique( id_table[:data_num], return_counts=True )

cnty_num = len( cnty_id )
date_ttl = counts[0]
date_width = 10

cnfrm_case_ttl = cnfrm_case_ttl[:data_num].reshape( cnty_num, -1 )


## build label data of confirmed cases 
label_data = []
idx = date_width - 1
while(1):
    if( idx > cnfrm_case_ttl.shape[1] ):
        break
    elif( idx - date_width < 0 ):
        label_data.append( cnfrm_case_ttl[:,idx] )
        idx = idx + date_width
    else:
        label_data.append( cnfrm_case_ttl[:,idx] - cnfrm_case_ttl[:,idx-date_width] )
        idx = idx + date_width
        
label_data = np.vstack(label_data).T
label_data = label_data.astype('float')

FIPS_cnty_name_table = []
for i in range(cnty_num):
    FIPS_cnty_name_table.append( [ int(cnty_id_ttl[i*date_ttl]), str(state_name_ttl[i*date_ttl]), str(state_cnty_name_ttl[i*date_ttl]) ] ) 
    
del cnfrm_case_ttl, state_name_ttl, state_cnty_name_ttl


raw_feat_data = pd.read_csv( "feature_data.csv" )
state_name_ttl = raw_feat_data["State"].tolist()
state_cnty_name_ttl = raw_feat_data["County"].tolist()
raw_feat_data_value = raw_feat_data.values
crrspnd_table = []
crrspnd_idx_table = -1 * np.ones(cnty_num,dtype='int')
crrspnd_num = 0
for i in range(cnty_num):
    key_state = FIPS_cnty_name_table[i][1]
    key_state_cnty = FIPS_cnty_name_table[i][2]
    for j in range( len(state_cnty_name_ttl) ):
        if( key_state in state_name_ttl[j] and key_state_cnty in state_cnty_name_ttl[j] ):
            crrspnd_idx_table[i] = j
            crrspnd_table.append( [j, key_state, key_state_cnty ] )
            crrspnd_num = crrspnd_num + 1
            break

feat_data = []
to_del = []
for i in range(cnty_num):
    if( crrspnd_idx_table[i] == -1 ):
        to_del.append( i )
    else:
        feat_data.append( raw_feat_data_value[ crrspnd_idx_table[i], 2:  ].astype('float') )

feat_data = np.vstack( feat_data )
label_data = np.delete( label_data, to_del, axis=0 )    

# label_data = np.log( label_data + 1e-3 )

to_del = sorted(to_del, reverse=True)
for i in to_del:
    del FIPS_cnty_name_table[i]

cnty_num = len( label_data )
label_dim = label_data.shape[1]
feat_dim = feat_data.shape[1]
feat_data
model_path = "model/COVID_19(" + str(seeds[MODEL_NUMBER-1]) + ")"

min_label_data = np.min( label_data, axis=0 )
max_label_data = np.max( label_data, axis=0 )
nrmlz_label_data = np.copy( label_data )
for i in range(0,label_data.shape[1]):
    nrmlz_label_data[:,i] = ( label_data[:,i] - min_label_data[i] ) / ( max_label_data[i] - min_label_data[i] )

# nrmlz_label_data = np.log( nrmlz_label_data )
    
min_feat_data = np.min( feat_data, axis=0 )
max_feat_data = np.max( feat_data, axis=0 )
nrmlz_feat_data = np.copy( feat_data )
for i in range(0,feat_data.shape[1]): 
    nrmlz_feat_data[:,i] = ( feat_data[:,i] - min_feat_data[i] ) / ( max_feat_data[i] - min_feat_data[i] ) 


        
X_train , Y_train , X_valid , Y_valid = cut_valid( nrmlz_feat_data, nrmlz_label_data, int( valid_rate * cnty_num ) ) # cut out validation data        
DNN_model = get_DNN_Model( feat_dim, label_dim )
DNN_model, hist = train_DNN_model( DNN_model, X_train, Y_train, 
                                    X_valid, Y_valid, model_path, to_load_model=True )
if( len(hist) > 0 ):
    np.savetxt( model_path + "_loss.csv", hist.history['loss'] )
    np.savetxt( model_path + "_val_loss.csv", hist.history['val_loss'] )        

## Test ##
for i in range(10):
    idx = np.random.randint( X_valid.shape[0] )
    y_test = DNN_model.predict( X_valid[idx,:].reshape(1,-1) ).reshape(-1)
    plt.plot( y_test * ( max_label_data - min_label_data ) + min_label_data )
    plt.plot( Y_valid[idx,:] * ( max_label_data - min_label_data ) + min_label_data )
    plt.show()

        