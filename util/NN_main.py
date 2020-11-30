    # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from NN_library import sse, cut_valid, get_DNN_Model, PlotTrainingProcess
from NN_library import train_DNN_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MODEL_NUMBER = 1
seeds = [ 1234 , 123 , 445 , 2500, 1111 ]
np.random.seed( seeds[MODEL_NUMBER-1] )

## DNN training parameter
valid_rate = 0.1
epoch_num = 100
batch_size = 64


########################
###       Main       ###
########################

## build label data of confirmed_cases/deaths of increase rates/seriousness
'''
########## Old #################
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
########## Old #################
'''
label_data_path = "cases_increase_10"
raw_label_data = np.loadtxt( "../label_process/labels/" + label_data_path + ".csv",
                             dtype=np.object, delimiter = "," )
# raw_label_data = pd.read_csv( "../label_process/labels/cases_increase_rates_N_10.csv" )
# foo = raw_label_data.values
FIPS_cnty_name_table = raw_label_data[:,0:3].astype(str).tolist()
for i in FIPS_cnty_name_table:
    i[0] = int(i[0])
label_data = raw_label_data[:,3:].astype(float)
cnty_num = label_data.shape[0]

raw_feat_data = pd.read_csv( "../feature_process/feature_data.csv" )
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

to_del = sorted(to_del, reverse=True)
for i in to_del:
    del FIPS_cnty_name_table[i]


### data pre-processing ###
cnty_num = len( label_data )
label_dim = label_data.shape[1]
feat_dim = feat_data.shape[1]
model_path = "../models/NN/"+ label_data_path + "(" + str(seeds[MODEL_NUMBER-1]) + ")" 

# label_data = np.log( label_data + 1e-3 )

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

## Use 10-fold cross validation 
### Shuffle
index_table = np.arange(len(nrmlz_label_data))
X_and_Y_train_and_ID = np.concatenate( ( nrmlz_feat_data,
                                         nrmlz_label_data,
                                         index_table.reshape(-1,1) ), axis=1 )
np.random.shuffle( X_and_Y_train_and_ID )
X_train = np.copy( X_and_Y_train_and_ID[ :, 0 : - label_dim - 1 ] )
Y_train = np.copy( X_and_Y_train_and_ID[ :, - label_dim - 1 : -1 ] )
index_table = np.copy( X_and_Y_train_and_ID[ :,-1] ).reshape(-1)      

### K-fold cross-validation
K = 5
NN_model_set = []
mse_test = 0.
data_num_K = int( np.floor( len( Y_train ) / K ) )
Y_prdct = []
index_prdct = []
for i in range(K):
    NN_model = get_DNN_Model( feat_dim, label_dim )
    if( i == 0 ):
        X_test = X_train[  : data_num_K, : ]
        Y_test = Y_train[  : data_num_K, : ]
        index_test = index_table[  : data_num_K ]        
        NN_model_fold, hist = train_DNN_model( NN_model, X_train[ data_num_K : , : ], Y_train[ data_num_K : , : ], 
                                               X_test, Y_test, model_path + "_" + str(i), to_load_model=False )
        
        
    elif( i == K - 1 ):
        X_test = X_train[ i * data_num_K : , : ]
        Y_test = Y_train[ i * data_num_K : , : ]
        index_test = index_table[ i * data_num_K : ]
        NN_model_fold, hist = train_DNN_model( NN_model, X_train[  : i * data_num_K, : ], Y_train[  : i * data_num_K, : ], 
                                               X_test, Y_test, model_path + "_" + str(i), to_load_model=False )

    else:
        X_test = X_train[ i * data_num_K : ( i + 1 ) * data_num_K, : ]
        Y_test = Y_train[ i * data_num_K : ( i + 1 ) * data_num_K, : ]
        index_test = index_table[ i * data_num_K : ( i + 1 ) * data_num_K ]
        NN_model_fold, hist = train_DNN_model( NN_model, np.concatenate( ( X_train[ : i * data_num_K, : ],
                                                         X_train[ (i+1) * data_num_K : , : ] ), axis=0 ),
                                                         np.concatenate( ( Y_train[ : i * data_num_K ],
                                                         Y_train[ (i+1) * data_num_K : , : ] ), axis=0 ),
                                               X_test, Y_test, model_path + "_" + str(i), to_load_model=False )

    if( hist != None ):
        np.savetxt( model_path + "_" + str(i) + "_loss.csv",
                    np.array( [ hist.history['loss'], hist.history['val_loss'] ] ).T,
                    delimiter="," )
    NN_model_set.append( NN_model_fold )
    
    
    Y_prdct_fold = NN_model_fold.predict( X_test )
    Y_prdct.append( Y_prdct_fold )
    index_prdct.append( index_test )
    mse_test = mse_test + np.sum( ( Y_prdct_fold - Y_test )**2 )
mse_test = mse_test / cnty_num          
print( "K-fold's correct rate: ", np.round( mse_test, 8 ) )
Y_prdct_shuffled = np.vstack(Y_prdct)

Y_prdct = []
for i in range(cnty_num):
    Y_prdct.append( Y_prdct_shuffled[ np.where(index_table==i)[0][0], : ] )
np.savetxt( "../results/NN_" + label_data_path + "_pridict_mse", prdct_data, delimiter="," );

Y_prdct = np.vstack( Y_prdct )
prdct_data = np.copy( Y_prdct )
for i in range( prdct_data.shape[1] ): 
    prdct_data[:,i] = Y_prdct[:,i] * ( max_label_data[i] - min_label_data[i] ) + min_label_data[i]
np.savetxt( "../results/NN_" + label_data_path + "_pridict.csv", prdct_data, delimiter="," );

''' 
X_train , Y_train , X_valid , Y_valid = cut_valid( nrmlz_feat_data, nrmlz_label_data, int( valid_rate * cnty_num ) ) # cut out validation data        
DNN_model = get_DNN_Model( feat_dim, label_dim )
DNN_model, hist = train_DNN_model( DNN_model, X_train, Y_train, 
                                    X_valid, Y_valid, model_path, to_load_model=False )
if( hist != None ):
    np.savetxt( model_path + "_training_loss.csv",
                np.array( [ hist.history['loss'], hist.history['val_loss'] ] ).T,
                delimiter="," )
'''
## Test ##
for i in range(10):
    idx = np.random.randint( cnty_num )
    plt.plot( prdct_data[idx,:] )
    plt.plot( label_data[idx,:] )
    plt.legend(['predict', 'real'], loc='upper left')
    plt.show()


        