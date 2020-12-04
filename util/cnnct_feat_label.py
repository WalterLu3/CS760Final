# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# label_data_path = "cases_increase_10"
# label_data_path = "cases_increase_5"
# label_data_path = "cases_increase_1"
# label_data_path = "cases_seriousness_1"
# label_data_path = "deaths_increase_10"
# label_data_path = "deaths_increase_5"
# label_data_path = "deaths_increase_1"
label_data_path = "deaths_seriousness_1"


if( "seriousness" in label_data_path ):
    problem_type = "classification"
else:
    problem_type = "regression"

# load the raw label data and extract the FIPS-county-name table to build a list
raw_label_data = np.loadtxt( "../label_process/labels/" + label_data_path + ".csv",
                             dtype=np.object, delimiter = "," )

FIPS_cnty_name_table = raw_label_data[:,0:3].astype(str).tolist()
for i in FIPS_cnty_name_table: # transfer FIPS code string into int
    i[0] = int(i[0])

if( problem_type == "classification" ):
    label_data = raw_label_data[:,3:].astype(int)
else:
    label_data = raw_label_data[:,3:].astype(float)

cnty_num = label_data.shape[0]

# load the raw feature data and extract the state name, county name, and the value contents
raw_feat_data = pd.read_csv( "../feature_process/feature_data.csv" )
state_name_ttl = raw_feat_data["State"].tolist()
state_cnty_name_ttl = raw_feat_data["County"].tolist()
raw_feat_data_value = raw_feat_data.values

# Iteratively search whether the state-county name pair of FIPS-county-name table
# matchs the state-county name pair from feature data by brute force method, and count
# the number of successful matching
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

# build a to-delete list of non-matched counties in FIPS-county-name table
# rearrange the order of those matched counties from feature data to make them correspond
# to the order of FIPS-county-name table
feat_data = []
to_del = []
for i in range(cnty_num):
    if( crrspnd_idx_table[i] == -1 ):
        to_del.append( i )
    else:
        feat_data.append( np.copy( raw_feat_data_value[ crrspnd_idx_table[i], 2:  ] ).astype('float') )

feat_data = np.vstack( feat_data )
# delete the rows values of those unmatched counties in label data
label_data = np.delete( label_data, to_del, axis=0 )    

# reversely delete those unmatched counties in FIPS-county-name table
to_del = sorted(to_del, reverse=True)
for i in to_del:
    del FIPS_cnty_name_table[i]

