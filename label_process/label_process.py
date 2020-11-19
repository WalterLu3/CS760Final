import json
import os
import pandas as pd
import math


df = None # coronavirus dataframe
population = None # election dataset
mapping_dict = None # county name mapping dictionry(map county to state and county)
population_mapping_dict = None # county population dictionary (map county)
usa_total_cases_ratio_11_16 = None
usa_total_deaths_ratio_11_16 = None


# You have to run init() the first time you imoprt label_process.py
# then you can get the label using y_label_generator

def init(): # have to run this first to get all data. Please avoid running it multiple times.
	# load covid datset
	global df
	global population
	global population_mapping_dict
	global mapping_dict
	global usa_total_cases_ratio_11_16 
	global usa_total_deaths_ratio_11_16 

	json_file_path = 'coronavirus-covid-19-pandemic-usa-counties.json'
	with open(json_file_path,'r') as f:
	    data = None
	    data = json.load(f)
	dataset = []
	for row in data:
	    dataset.append(row['fields'])
	data = None # save memory
	df = pd.DataFrame(dataset)

	#load election dataset
	json_file_path = 'usa-2016-presidential-election-by-county.json'
	with open(json_file_path,'r') as f:
	    data = None
	    data = json.load(f)
	population = data

	# create population dictionary
	population_mapping_dict = {}
	adjustment_ratio = 328/305
	for row in population:
	    population_mapping_dict[tuple(county_mapping(row['fields']['county']))]= math.floor(row['fields']['total_population']*adjustment_ratio)

	# creat county name mapping 
	
	mapping_dict = {}
	for row in population:
		mapping_dict[row['fields']['county']] = county_mapping(row['fields']['county'])
    # calculate  ratio
    # total_statics from us
	usa_total_population = sum(population_mapping_dict.values())
	usa_total_cases_11_16 = df[df['date'] == df['date'].max()]['tot_confirmed'].sum()
	usa_total_deaths_11_16 = df[df['date'] == df['date'].max()]['tot_death'].sum()

	usa_total_cases_ratio_11_16 = usa_total_cases_11_16/usa_total_population
	usa_total_deaths_ratio_11_16 = usa_total_deaths_11_16/usa_total_population




# county string mapping
# give a county string ex. 'Dane County, Wiscosin', return a list where the first element is state, an the 
# second is county ex. ['Madison','Dane']
def county_mapping(county_column):
    county_string = county_column
    county_string = county_string.split(',')
    for i in range(len(county_string)):
        if i == 0: ## county string
            county_string[i] = county_string[i].replace(" County", "")
        elif i == 1: ## state string
            county_string[i] = county_string[i][1:]
    state_name = county_string[1]
    county_name = county_string[0]
    return [state_name, county_name]

#input state, county, label type, and group date
#
#you can either use getmapping dictionary or get mapping function to get the input for state name and county name
#
#label_type has 6 different value
#1. "normalized_cases" 2. "normalized_deaths" 3.deaths_increase_rates  (per day) 
#4. cases_increase_rates(per day) 5. seriousness_label_cases (compared to whole US and is not affected by 
#group date) 6. seriousness_label_deaths (compared to whole US and is not affected by group date)
# 
# The return will be a list of label or labels for a county
#


def y_label_generator(state, county, label_type, group_date = 300):
    label_list = None
    target_state = df[df['province_state'] == state] 
    target_county = target_state[ target_state['admin2'] == county]
    target_county = target_county.sort_values(by = 'date')
    # target county will store a dataframe for a given county and in time order
    
    county_population = population_mapping_dict[(state,county)] # get total_population for nomalization
    
    
    if label_type == "normalized_cases": # ask for number of cases cut by each group data
        num_cuts = len(target_county)/group_date
        assert(num_cuts >= 1), "group_date is too big"
        label_list = []
        for i in range(math.floor(num_cuts)):
            if i <= math.floor(num_cuts) - 1:
                index = (i+1)*group_date-1
                label_list.append(target_county['tot_confirmed'].iloc[index])
        if math.floor(num_cuts) != num_cuts:
            label_list.append(target_county['tot_confirmed'].iloc[-1])
            
        ## perform normalization 
        for i in range(len(label_list)):
            label_list[i] = label_list[i]/county_population
       
    
    elif label_type == "normalized_deaths": # ask for number of deaths cut by each group data
        num_cuts = len(target_county)/group_date
        assert(num_cuts >= 1), "group_date is too big"
        label_list = []
        for i in range(math.floor(num_cuts)):
            if i <= math.floor(num_cuts) - 1:
                index = (i+1)*group_date-1
                label_list.append(target_county['tot_death'].iloc[index])
        if math.floor(num_cuts) != num_cuts:
            label_list.append(target_county['tot_death'].iloc[-1])         
        ## perform normalization 
        for i in range(len(label_list)):
            label_list[i] = label_list[i]/county_population
            
            
    elif label_type == "cases_increase_rates": # ask for the increase rate of cases cut by each group data
        num_cuts = len(target_county)/group_date
        assert(num_cuts >= 1), "group_date is too big"
        label_list = []
        for i in range(math.floor(num_cuts)):
            if i == 0:
                index = (i+1)*group_date-1
                label_list.append(target_county['tot_confirmed'].iloc[index]/group_date)
            elif i <= math.floor(num_cuts) - 1:
                index = (i+1)*group_date-1
                label_list.append((target_county['tot_confirmed'].iloc[index]- \
                                 target_county['tot_confirmed'].iloc[index-group_date])/group_date)
        if math.floor(num_cuts) != num_cuts:
            label_list.append((target_county['tot_confirmed'].iloc[-1] - \
                             target_county['tot_confirmed'].iloc[group_date*math.floor(num_cuts)-1])/ \
                              (len(target_county)-group_date*math.floor(num_cuts)))
            
            
    elif label_type == "deaths_increase_rates": # ask for the increase rate of deaths cut by each group data
        num_cuts = len(target_county)/group_date
        assert(num_cuts >= 1), "group_date is too big"
        label_list = []
        for i in range(math.floor(num_cuts)):
            if i == 0:
                index = (i+1)*group_date-1
                label_list.append(target_county['tot_death'].iloc[index]/group_date)
            elif i <= math.floor(num_cuts) - 1:
                index = (i+1)*group_date-1
                label_list.append((target_county['tot_death'].iloc[index]- \
                                 target_county['tot_death'].iloc[index-group_date])/group_date)
        if math.floor(num_cuts) != num_cuts:
            label_list.append((target_county['tot_death'].iloc[-1] - \
                             target_county['tot_death'].iloc[group_date*math.floor(num_cuts)-1])/ \
                              (len(target_county)-group_date*math.floor(num_cuts)))
      
    
    elif label_type == "seriousness_label_cases": 
        if target_county['tot_confirmed'].iloc[-1]/county_population > usa_total_cases_ratio_11_16:
            return [1]
        else:
            return [0]
        
    elif label_type == "seriousness_label_deaths": 
        if target_county['tot_death'].iloc[-1]/county_population > usa_total_deaths_ratio_11_16:
            return [1]
        else:
            return [0]       
    
    return label_list
