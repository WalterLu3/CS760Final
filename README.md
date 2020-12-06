# CS760Final

# Related dataset #
1. US County Boundaries (The latitude and location may affect the disease spreading)
https://public.opendatasoft.com/explore/embed/dataset/us-county-boundaries/table/?disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name&sort=-stusab

2. USA 2016 Presidential Election by County
https://public.opendatasoft.com/explore/embed/dataset/usa-2016-presidential-election-by-county/table/?disjunctive.state

3. COVID-19 Pandemic - USA counties
https://public.opendatasoft.com/explore/embed/dataset/coronavirus-covid-19-pandemic-usa-counties/table/?disjunctive.province_state&disjunctive.admin2&sort=date


# Increase Rate Results #
rmse = sqrt( sum( sum( ( y_label_ij - y_label_ij / y_label_ij )^2, i = 1 to label_dim ), j = 1 to data_num ) / data_num / label_dim )

correct rate

Continuous labels' evaluation standard : MAPE (2020/11/30, give up for zero in denominator)

Continuous labels' evaluation standard : MAE (2020/11/30)

Binary Classification's evaluation standard : Accuracy

# Results before Feature Selection #
## NN ##
1. cases_increase_10: 0.00163403
2. cases_increase_5: 0.00297517
3. cases_increase_1: 0.06507973, non-trainable
4. cases_seriousness_1 (classification): 0.77525842, non-trainable
5. deaths_increase_10: 5.577e-05
6. deaths_increase_5: 9.955e-05
7. deaths_increase_1: 0.00266079
8. deaths_seriousness_1 (classification): 0.74724908,trainable but not for valid_loss

## Random Forest (on testing data) ##
1. cases_increase_10: 0.0021357080800325145
2. cases_increase_5: 0.0038593331759570575
3. cases_increase_1: 0.014728564849270542
4. cases_seriousness_1 (classification): 0.861
5. deaths_increase_10: 4.6659028876153586e-05
6. deaths_increase_5: 8.310205512667283e-05
7. deaths_increase_1: 0.0002782352006245808
8. deaths_seriousness_1 (classification): 0.833

## SVM (on testing data) ##
1. cases_increase_10: 0.029492560625535333
2. cases_increase_5: 0.04501633457449545
3. cases_increase_1: 0.04480892217406902
4. cases_seriousness_1 (classification): 0.875
5. deaths_increase_10: 0.0007445946470889617
6. deaths_increase_5: 0.0012672066744095131
7. deaths_increase_1: 0.0030582337692961637
8. deaths_seriousness_1 (classification): 0.944

## Logtistic Regression (on testing data) ##

4. cases_seriousness_1 (classification): 0.50
8. deaths_seriousness_1 (classification): 0.861

## Linear Regression (on testing data) ##

1. cases_increase_10: 0.0027065228878186917
2. cases_increase_5: 0.004983150523837751
3. cases_increase_1: 0.017813254536689346
4. NAN
5. deaths_increase_10: 4.6564476462268724e-05
6. deaths_increase_5: 8.22592684093128e-05
7. deaths_increase_1: 0.0002578075074066752
8. NAN

# Results after Feature Selection (We only use top 10 features) #

## NN ##
1. cases_increase_10: 
2. cases_increase_5: 
3. cases_increase_1: 
4. cases_seriousness_1 (classification): 
5. deaths_increase_10: 
6. deaths_increase_5: 
7. deaths_increase_1: 
8. deaths_seriousness_1 (classification):

## Random Forest (on testing data) ##
1. cases_increase_10: 0.0025085474098432265
2. cases_increase_5: 0.004706894720148842
3. cases_increase_1: 0.015550907630249574
4. cases_seriousness_1 (classification): 0.833
5. deaths_increase_10: 5.097310475719473e-05
6. deaths_increase_5: 8.802277105993334e-05
7. deaths_increase_1: 0.0002961636956821826
8. deaths_seriousness_1 (classification): 0.805 

## SVM (on testing data) ##
1. cases_increase_10: 0.029492560625535333
2. cases_increase_5: 0.04501633457449545
3. cases_increase_1: 0.036717001001727576
4. cases_seriousness_1 (classification): 0.083
5. deaths_increase_10: 0.0007445946470889617
6. deaths_increase_5: 0.0012672066744095131
7. deaths_increase_1: 0.0030582337692961637
8. deaths_seriousness_1 (classification): 0.861

## Logtistic Regression (on testing data) ##

4. cases_seriousness_1 (classification): 0.54
8. deaths_seriousness_1 (classification): 0.861

## Linear Regression (on testing data) ##

1. cases_increase_10: 0.0026883700146923214
2. cases_increase_5: 0.00497572128272232
3. cases_increase_1: 0.018138018325361424
4. NAN
5. deaths_increase_10: 4.701973409597593e-05
6. deaths_increase_5: 8.47698810552995e-05
7. deaths_increase_1: 0.0002588867651615653
8. NAN
