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
Continuous labels' evaluation standard : MAPE
Binary Classification's evaluation standard : Cross-Entropy
## NN ##
1. cases_increase_10: 0.88281078
2. cases_increase_5: 0.90947305
3. cases_increase_1: 50.61628634, non-trainable
4. cases_seriousness_1 (classification): 0.77525842, non-trainable
5. deaths_increase_10: 0.66714937
6. deaths_increase_5: 0.73615916
7. deaths_increase_1: 0.95839941
8. deaths_seriousness_1 (classification): 0.74724908,trainable but not for valid_loss

## Random Forest ##
1. cases_increase_10: 0.0021357080800325145
2. cases_increase_5: 0.0038593331759570575
3. cases_increase_1: 0.014728564849270542
4. cases_seriousness_1 (classification): 
5. deaths_increase_10: 4.6659028876153586e-05
6. deaths_increase_5: 8.310205512667283e-05
7. deaths_increase_1: 0.0002782352006245808
8. deaths_seriousness_1 (classification):

## SVM ##
1. cases_increase_10: 0.029492560625535333
2. cases_increase_5: 0.04501633457449545
3. cases_increase_1: 0.04480892217406902
4. cases_seriousness_1 (classification): 
5. deaths_increase_10: 0.0007445946470889617
6. deaths_increase_5: 0.0012672066744095131
7. deaths_increase_1: 0.0030582337692961637
8. deaths_seriousness_1 (classification):

