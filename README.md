# CS760Final

# Related dataset #
1. US County Boundaries (The latitude and location may affect the disease spreading)
https://public.opendatasoft.com/explore/embed/dataset/us-county-boundaries/table/?disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name&sort=-stusab

2. USA 2016 Presidential Election by County
https://public.opendatasoft.com/explore/embed/dataset/usa-2016-presidential-election-by-county/table/?disjunctive.state

3. COVID-19 Pandemic - USA counties
https://public.opendatasoft.com/explore/embed/dataset/coronavirus-covid-19-pandemic-usa-counties/table/?disjunctive.province_state&disjunctive.admin2&sort=date


# Results #
rmse = sqrt( sum( sum( ( y_label_ij - y_label_ij / y_label_ij )^2, i = 1 to label_dim ), j = 1 to data_num ) / data_num / label_dim )
correct rate
## NN ##
1. cases_increase_10: 0.88281078
2. cases_increase_5: 0.90947305
3. cases_increase_1: 50.61628634, non-trainable
4. cases_seriousness_1: 0.77525842, non-trainable
5. deaths_increase_10: 0.66714937
6. deaths_increase_5: 0.73615916
7. deaths_increase_1: 0.95839941
8. deaths_seriousness_1: 0.74724908,trainable but not for valid_loss
