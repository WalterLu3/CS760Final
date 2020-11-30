# CS760Final

# Related dataset #
1. US County Boundaries (The latitude and location may affect the disease spreading)
https://public.opendatasoft.com/explore/embed/dataset/us-county-boundaries/table/?disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name&sort=-stusab

2. USA 2016 Presidential Election by County
https://public.opendatasoft.com/explore/embed/dataset/usa-2016-presidential-election-by-county/table/?disjunctive.state

3. COVID-19 Pandemic - USA counties
https://public.opendatasoft.com/explore/embed/dataset/coronavirus-covid-19-pandemic-usa-counties/table/?disjunctive.province_state&disjunctive.admin2&sort=date


# Results #
mse = sum( sum( ( y_label_ij - y_label_ij )^2, i = 1 to label_dim ), j = 1 to data_num ) / data_num
## NN ##
1. cases_increase_10: 0.02840063
2. cases_increase_5: 0.01449544
