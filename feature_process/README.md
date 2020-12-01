# Details

aligned_features.csv is ordered by fips code. So are all the label data in "label_process/labels". So you can directly map them. There are 2999 data in total. When you split training and testing set(Wisconsin), you just need to filter out the data whose fips code is 55000<= fips < 56000.