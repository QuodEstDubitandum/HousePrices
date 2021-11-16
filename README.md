# Estimation of House Prices using a Kaggle dataset

This repository includes a dataset from Kaggle (train.csv, test.csv) that shows different features of houses.
The task is to predict the price of houses in the test set (test.csv) while given the prices of houses in the training set (train.csv).
For this, I only tried a single ML model (Lasso regression) since investing time into simply optimizing parameters of other ML models and
potentially finding a better model for this case did not seem like a good learning experience to me, so I decided to use this time 
to learn other stuff instead.
Since there are a total of 80 features, Kaggle provided a .txt file (data_description.txt), which gives an overview of those features
and the possible values they can take on.
The final result was enough to place me in the top 20% of competitors, which seems acceptable for the time investment.

## Imputing 

Overall, mostly only a couple of values were missing which I mostly imputed by either replacing the empty cells by 0.
Sometimes it was more benifical to use the median along the feature instead. 
In the special case of the 'LotFrontage' feature, I looked at the correlation between the feature and other features and 
used a combination of features that had high correlation with 'LotFrontage' to create a new feature that I used to replace the 
missing values of 'LotFrontage'.
Another special case was 'MasVnrType'. I used the feature 'TotalBsmtSF' (which had a high correlation with 'MasVnrType') to 
replace missing values based on the value of 'TotalBsmtSF' the house with the missing value took on.

## Encoding 

Besides the typical ordinal encoding and one-hot encoding done on either ordinal or nominal variables, there were also
special cases, which were:

1) 'MSSubClass' was a integer variable, but had to be encoded as a nominal variable, since the integers did not have
any relationship (170 was neither better nor worse than 50 f.e.)
2) 'MoSold' was also an integer variable which had to be one-hot encoded since selling a house in July does not mean
that it gets a higher price than January f.e.
3) 'YrSold' had only 5 different values, so treating it as a categorical variable also made more sense

## Feature Engineering

I created 8 new features to capture key characteristics of the houses. 
To give an example: 
I added the values from 'GrLivArea' and 'TotalBsmtSF' to add the square feet from above ground property and basement property
giving us a better feature which captures the whole square feet of the house. 

## Scaling/Transformation

Features that were not nominal variables and were skewed (to the right or left) were transformed using the box-cox transformation
to make them follow a distribution more akin to a normal distribution. 
After that, the data was scaled using RobustScaler since Lasso Regression uses regularization in which the features should
be on a similar scale. 

## Hyperparameter Tuning

The parameters of Lasso Regression were tuned by using Grid Search with a 5-fold CV.