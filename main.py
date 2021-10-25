import pandas as pd
from sklearn.impute import SimpleImputer

X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

X_train = X_train.set_index('Id')
X_test = X_test.set_index('Id')
X_corr = X_train.corr()

y = X_train.SalePrice
X_train = X_train.drop('SalePrice',axis=1)
combine = pd.concat([X_train,X_test])

## combine = pd.get_dummies(combine,columns=['MSSubClass','MSZoning'])
combine['MSZoning'] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['MSZoning']])
combine['Street'] = combine['Street'].apply(lambda x: 1 if x=='Grvl' else 0)
combine['LotFrontage_Imputer'] = 0.2*combine['LotArea']+combine['1stFlrSF']+combine['TotalBsmtSF']+combine['GrLivArea']
LotFrontage_mean = combine['LotFrontage'].mean()
LotFrontage_std = combine['LotFrontage'].std()
LotFrontage_Imputer_mean = combine['LotFrontage_Imputer'].mean()
LotFrontage_Imputer_std = combine['LotFrontage_Imputer'].std()
combine['LotFrontage_Imputer'] = combine['LotFrontage_Imputer'].apply(lambda x: LotFrontage_mean + LotFrontage_std * (x-LotFrontage_Imputer_mean)/LotFrontage_Imputer_std)
combine['LotFrontage'] = combine['LotFrontage'].where(pd.notnull(combine['LotFrontage']),other=round(combine['LotFrontage_Imputer']))

combine.info()

combine_corr = combine.corr()

print(combine['MSZoning'].value_counts())



# MSZoning
## LotFrontage
## Alley
# Utilities
# Exterior1st
# Exterior2nd
## MasVnrType
## MasVnrArea
## BsmtQual
## BsmtCond
## BsmtExposure
## BsmtFinType1
# BsmtFinSF1
## BsmtFinType2
# BsmtFinSF2
# BsmtUnfSF
# TotalBsmtSF
## Electrical
# BsmtFullBath
# BsmtHalfBath
# KitchenQual
# Functional
## FireplaceQu
## GarageType
## GarageYrBlt
## GarageFinish
# GarageCars
# GarageArea
## GarageQual
## GarageCond
## PoolQC
## Fence
## MiscFeature
# SaleType

