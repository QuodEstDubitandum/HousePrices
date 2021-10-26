import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

X_train.set_index('Id',inplace=True)
X_test.set_index('Id',inplace=True)
X_corr = X_train.corr()

y = X_train.SalePrice
X_train.drop('SalePrice',axis=1,inplace=True)
combine = pd.concat([X_train,X_test])

lotshape_enc={'Reg':1,'IR1':2,'IR2':3,'IR3':4}    
landslope_enc={'Gtl':1,'Mod':2,'Sev':3}
combine['LotShape'] = combine['LotShape'].map(lotshape_enc)
combine['LandSlope'] = combine['LandSlope'].map(landslope_enc)

## combine = pd.get_dummies(combine,columns=['MSSubClass','MSZoning','LandContour','LotConfig','Neighborhood',
## 'Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd'])
## combine = pd.get_dummies(combine,dummy_na=True,columns=['Alley'])
combine['MSZoning'] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['MSZoning']])
combine['Street'] = combine['Street'].apply(lambda x: 1 if x=='Grvl' else 0)
combine['LotFrontage_Imputer'] = 0.2*combine['LotArea']+combine['1stFlrSF']+combine['TotalBsmtSF']+combine['GrLivArea']
LotFrontage_mean = combine['LotFrontage'].mean()
LotFrontage_std = combine['LotFrontage'].std()
LotFrontage_Imputer_mean = combine['LotFrontage_Imputer'].mean()
LotFrontage_Imputer_std = combine['LotFrontage_Imputer'].std()
combine['LotFrontage_Imputer'] = combine['LotFrontage_Imputer'].apply(lambda x: LotFrontage_mean + LotFrontage_std * (x-LotFrontage_Imputer_mean)/LotFrontage_Imputer_std)
combine['LotFrontage'] = combine['LotFrontage'].where(pd.notnull(combine['LotFrontage']),other=round(combine['LotFrontage_Imputer']))
combine.drop(['LotFrontage_Imputer','Utilities','Condition2'],axis=1,inplace=True)
combine[['Exterior1st','Exterior2nd']] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['Exterior1st','Exterior2nd']])

def masvnrtype_imp(x):
    if x<1050:
        x='None'
    elif x<1330:
        x='BrkFace'
    else:
        x='Stone'
    return x

combine['MasVnrType'] = combine['MasVnrType'].where(combine['MasVnrType'].notnull(),other=combine['TotalBsmtSF'].apply(lambda x: masvnrtype_imp(x)))
masvnrarea_list = (combine[['MasVnrType','MasVnrArea']].groupby(['MasVnrType']).mean().MasVnrArea.round()-1).tolist()
masvnrarea_dict = {'BrkCmn':masvnrarea_list[0],'BrkFace':masvnrarea_list[1],'None':masvnrarea_list[2],'Stone':masvnrarea_list[3]}
combine['MasVnrArea'] = combine['MasVnrArea'].where(combine['MasVnrArea'].notnull(),other=combine['MasVnrType'].map(masvnrarea_dict))
combine.info()

combine_corr = combine.corr()
print(combine[['MasVnrType','MasVnrArea']].groupby(['MasVnrType']).mean())
print(combine['MasVnrArea'].value_counts())



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

