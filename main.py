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

## Ordinal Encoding

def impute_nan(x):
    if pd.isnull(x):
        x=0
    return x

lotshape_enc={'Reg':1,'IR1':2,'IR2':3,'IR3':4}    
landslope_enc={'Gtl':1,'Mod':2,'Sev':3}
exter_enc={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
bsmtexposure_enc={'No':1,'Mn':2,'Av':3,'Gd':4}
bsmtfintype_enc={'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
functional_enc={'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7}
garagefinish_enc={'Unf':1,'RFn':2,'Fin':3}
combine['LotShape'] = combine['LotShape'].map(lotshape_enc)
combine['LandSlope'] = combine['LandSlope'].map(landslope_enc)
combine['ExterQual'] = combine['ExterQual'].map(exter_enc)
combine['ExterCond'] = combine['ExterCond'].map(exter_enc)
combine['BsmtQual'] = combine['BsmtQual'].map(exter_enc).apply(lambda x: impute_nan(x))
combine['BsmtCond'] = combine['BsmtCond'].map(exter_enc).apply(lambda x: impute_nan(x))
combine['BsmtExposure'] = combine['BsmtExposure'].map(bsmtexposure_enc).apply(lambda x: impute_nan(x))
combine['BsmtFinType1'] = combine['BsmtFinType1'].map(bsmtfintype_enc).apply(lambda x: impute_nan(x))
combine['BsmtFinType2'] = combine['BsmtFinType2'].map(bsmtfintype_enc).apply(lambda x: impute_nan(x))
combine['HeatingQC'] = combine['HeatingQC'].map(exter_enc)
combine['CentralAir'] = combine['CentralAir'].apply(lambda x: 1 if x=='Y' else 0)
combine['KitchenQual'] = combine['KitchenQual'].map(exter_enc).apply(lambda x: 3 if pd.isnull(x) else x).astype(int)
combine['Functional'] = combine['Functional'].map(functional_enc)
combine['FireplaceQu'] = combine['FireplaceQu'].map(exter_enc).apply(lambda x: impute_nan(x))
combine['GarageFinish'] = combine['GarageFinish'].map(garagefinish_enc)
combine['GarageQual'] = combine['GarageQual'].map(exter_enc)
combine['GarageCond'] = combine['GarageCond'].map(exter_enc)

## Imputing
combine['MSZoning'] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['MSZoning']])
combine['Street'] = combine['Street'].apply(lambda x: 1 if x=='Grvl' else 0)
combine['LotFrontage_Imputer'] = 0.2*combine['LotArea']+combine['1stFlrSF']+combine['TotalBsmtSF']+combine['GrLivArea']
LotFrontage_mean = combine['LotFrontage'].mean()
LotFrontage_std = combine['LotFrontage'].std()
LotFrontage_Imputer_mean = combine['LotFrontage_Imputer'].mean()
LotFrontage_Imputer_std = combine['LotFrontage_Imputer'].std()
combine['LotFrontage_Imputer'] = combine['LotFrontage_Imputer'].apply(lambda x: LotFrontage_mean + LotFrontage_std * (x-LotFrontage_Imputer_mean)/LotFrontage_Imputer_std)
combine['LotFrontage'] = combine['LotFrontage'].where(pd.notnull(combine['LotFrontage']),other=round(combine['LotFrontage_Imputer']))
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
combine[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']] = combine[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']].applymap(lambda x: impute_nan(x))
combine['Electrical'] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['Electrical']])
combine[['BsmtFullBath','BsmtHalfBath']] = combine[['BsmtFullBath','BsmtHalfBath']].applymap(lambda x: impute_nan(x)).astype(int)
combine['Functional'] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['Functional']])
combine['GarageYrBlt'] = combine['GarageYrBlt'].where(pd.notnull(combine['GarageYrBlt']) | pd.isnull(combine['GarageType']), other=combine['YearBuilt'])
combine['GarageYrBlt'] = combine['GarageYrBlt'].apply(lambda x: impute_nan(x))
combine['GarageFinish'] = combine['GarageFinish'].where(combine['GarageYrBlt']!=0,other=0)
combine['GarageFinish'] = combine['GarageFinish'].apply(lambda x: 1 if pd.isnull(x) else x)
combine['GarageCars'] = combine['GarageCars'].apply(lambda x: 1 if pd.isnull(x) else x)
combine['GarageArea'] = combine['GarageArea'].apply(lambda x: round(combine[['GarageCars','GarageArea']].groupby('GarageCars').mean().GarageArea.iloc[1]) if pd.isnull(x) else x)
combine['GarageQual'] = combine['GarageQual'].where(combine['GarageYrBlt']!=0,other=0)
combine['GarageQual'] = combine['GarageQual'].apply(lambda x: 3 if pd.isnull(x) else x)
combine['GarageCond'] = combine['GarageCond'].where(combine['GarageYrBlt']!=0,other=0)
combine['GarageCond'] = combine['GarageCond'].apply(lambda x: 3 if pd.isnull(x) else x)

## Dropping Features
combine.drop(['LotFrontage_Imputer','Utilities','Condition2'],axis=1,inplace=True)

combine.info()

## Label Encoding
# combine = pd.get_dummies(combine,dummy_na=True,columns=['Alley','GarageType'])
# combine = pd.get_dummies(combine,columns=['MSSubClass','MSZoning','LandContour','LotConfig','Neighborhood',
# 'Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
# 'Foundation','Heating','Electrical'])

combine_corr = combine.corr()
print(combine['GarageCond'].value_counts())
print(combine['GarageQual'].loc[pd.isnull(combine['GarageQual']) & pd.notnull(combine['GarageType'])])

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

