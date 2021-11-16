import sys
sys.path.append('C:\\Users\\User\\Documents\\scripts.py\\DataScience_Ensemble')
from my_own_methods import impute_nan,correlations,grid_search,ignore_warnings,feature_importances,skewed_transform
ignore_warnings()

import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import RobustScaler

X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
X_train_old = X_train

X_train.set_index('Id',inplace=True)
X_test.set_index('Id',inplace=True)

## Correlation between features and target variable
correlation_coefficients = correlations(X_train,'SalePrice',False)

y = X_train.SalePrice
m = y.size
X_train.drop('SalePrice',axis=1,inplace=True)
combine = pd.concat([X_train,X_test])

## Ordinal Variables Encoding / Imputing
lotshape_enc={'Reg':1,'IR1':2,'IR2':3,'IR3':4}    
landslope_enc={'Gtl':1,'Mod':2,'Sev':3}
exter_enc={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
bsmtexposure_enc={'No':1,'Mn':2,'Av':3,'Gd':4}
bsmtfintype_enc={'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
functional_enc={'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7}
garagefinish_enc={'Unf':1,'RFn':2,'Fin':3}
paved_enc={'N':0,'P':1,'Y':2}
poolqc_enc={'Fa':1,'TA':2,'Gd':3,'Ex':4}
fence_enc={'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4}
combine['LotShape'] = combine['LotShape'].map(lotshape_enc)
combine['LandSlope'] = combine['LandSlope'].map(landslope_enc)
combine['ExterQual'] = combine['ExterQual'].map(exter_enc)
combine['ExterCond'] = combine['ExterCond'].map(exter_enc)
combine['BsmtQual'] = combine['BsmtQual'].map(exter_enc).apply(lambda x: impute_nan(x,0))
combine['BsmtCond'] = combine['BsmtCond'].map(exter_enc).apply(lambda x: impute_nan(x,0))
combine['BsmtExposure'] = combine['BsmtExposure'].map(bsmtexposure_enc).apply(lambda x: impute_nan(x,0))
combine['BsmtFinType1'] = combine['BsmtFinType1'].map(bsmtfintype_enc).apply(lambda x: impute_nan(x,0))
combine['BsmtFinType2'] = combine['BsmtFinType2'].map(bsmtfintype_enc).apply(lambda x: impute_nan(x,0))
combine['HeatingQC'] = combine['HeatingQC'].map(exter_enc)
combine['CentralAir'] = combine['CentralAir'].apply(lambda x: 1 if x=='Y' else 0)
combine['KitchenQual'] = combine['KitchenQual'].map(exter_enc).apply(lambda x: impute_nan(x,3))
combine['Functional'] = combine['Functional'].map(functional_enc)
combine['FireplaceQu'] = combine['FireplaceQu'].map(exter_enc).apply(lambda x: impute_nan(x,0))
combine['GarageFinish'] = combine['GarageFinish'].map(garagefinish_enc).apply(lambda x: impute_nan(x,0))
combine['GarageQual'] = combine['GarageQual'].map(exter_enc)
combine['GarageCond'] = combine['GarageCond'].map(exter_enc)
combine['PavedDrive'] = combine['PavedDrive'].map(paved_enc)
combine['Fence'] = combine['Fence'].map(fence_enc).apply(lambda x: impute_nan(x,0))

## Imputing
combine['MSZoning'] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['MSZoning']])
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
combine[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']] = combine[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']].applymap(lambda x: impute_nan(x,0))
combine['Electrical'] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['Electrical']])
combine[['BsmtFullBath','BsmtHalfBath']] = combine[['BsmtFullBath','BsmtHalfBath']].applymap(lambda x: impute_nan(x,0))
combine['Functional'] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['Functional']])
combine['GarageYrBlt'] = combine['GarageYrBlt'].apply(lambda x: impute_nan(x,0))
combine['GarageCars'] = combine['GarageCars'].apply(lambda x: impute_nan(x,0))
combine['GarageArea'] = combine['GarageArea'].apply(lambda x: impute_nan(x,0))
combine['GarageQual'] = combine['GarageQual'].apply(lambda x: impute_nan(x,0))
combine['GarageCond'] = combine['GarageCond'].apply(lambda x: impute_nan(x,0))
combine['SaleType'] = SimpleImputer(strategy='most_frequent').fit_transform(combine[['SaleType']])

## Dropping some Features
combine.drop(['LotFrontage_Imputer','Utilities','Condition2','PoolQC','PoolArea','Street'],axis=1,inplace=True)

## Creating new Features
combine['TotalQuality'] = combine['OverallQual'] + combine['OverallCond']
combine['ExterQuality'] = combine['ExterQual'] + combine['ExterCond']
combine['BsmtQuality'] = combine['BsmtQual'] + combine['BsmtCond']
combine['Has2ndFloor'] = combine['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)
combine['TotalSF'] = combine['GrLivArea'] + combine['TotalBsmtSF']
combine['HouseAge'] = combine['YrSold'] - combine['YearRemodAdd']
combine['TotalBath'] = combine['BsmtFullBath'] + combine['FullBath'] + 0.5*(combine['BsmtHalfBath'] + combine['HalfBath'])
combine['TotalPorchSF'] = combine['OpenPorchSF'] + combine['EnclosedPorch'] + combine['3SsnPorch'] + combine['ScreenPorch']

## Nominal Variables Encoding
combine = pd.get_dummies(combine,dummy_na=True,columns=['Alley','GarageType','MiscFeature'])
combine = pd.get_dummies(combine,columns=['MSSubClass','MSZoning','LandContour','LotConfig','Neighborhood',
'Condition1','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
'Foundation','Heating','Electrical','MoSold','SaleType','SaleCondition','YrSold'])

## Boxcox-Transformation and Scaling data
skewed_columns = skewed_transform(combine,'box1p',0.8,False)
combine = pd.DataFrame(data=RobustScaler().fit_transform(combine),index=combine.index,columns=combine.columns)

X_train = combine[:m]
X_test = combine[m:]

## log-transform target variable for linear models
log_y = y.apply(lambda x: np.log(x))

lasso_grid={
    'alpha':[0.001,0.0006,0.00004,0.0005,0.0001],
    'max_iter':[1000,100,40,45,35],
    'tol':[0,0.001,0.01],
    'warm_start':[True,False],
    'selection':['cyclic','random']}

model_1 = Lasso(
    alpha = 0.0005,
    max_iter = 40,
    selection = 'random',
    tol = 0,
    warm_start = True,
    random_state = 420)

## Grid Search (uncomment to look for better parameters)
# best_params = grid_search(model_1,lasso_grid,X_train,log_y,'neg_mean_squared_error')

## Error with 5-fold CV
cv_error_lasso = np.sqrt(-1*cross_val_score(model_1,X_train,log_y,cv=5,scoring='neg_mean_squared_error')).mean()

## Error for Training set (less irrelevant due to overfitting)
model_1.fit(X_train,log_y)
model1_pred = model_1.predict(X_train)
model1_pred = np.exp(model1_pred)
train_error_lasso = np.sqrt(mean_squared_log_error(y,model1_pred))

## Computing feature importances based on permutation importance
feature_importance = feature_importances(model_1,X_train,log_y,False)

X_test['SalePrice'] = np.exp(model_1.predict(X_test))
X_test.index = range(len(X_train)+1,len(combine)+1)
X_test.index.name = 'Id'
X_test['SalePrice'].to_csv('test_results.csv')







