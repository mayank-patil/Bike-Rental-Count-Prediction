#importing library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Reading data file
df = pd.read_csv("day.csv")

df.head()

# changing datatype of variables
df['season']= df['season'].astype('object')
df['mnth']=df['mnth'].astype('object')
df['weekday']=df['weekday'].astype('object')
df['weathersit']=df['weathersit'].astype('object')
df['workingday']=df['workingday'].astype('object')
df['yr']=df['yr'].astype('object')
df['holiday']=df['holiday'].astype('object')

df.info()

### Finding and Removing Missing Values

miss = pd.DataFrame(df.isnull().sum())
miss = miss.rename(columns={0:"miss_count"})
miss["miss_%"] = (miss.miss_count/len(df.instant))*100
miss

#* The data doesn't seems to have any missing Values.

### Detecting and removing outlier Removing Outliers

# using box plot
df.boxplot(figsize=(15,5))

df1 = df.copy()
cnames=list(df1.columns)
for i in cnames:
    if isinstance(df1[i].iloc[1] , float) or isinstance(df1[i].iloc[1] , int) or isinstance(df1[i].iloc[1] , np.int64) :
        print(i)
        q75, q25 = np.percentile(df1.loc[:,i], [75 ,25])
        iqr = q75 - q25

        min = q25 - (iqr*1.5)
        max = q75 + (iqr*1.5)
        print("min: "+str(min))
        print("max: "+str(max))

        df1 = df1.drop(df1[df1.loc[:,i] < min].index)
        df1 = df1.drop(df1[df1.loc[:,i] > max].index)
        #df1[df1.loc[:,i] < min] = np.nan
        #df1[df1.loc[:,i] > max] = np.nan
        print("length: "+str(len(df1.instant)))
    elif isinstance(df1[i].iloc[1] , str):
        continue
df1.reset_index(drop=True,inplace=True)
df1.shape
# creating the list of our purchase type
#label = list(data3.PURCHASE_TYPE )

df1.cnt

# saving the target variable
count = df1.cnt
df1 = df1.iloc[:,2:15]

# using box plot
df1.boxplot(figsize=(15,5))

#* Only 655 data points left After Removal of all possible outliers.

### Checking the Distribution of data

#calculating no. of bins
aa=round(np.sqrt(len(df1.yr)))
#Creating a histogram
df1.hist(figsize=(20,10),bins=int(aa))

#### Scaling Variables casual and registered with minmax scaler

sccase = df1.loc[:,['casual']]
scregis = df1.loc[:,['registered']]

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
sccase =list((scaler.fit_transform(sccase)))
scregis = list((scaler.fit_transform(scregis)))
df1.loc[:,['casual']] = sccase
df1.loc[:,['registered']] = scregis
df1



## Model1
### Linear Regression
#### Assumption
#1. There must be a linear relationship between the dependent variable and the independent variables.  Scatterplots can show whether there is a linear or curvilinear relationship.
#2. Multivariate Normality–Multiple regression assumes that the residuals are normally distributed.
#3. No Multicollinearity—Multiple regression assumes that the independent variables are not highly correlated with each other.  This assumption is tested using Variance Inflation Factor (VIF) values.
#4. Homoscedasticity–This assumption states that the variance of error terms are similar across the values of the independent variables.  A plot of standardized residuals versus predicted values can show whether points are equally distributed across all values of the independent variables.

df2 =df1.copy()
# changing datatype of variables
df2['season']= df2['season'].astype('int64')
df2['mnth']=df2['mnth'].astype('int64')
df2['weekday']=df2['weekday'].astype('int64')
df2['weathersit']=df2['weathersit'].astype('int64')
df2['workingday']=df2['workingday'].astype('int64')
df2['yr']=df2['yr'].astype('int64')
df2['holiday']=df2['holiday'].astype('int64')
df2["count"] = count
df2.info()

df2.iloc[:,0:13]

### 1. Identifying whether there is a linear relationship or not between the Target and the dependent variables.

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
train, test = train_test_split(df2, test_size=0.2)
model1 = sm.OLS(train.iloc[:,13], train.iloc[:,0:13]).fit()
model1.summary()

def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
testp = model1.predict(test.iloc[:,0:13])

MAPE(test.iloc[:,13],testp)

r2_score(test.iloc[:,13],testp),math.sqrt(mean_squared_error(test.iloc[:,13],testp))

#%matplotlib inline
#%config InlineBackend.figure_format ='retina'
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)

model1 = sm.OLS(df2.iloc[:,13], df2.iloc[:,0:13]).fit()
model1.summary()

def linearity_test(model, y):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model.
    It plots observed vs. predicted values and residuals vs. predicted values.

    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    '''
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1,2)

    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')

linearity_test(model1, count)
#count

##### from the above Graph we can see that there is a linear relationship between variables.
#* they’re pretty symmetrically distributed, tending to cluster towards the middle of the plot

### 2. Identifying Multivariate Normality.
#* plotting Residuals with theoritical quantiles.

import scipy as sp
resids = model1.resid
fig, ax = plt.subplots(figsize=(6,5))
_, (__, ___, r) = sp.stats.probplot(resids, plot=ax, fit=True)

#* The good fit indicates that normality is a reasonable approximation.

### 3. Identifying and Removing Multicolinearity from the data

##Correlation analysis
#Correlation plot
df_corr = df2

#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(15, 7))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# correlation matrix
corr

##### we can clearly see that some independent variables are highly correlated to each other
#* lets calculate the vif and remove those variables in order to lessen the complexity of the model.

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
fvif = df2.iloc[:,0:13]
X = add_constant(fvif)

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=fvif.columns).T

#### Since atemp has a High VIF score we will remove it from our data set.

cc =['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'casual', 'registered', 'count']
reg = df2.loc[:,cc]

train, test = train_test_split(reg, test_size=0.2)
model2 = sm.OLS(train.iloc[:,12], train.iloc[:,0:12]).fit()
model2.summary()

##### Since p-value of variables ["season","weathersit", "yr","mnth"]
#* are statistically insignificant for our model
#* because there p value is greater than alpha =0.05.
#* so we will remove them from our data frame

rm = ["season","weathersit", "yr","mnth"]
reg2=reg.drop(columns=rm)

testp = model2.predict(test.iloc[:,0:12])


MAPE(test.iloc[:,12],testp)

r2_score(test.iloc[:,12],testp),math.sqrt(mean_squared_error(test.iloc[:,12],testp))

### 4. Detecting Hetroscedaticity
#* Using goldfeld quandt test

gq_test = pd.DataFrame(sms.het_goldfeldquandt(model2.resid, model2.model.exog)[:-1],
                           columns=['value'],
                           index=['F statistic', 'p-value'])
gq_test

##### Since p-value is greater than the alpha = 0.05, we can asume that data is homoscedatic.

### Building a final model after satisfying all the assumption.

train, test = train_test_split(reg2, test_size=0.2)
model3 = sm.OLS(train.iloc[:,8], train.iloc[:,0:8]).fit()
model3.summary()

testp = model3.predict(test.iloc[:,0:8])

MAPE(test.iloc[:,8],testp),r2_score(test.iloc[:,8],testp),math.sqrt(mean_squared_error(test.iloc[:,8],testp))

#### Applyting K-Fold cross validation with 10 Folds to finalize the model.

from sklearn.model_selection import KFold # import KFold
reg2.reset_index(drop=True,inplace=True)
y = reg2["count"].values
X = reg2.iloc[:,0:8].values
kf = KFold(n_splits=10)
kf.get_n_splits(X)

mape = []
rmse = []
r_2 = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    models = sm.OLS(y_train, X_train).fit()
    testp = models.predict(X_test)
    mape.append(MAPE(y_test,testp))
    rmse.append(math.sqrt(mean_squared_error(y_test,testp)))
    r_2.append(r2_score(y_test,testp))

np.mean(mape),np.mean(r_2),np.mean(rmse)

#### The Final Model satisfy all assumptions of linear regression


print("1. explains around 99% variance \n2. RMSE of "+str(np.mean(rmse))+" and \n3. MAPE of "+str(np.mean(mape)))

## Model2
### Decision Tree Regressor

df3 = reg2.copy()
df3["count"] = count
df3

#### Setting up cross validation with 5 fold approx 20% data

from sklearn.model_selection import KFold # import KFold
df3.reset_index(drop=True,inplace=True)
y = df3["count"].values
X = df3.iloc[:,0:8].values
kf = KFold(n_splits=5)
kf.get_n_splits(X)

#### Training and testing with K-fold

from sklearn import tree
mape = []
rmse = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    models = tree.DecisionTreeRegressor(criterion="mse").fit(X_train,y_train)
    testp = models.predict(X_test)
    mape.append(MAPE(y_test,testp))
    rmse.append(math.sqrt(mean_squared_error(y_test,testp)))


np.mean(mape),np.mean(rmse)

#### The Final decision tree model

print("1. RMSE of "+str(np.mean(rmse))+" and \n3. MAPE of "+str(np.mean(mape)))

#* Which is quite higher than that of the linear regression model.

## Model 3
### Using Ensembled method Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

y = df3["count"].values
X = df3.iloc[:,0:8].values

obb = []
for i in tqdm(range(100,600,10)):
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
    clf=RandomForestRegressor( n_estimators=i,oob_score=True,max_features="sqrt").fit(x_train,y_train)
    pred = clf.predict(x_test)
    obb.append(clf.oob_score_)


np.max(obb)

#### Finding Optimal The No. Of Estimator with Maximum Obb Score

ss = list(range(100,600,10))

# Index of the greatest obb
mobb = obb.index(np.max(obb))
nestim = ss[mobb]

#### Setting up cross validation with 5 fold approx 20% data
#with optimal no. of estimator

from sklearn.model_selection import KFold # import KFold
y = df3["count"].values
X = df3.iloc[:,0:8].values
kf = KFold(n_splits=5)
kf.get_n_splits(X)

mape = []
rmse = []

for train_index, test_index in tqdm(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    models = RandomForestRegressor( n_estimators=nestim,oob_score=True,max_features="sqrt").fit(X_train,y_train)
    testp = models.predict(X_test)
    mape.append(MAPE(y_test,testp))
    rmse.append(math.sqrt(mean_squared_error(y_test,testp)))

np.mean(mape),np.mean(rmse)

#### The Final Random Forest Model

print("1. RMSE of "+str(np.mean(rmse))+" and \n3. MAPE of "+str(np.mean(mape)))

## Model 4
### Using Distance based method KNN

from sklearn import neighbors
y = df3["count"].values
x = df3.iloc[:,0:8].values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = math.sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

k = list(range(1,21))

plt.plot(k,rmse_val,marker="o")

# Index of the smallest RMSE
mobb = rmse_val.index(np.min(rmse_val))
k_n = k[mobb]
k_n

mape = []
rmse = []
y = df3["count"].values
X = df3.iloc[:,0:8].values
for train_index, test_index in tqdm(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    models = neighbors.KNeighborsRegressor(n_neighbors = k_n).fit(X_train,y_train)
    testp = models.predict(X_test)
    mape.append(MAPE(y_test,testp))
    rmse.append(math.sqrt(mean_squared_error(y_test,testp)))

np.mean(mape),np.mean(rmse)

#### The Final KNN Model

print("1. RMSE of "+str(np.mean(rmse))+" and \n3. MAPE of "+str(np.mean(mape)))
