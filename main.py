# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:47:57 2018

@author: Vigèr Durand AZIMEDEM TSAFACK
"""

#%%
#importation of routine packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pprint
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
#%%

#%%
#import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#%%

#%%
#Correlation map to see how numerical features are correlated between them.
#getting all numerical variables
numeric_train = train._get_numeric_data()
corrmatrix = numeric_train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmatrix, vmax=0.9, square=True)
plt.savefig('numeric_train_corr.png')
plt.show()
#%%

#%%
#removal of outliers
#let's look at the relationship between the 3 features that are correlated the most with SalePrice and SalePrice itself, since they are more 
#likely to influence the SalePrice of a house
plt.scatter(train["OverallQual"], train["SalePrice"])
plt.show()
plt.scatter(train["GrLivArea"], train["SalePrice"])
plt.show()
plt.scatter(train["GarageCars"], train["SalePrice"])
plt.show()
#looking at those correlations, the only criticals outliers we can see here are on the relationship between SalePrice and GrLivArea. Let's remove them
train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))]
#%%

#%%
#Checking multicolinearity
#to do so, let's compute the correlation matrix of all the numerecal features excepting SalePrice
numeric_train.drop(['SalePrice', 'Id'], axis=1, inplace=True)
corrmatrix2 = numeric_train.corr()
corrmatrix2_numpy = corrmatrix2.as_matrix()
#now let's find the couples of highly correlated features
#this loop is used to keep only couple of variables with a heigh correlation (greater than 0.7 and less -0.7)
corlist = []
for c in corrmatrix2.columns.tolist():
    for r in  corrmatrix2.index.tolist():
        i = corrmatrix2.index.get_loc(r)
        j = corrmatrix2.columns.get_loc(c)
        if (corrmatrix2_numpy[i, j] >= 0.7 or corrmatrix2_numpy[i, j] <= -0.7) and (r != c):
            corlist.append([(r, c), corrmatrix2_numpy[i, j]])
pprint.pprint(corlist)

#we have 4 couples, so 4 possible canditates to elimination
#we decided to remove '1stFlrSF', 'GarageArea' and 'TotRmsAbvGrd' because from the correlation matrix, they are less correlated to the target variable 
#SalePrice than the other members of their couples.
#We are however keeping 'GarageYrBlt' and 'YearBuilt' all together because they have weak correlation with Saleprice. Thus, their effect is not so much considerable    

#merge the train and the test set excluding ID and SalePrice for quick feature engineering
full = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                  test.loc[:,'MSSubClass':'SaleCondition']))

full.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis = 1, inplace = True)
#%%

#%%
#Checking the skewness
#first of all, let's check the skewness of the Response variable(SalePrice)
print(skew(train['SalePrice'].dropna()))
#1.87 is pretty heigh for a skew wich ideally show between -1 and 1
plt.hist(train['SalePrice'])
plt.show()
#loking at the histogram, we can state that the LogTransformation is more indacated to reduce the skewness of SalePrice, since the log increases quickly and then increases constantly.
#and that is the behaviour we need to make this curve symetric.
train['SalePrice'] = np.log1p(train['SalePrice'])
print(skew(train['SalePrice'].dropna()))
#0.12 is the new skew wich is quite good
plt.hist(train['SalePrice'])
plt.show()

#now let's compute the skewness of the other numerical variables
numeric_vars = full._get_numeric_data().columns
skewed_vars = train[numeric_vars].apply(lambda var: skew(var.dropna()))
skewed_vars = skewed_vars[skewed_vars > 0.65]
skewed_vars = skewed_vars.index

#let's correct the skewness using the box-cox transformation
full[skewed_vars] = boxcox1p(full[skewed_vars], 0.14)
#%%

#%%
#dealing with the non numerical variabless
#we are going to convert them into dummie variables using pandas.get_dummies
full = pd.get_dummies(full)
#%%

#%%
#dealing with NA's
#replacing NA's by the mean of the other elements in the column
full = full.fillna(full.mean())
#%%

#%%
#models selection
X_train = full[:train.shape[0]]
X_test = full[train.shape[0]:]
y = train['SalePrice']

model = Lasso(alpha=0.0004)
model.fit(X_train, y)


#%%

#%%
### prediction
model.fit(X_train, y)

preds = np.expm1(model.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("full_lasso.csv", index = False)
#%%