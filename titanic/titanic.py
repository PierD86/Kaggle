import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
np.random.seed(1926)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict
import sklearn.metrics

#titanic input files' path
!dir
os.chdir(r'C:\Users\ER180124\Code\Kaggle\titanic')
path_tr = os.path.join(os.getcwd(),'train.csv')
path_ts = os.path.join(os.getcwd(),'test.csv')
path_gs = os.path.join(os.getcwd(),'gender_submission.csv')
#csv inport
df_tr = pd.read_csv(path_tr)
df_ts = pd.read_csv(path_ts)
df_gs = pd.read_csv(path_gs)


df_tr.info()
df_ts.info()

#general overview of train dataset
sns.pairplot(df_tr)

#set PassengerId as index
df_tr.set_index('PassengerId', inplace = True)
df_ts.set_index('PassengerId', inplace = True)


#divide variables according to their type
df_tr.columns
cat_var = ['Pclass','Sex','Embarked'] #categorical variables
dsc_var = ['Age','SibSp','Parch'] #discrete variables
str_var = ['Name','Ticket','Cabin'] #string variables
cnt_var = ['Fare'] #continuos variable
trg_var = ['Survived'] #target variable

############ Let's deal with NaN ######################
#missing values (train set)
df_tr['Pclass'].isnull().sum() #no missing values
df_tr['Sex'].isnull().sum() #no missing values
df_tr['Embarked'].isnull().sum() #2oo891 missing values
df_tr['Age'].isnull().sum() #177oo891 missing values
df_tr['SibSp'].isnull().sum() #no missing values
df_tr['Parch'].isnull().sum() #no missing values
df_tr['Name'].isnull().sum() #no missing values
df_tr['Ticket'].isnull().sum() #no missing values
df_tr['Cabin'].isnull().sum() #687oo891 missing values
df_tr['Fare'].isnull().sum() #no missing values
df_tr['Survived'].isnull().sum() #no missing values
#missing values (test set)
df_ts['Pclass'].isnull().sum() #no missing values
df_ts['Sex'].isnull().sum() #no missing values
df_ts['Embarked'].isnull().sum() #no missing values
df_ts['Age'].isnull().sum() #86oo418 missing values
df_ts['SibSp'].isnull().sum() #no missing values
df_ts['Parch'].isnull().sum() #no missing values
df_ts['Name'].isnull().sum() #no missing values
df_ts['Ticket'].isnull().sum() #no missing values
df_ts['Cabin'].isnull().sum() #327oo418 missing values
df_ts['Fare'].isnull().sum() #1 missing values


#where missing values are a low percentage, we can replace NaN with most frequent value
df_tr['Embarked'].fillna(df_tr['Embarked'].mode()[0], inplace = True)
df_ts['Fare'].fillna(df_ts['Fare'].mode()[0], inplace = True)

#lets check if there is some relation among the discrete variables
sns.pairplot(df_tr[dsc_var])
sns.scatterplot(x = df_tr['Fare'], y = df_tr['Age'].dropna(), hue = df_tr['SibSp'])
sns.scatterplot(x = df_tr['Fare'], y = df_tr['Age'].dropna(), hue = df_tr['Parch'])
sns.scatterplot(x = df_tr['Fare'], y = df_tr['Age'].dropna(), hue = df_tr['Sex'])
#checking the heatmap Pclass, SibSp, and Parch seem to be correlated
sns.heatmap(df_tr.corr(),annot = True, fmt = '.2f')

# Try to regress Age
rgr = DecisionTreeRegressor()
Yage = pd.concat([df_tr['Age'], df_ts['Age']], axis = 0)
Xage = pd.concat([df_tr[['Pclass','SibSp','Parch']], df_ts[['Pclass','SibSp','Parch']]], axis = 0)[~Yage.isnull()]
Yage = Yage.dropna()
#check cross val score
rmse_cv = -cross_val_score(rgr, Xage, Yage, cv=5, scoring ='neg_root_mean_squared_error')
rmse_mu = np.mean(-cross_val_score(rgr, Xage, Yage, cv=5, scoring ='neg_root_mean_squared_error')) #rmse is about 11 years, for our purpose is accettable since we will bin age
print(rmse_cv)
print(rmse_mu)
#fit the regressor
rgr.fit(Xage, Yage)
#make predictions
Xnan_tr = df_tr[['Pclass','SibSp','Parch']][df_tr['Age'].isnull()]
Yhat_nan_tr = rgr.predict(Xnan_tr)
Xnan_ts = df_ts[['Pclass','SibSp','Parch']][df_ts['Age'].isnull()]
Yhat_nan_ts = rgr.predict(Xnan_ts)
df_tr['Age'][df_tr['Age'].isnull()]=Yhat_nan_tr
df_ts['Age'][df_ts['Age'].isnull()]=Yhat_nan_ts
#let's plot Age distribution
sns.distplot(df_tr['Age']) #it looks a normal distribution


#deal with Cabin
df_tr['Cabin'].unique()
df_tr.info() #now we have NaN only on Cabin column
df_tr['Cabin_Letter'] = df_tr_cabin['Cabin'].apply(lambda x: x[0])
df_tr = df_tr.fillna('Z') #let's check if Cabin column, where present, influence other variable
df_tr['Cabin_Letter'].unique()
df_tr.info()
df_tr = df_tr.drop(columns='Cabin', axis =1) #we can use Cabin Letter instead of Cabin




############## NEW VARIABLES #########################
#maybe create a new variable will be useful
#relatives
df_tr['Relatives'] = df_tr['Parch'] + df_tr['SibSp']
df_ts['Relatives'] = df_ts['Parch'] + df_ts['SibSp']
#isalone


dsc_var = ['Age','SibSp','Parch','Relatives','IsAlone'] #update discrete variables









