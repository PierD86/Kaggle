#score achieved --> 0.79425
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
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

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
cat_var = ['Sex','Embarked'] #categorical variables
dsc_var = ['SibSp','Parch','Pclass'] #discrete variables
str_var = ['Name','Ticket','Cabin'] #string variables
cnt_var = ['Fare','Age'] #continuos variable
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
#let's create some age band and lets use it instead of Age
df_tr['AgeBand'] = pd.cut(df_tr['Age'],bins=[0,2,17,65,99],labels=['Baby','Child','Adult','Elderly'])
df_ts['AgeBand'] = pd.cut(df_ts['Age'],bins=[0,2,17,65,99],labels=['Baby','Child','Adult','Elderly'])
sns.barplot(x ='AgeBand', y='Survived', data = df_tr)
df_tr = df_tr.drop(columns= 'Age', axis =1)
df_ts = df_ts.drop(columns= 'Age', axis =1)

#deal with Cabin
df_tr['Cabin'].unique()
df_tr.info() #now we have NaN only on Cabin column
df_tr['Cabin_Letter'] = df_tr['Cabin'].apply(lambda x: str(x)[0])
#let's check if Cabin column, where present, influence other variable
df_tr['Cabin_Letter'].unique()
df_tr.info()
sns.barplot(x ='Cabin_Letter', y='Survived', data = df_tr) #from the plot it seems that #there is no difference among the type of cabin, the only letter strongly different is 'n'. #Can be interesting to create the variable 'Cabin available'
df_tr['Cabin_Available'] = np.NaN
df_tr['Cabin_Available'][df_tr['Cabin'] == 'n'] = 0
df_tr['Cabin_Available'][df_tr['Cabin'] != 'n'] = 1
sns.barplot(x ='Cabin_Available', y='Survived', data = df_tr)
df_tr = df_tr.drop(columns= ['Cabin','Cabin_Letter'], axis =1) #we can use Cabin Available #instead of Cabin and Cabin Letter
#repeat this substitution also on testset
df_ts['Cabin_Letter'] = df_ts['Cabin'].apply(lambda x: str(x)[0])
df_ts['Cabin_Available'] = np.NaN
df_ts['Cabin_Available'][df_ts['Cabin'] == 'n'] = 0
df_ts['Cabin_Available'][df_ts['Cabin'] != 'n'] = 1
df_ts = df_ts.drop(columns= ['Cabin','Cabin_Letter'], axis =1)

#visualize Embarked vs Survived
sns.barplot(x ='Embarked', y='Survived', data = df_tr)
#visualize Pclass vs Survived
sns.barplot(x ='Pclass', y='Survived', data = df_tr)

#deal with Fare
#let's plot Fare distribution
sns.distplot(df_tr['Fare']) #it looks a skewed distribution
sns.distplot(df_ts['Fare']) #it looks a skewed distribution
#sqrt transformation
df_tr['Fare_sqrt'] = np.sqrt(df_tr['Fare'])
df_ts['Fare_sqrt'] = np.sqrt(df_ts['Fare'])
sns.distplot(df_tr['Fare_sqrt']) #it looks a skewed distribution
sns.distplot(df_ts['Fare_sqrt']) #it looks a skewed distribution

df_tr = df_tr.drop(columns= 'Fare', axis =1)
df_ts = df_ts.drop(columns= 'Fare', axis =1)

#let's create some fare band and lets use it instead of fare
df_tr['FareBand'] = pd.qcut(df_tr['Fare_sqrt'],4,labels=['Low_fare','MediumLow_Fare','MediumHigh_Fare','High_Fare'])
df_ts['FareBand'] = pd.qcut(df_ts['Fare_sqrt'],4,labels=['Low_fare','MediumLow_Fare','MediumHigh_Fare','High_Fare'])
sns.barplot(x ='FareBand', y='Survived', data = df_tr)
df_tr = df_tr.drop(columns= 'Fare_sqrt', axis =1)
df_ts = df_ts.drop(columns= 'Fare_sqrt', axis =1)


#for further investigation:
    #decide how to trate:
        #ticket
        #name
        #Pclass

############## NEW VARIABLES #########################
#maybe create a new variable will be useful
#relatives
df_tr['Relatives'] = df_tr['Parch'] * df_tr['SibSp']
df_ts['Relatives'] = df_ts['Parch'] * df_ts['SibSp']
#isalone
df_tr['Alone'] = df_tr['Relatives'].apply(lambda x: 1 if x == 0 else 0)
df_ts['Alone'] = df_ts['Relatives'].apply(lambda x: 1 if x == 0 else 0)


cat_var = ['Sex','Embarked','AgeBand','FareBand'] #categorical variables
bin_var = ['Alone','Cabin_Available']
dsc_var = ['Pclass','SibSp','Parch','Relatives'] #update discrete variables
str_var = ['Name','Ticket'] #string variables
cnt_var = [] #continuos variable
trg_var = ['Survived'] #target variable


#we will not use ticket and name at the moment
df_tr = df_tr.drop(columns= str_var, axis = 1)
df_ts = df_ts.drop(columns= str_var, axis = 1)

# lets get dummies from categorical variables
df_tr = pd.get_dummies(df_tr, columns = cat_var)
df_ts = pd.get_dummies(df_ts, columns = cat_var)

# normalization of discrete and continue variables (later we will evaluate if use onehotencoding for discrete variables)
mmsx = MinMaxScaler()

df_tr[dsc_var+cnt_var] = mmsx.fit_transform(df_tr[dsc_var+cnt_var])
df_ts[dsc_var+cnt_var] = mmsx.transform(df_ts[dsc_var+cnt_var])

df_tr.columns
df_ts.columns

#fit model
Xtr = df_tr.drop(columns='Survived')
Ytr = df_tr['Survived']
Xts = df_ts.copy()

xgb = XGBClassifier()
xgb.fit(Xtr, Ytr)

#predict survived
Yhat_ts = xgb.predict(Xts)

#create submission file
path_sb = os.path.join(os.getcwd(),'submission.csv')
df_sb = df_gs.copy()
df_sb['Survived'] = Yhat_ts
df_sb.to_csv(path_sb, index = False)






