#score achieved --> 0.79425 (without hyperopt)
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
import xgboost as xgb
from sklearn.model_selection import train_test_split

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

df_tr.info()
# Int64Index: 891 entries, 1 to 891
# Data columns (total 11 columns):
# Survived    891 non-null int64
# Pclass      891 non-null int64
# Name        891 non-null object
# Sex         891 non-null object
# Age         714 non-null float64
# SibSp       891 non-null int64
# Parch       891 non-null int64
# Ticket      891 non-null object
# Fare        891 non-null float64
# Cabin       204 non-null object
# Embarked    889 non-null object
# dtypes: float64(2), int64(4), object(5)
# memory usage: 83.5+ KB
df_ts.info()
# Int64Index: 418 entries, 892 to 1309
# Data columns (total 10 columns):
# Pclass      418 non-null int64
# Name        418 non-null object
# Sex         418 non-null object
# Age         332 non-null float64
# SibSp       418 non-null int64
# Parch       418 non-null int64
# Ticket      418 non-null object
# Fare        417 non-null float64
# Cabin       91 non-null object
# Embarked    418 non-null object
# dtypes: float64(2), int64(3), object(5)
# memory usage: 35.9+ KB


#divide variables according to their type
df_tr.columns
categorical_mask = (df_tr.dtypes == object)
cat_var = ['Sex','Embarked'] #categorical variables
dsc_var = ['SibSp','Parch','Pclass'] #discrete variables
str_var = ['Name','Ticket','Cabin'] #string variables
cnt_var = ['Fare','Age'] #continuos variable
trg_var = ['Survived'] #target variable

############ Let's deal with NaN ######################
#missing values (train set)
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
#repeat this substitution also on testset
df_ts['Cabin_Letter'] = df_ts['Cabin'].apply(lambda x: str(x)[0])
df_ts['Cabin_Available'] = np.NaN
df_ts['Cabin_Available'][df_ts['Cabin'] == 'n'] = 0
df_ts['Cabin_Available'][df_ts['Cabin'] != 'n'] = 1
#visualize Cabin vs Survived
sns.countplot(x ='Cabin_Letter', hue='Survived', data = df_tr, palette="Set1")
sns.countplot(x ='Cabin_Available', hue='Survived', data = df_tr, palette="Set1")
#looks like who has a cabin available is more likely to survive
df_tr = df_tr.drop(columns= ['Cabin','Cabin_Letter'], axis =1) #we can use Cabin Available #instead of Cabin and Cabin Letter
df_ts = df_ts.drop(columns= ['Cabin','Cabin_Letter'], axis =1)



#visualize Embarked vs Survived
sns.barplot(x ='Embarked', y='Survived', data = df_tr)
sns.countplot(x ='Embarked', hue='Survived', data = df_tr, palette="Set1")
#visualize Pclass vs Survived
sns.barplot(x ='Pclass', y='Survived', data = df_tr)
sns.countplot(x ='Pclass', hue='Survived', data = df_tr, palette="Set1")

#it looks like that who has embarked in 'S' and is in Pclass = 3 is very likely to die, we can create a variable that join both the information
df_tr['Cls_x_Emb'] = df_tr['Pclass']*df_tr['Embarked'].replace({'S':3,'C':2,'Q':1})
df_ts['Cls_x_Emb'] = df_ts['Pclass']*df_ts['Embarked'].replace({'S':3,'C':2,'Q':1})
sns.countplot(x ='Cls_x_Emb', hue='Survived', data = df_tr, palette="Set1")


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

#let's deal with Name
df_tr['Name']


############## NEW VARIABLES #########################
#maybe create a new variable will be useful
#relatives
df_tr['Relatives'] = df_tr['Parch'] * df_tr['SibSp']
df_ts['Relatives'] = df_ts['Parch'] * df_ts['SibSp']
#isalone
df_tr['Alone'] = df_tr['Relatives'].apply(lambda x: 1 if x == 0 else 0)
df_ts['Alone'] = df_ts['Relatives'].apply(lambda x: 1 if x == 0 else 0)


cat_var = ['Sex','Embarked','AgeBand','FareBand'] #categorical variables
bin_var = ['Alone']#,'Cabin_Available']
dsc_var = ['Pclass','SibSp','Parch','Relatives','Cls_x_Emb'] #update discrete variables
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

model = xgb.XGBClassifier()
model.fit(Xtr, Ytr)

# ########### manually tune hyperparameter - learning API ############
# Train_DM = xgb.DMatrix(data=Xtr.values, label=Ytr, feature_names=Xtr.columns)
# Test_DM = xgb.DMatrix(data=Xts.values, feature_names=Xts.columns)
# params = {'objective':'binary:logistic','max_depth' : 4}
# cv_rslt = xgb.cv(dtrain = Train_DM, params = params, nfold = 4, num_boost_round = 10, metrics = 'error', as_pandas = True)
# accuracy = 1 - cv_rslt['test-error-mean'].iloc[-1]
# model = xgb.train(dtrain=Train_DM, params = params)
# Yhat_ts = model.predict(Test_DM)
# #xgb.plot_tree(model,num_trees=0)
# xgb.plot_importance(model)
# #####################################################################

########## manually tune hyperparameter - Sklearn API ############
X, Xval, Y, Yval = train_test_split(Xtr, Ytr, test_size=0.2, shuffle = True)
eval_set = [(X, Y), (Xval, Yval)]
model.fit(X, Y, eval_metric=['error'], eval_set=eval_set, verbose=False)

# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Valid')
ax.legend()

plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('XGBoost Accuracy')
plt.show()

#fit the model
model.fit(Xtr, Ytr, eval_metric=['error'], eval_set=eval_set, verbose=False)

#predict survived (manual tuning)
Yhat_ts = model.predict(Xts)
####################################################################



#create submission file
path_sb = os.path.join(os.getcwd(),'submission.csv')
df_sb = df_gs.copy()
df_sb['Survived'] = Yhat_ts
threshold = 0.5 #the best threshold seems to be 0.5
df_sb['Survived'] = df_sb['Survived'].apply(lambda x: 1 if x > threshold else 0)
df_sb.to_csv(path_sb, index = False)

# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.30000000000000004, gamma=0,
#               learning_rate=0.13960269512120327, max_delta_step=0, max_depth=6,
#               min_child_weight=1, missing=None, n_estimators=62, n_jobs=1,
#               nthread=None, objective='binary:logistic', random_state=0,
#               reg_alpha=0, reg_lambda=0.10100125461929543, scale_pos_weight=1,
#               seed=None, silent=None, subsample=0.2, verbosity=1)
