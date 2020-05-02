import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as st
import datetime

#import all input csv
#!dir
path_features = os.path.join(os.getcwd(),'features.csv')
path_stores = os.path.join(os.getcwd(),'stores.csv')
path_train = os.path.join(os.getcwd(),'train.csv')
path_test = os.path.join(os.getcwd(),'test.csv')
path_samplesub = os.path.join(os.getcwd(),'sampleSubmission.csv')

df_ft = pd.read_csv(path_features)
df_st = pd.read_csv(path_stores)
df_tr = pd.read_csv(path_train)
df_ts = pd.read_csv(path_test)
df_ss = pd.read_csv(path_samplesub)

#merge all information in a single dataframe (train)
df = pd.merge(df_tr, df_st, how = 'left', on = 'Store')
df = pd.merge(df, df_ft, how = 'left', on=['Date', 'Store','IsHoliday'])

#merge all information in a single dataframe (test)
df_ts = pd.merge(df_ts, df_st, how = 'left', on = 'Store')
df_ts = pd.merge(df_ts, df_ft, how = 'left', on=['Date', 'Store','IsHoliday'])


#checking NaN values
df.isnull().sum()
df.fillna(0,inplace=True)
df_ts.fillna(0,inplace=True)

#manipulate dates
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfYear'] = df['Date'].dt.dayofyear
df_ts['Date'] = pd.to_datetime(df_ts['Date'])
df_ts['DayOfYear'] = df_ts['Date'].dt.dayofyear

#get_dummies
df = pd.get_dummies(df, columns = ['Type'])
df_ts = pd.get_dummies(df_ts, columns = ['Type'])

#define variables type
cont_var = ['Size',
            'Temperature',
            'Fuel_Price',
            'MarkDown1',
            'MarkDown2',
            'MarkDown3',
            'MarkDown4',
            'MarkDown5',
            'CPI',
            'Unemployment',
            'DayOfYear']


#EDA
############### CONTINUOS VARS #########################
#this is how continuos variables look like
fig, ax = plt.subplots(4,3, figsize=(16,16))
sns.distplot(df['Weekly_Sales'], ax = ax[0,0], fit=st.norm)
sns.distplot(df['Size'], ax = ax[0,1], fit=st.norm)
sns.distplot(df['Temperature'], ax = ax[0,2], fit=st.norm)
sns.distplot(df['Fuel_Price'], ax = ax[1,0], fit=st.norm)
sns.distplot(df['MarkDown1'], ax = ax[1,1], fit=st.norm)
sns.distplot(df['MarkDown2'], ax = ax[1,2], fit=st.norm)
sns.distplot(df['MarkDown3'], ax = ax[2,0], fit=st.norm)
sns.distplot(df['MarkDown4'], ax = ax[2,1], fit=st.norm)
sns.distplot(df['MarkDown5'], ax = ax[2,2], fit=st.norm)
sns.distplot(df['CPI'], ax = ax[3,0], fit=st.norm)
sns.distplot(df['Unemployment'], ax = ax[3,1], fit=st.norm)
sns.distplot(df['DayOfYear'], ax = ax[3,2], fit=st.norm)


#unskewing data:
#let's check if normalization helps to unskew data
from sklearn.preprocessing import MinMaxScaler
mmsx = MinMaxScaler()
mmsy = MinMaxScaler()
df_s = df.copy()
df_ts_s = df_ts.copy()

df_s[cont_var] = mmsx.fit_transform(df[cont_var])
df_s[['Weekly_Sales']] = mmsx.fit_transform(df[['Weekly_Sales']])
df_ts_s[cont_var] = mmsx.transform(df_ts[cont_var])

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def original(x):
  return x

def power_2(x):
  return x**2  

def yeojohnson(x):
  try:
    for var in x.columns:
      x[var], _  = st.yeojohnson(x[var])
  except:
    x,_ = st.yeojohnson(x)
  return x 

dz = {}
dz['sigmoid'] = [sigmoid, 'r']
dz['tan'] = [np.tan, 'b']
dz['tanh'] = [np.tanh, 'm']
dz['sin'] = [np.sin, 'y']
dz['log'] = [np.log1p, 'k']
dz['sqrt'] = [np.sqrt, 'g']
dz['original'] = [original, 'c']
dz['power2'] = [power_2, 'lime']
dz['yeojohnson'] = [yeojohnson, 'violet']


def unskewner(dfx, dz_func, func_name, vars):
    fig, ax = plt.subplots(4,3, figsize=(16,16))
    sns.distplot(dz_func[func_name][0](dfx['Weekly_Sales']), ax = ax[0,0], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['Size']), ax = ax[0,1], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['Temperature']), ax = ax[0,2], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['Fuel_Price']), ax = ax[1,0], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['MarkDown1']), ax = ax[1,1], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['MarkDown2']), ax = ax[1,2], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['MarkDown3']), ax = ax[2,0], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['MarkDown4']), ax = ax[2,1], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['MarkDown5']), ax = ax[2,2], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['CPI']), ax = ax[3,0], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['Unemployment']), ax = ax[3,1], fit=st.norm, color = dz_func[func_name][1])
    sns.distplot(dz_func[func_name][0](dfx['DayOfYear']), ax = ax[3,2], fit=st.norm, color = dz_func[func_name][1])
    plt.suptitle(f'{func_name}')
    plt.savefig(f'dist_{func_name}.jpg')
    dz_mu, dz_sgm, dz_skw, dz_krt = {}, {}, {}, {}
    for i in vars:
        dz_mu[i] = np.mean(dz_func[func_name][0](dfx[i]))
        dz_sgm[i] = np.std(dz_func[func_name][0](dfx[i]))
        dz_skw[i] = st.skew(dz_func[func_name][0](dfx[i]))
        dz_krt[i] = st.kurtosis(dz_func[func_name][0](dfx[i]))
    return dz_mu, dz_sgm, dz_skw, dz_krt

#create distplot for each transformation in dz
rslt_mu, rslt_sgm, rslt_skw, rslt_krt = {}, {}, {}, {}
for k in dz:
    rslt_mu[k],rslt_sgm[k],rslt_skw[k],rslt_krt[k] = unskewner(df_s,dz,k,cont_var)

def heatmapper(dfx, dz_func, func_name, vars):
    plt.figure(figsize=(10,10)) 
    sns.heatmap(dz_func[func_name][0](dfx[vars]).corr(), annot = True, fmt='.2f', cmap ='Blues')
    plt.suptitle(func_name)
    plt.savefig(f'heatmap_{func_name}.jpg')

#create heatmaps for each transformation in dz
for k in dz:
  heatmapper(df_s,dz,k,cont_var)


#from all plots and analysing skewness and kurtosis of all transformations
#seems very usefull to use the 'yeojohnson' methodology for the 'norm_var'
#heatmaps shows that aweak  linear correlation with Weekly_Sales is present 
#only for 'Size' and 'MarkDown5'

norm_var = ['Temperature',
            'Fuel_Price',
            'MarkDown1',
            'MarkDown2',
            'MarkDown3',
            'MarkDown4',
            'MarkDown5',
            'Unemployment']
df_yeo = df_s.copy()
df_ts_yeo = df_ts.copy()
yeo_lambda = {}
yeo_lambda_ts = {}
for var in norm_var:
  df_yeo[var],yeo_lambda[var]  = st.yeojohnson(df_s[var])
  df_ts_yeo[var],yeo_lambda_ts[var]  = st.yeojohnson(df_ts_s[var])

#df_yeo['Weekly_Sales'],yeo_lambda['Weekly_Sales']  = st.yeojohnson(df_s['Weekly_Sales'])


#create a model - XGBoost
from xgboost import XGBRegressor
X_tr = df.drop(columns=['Date','Weekly_Sales'])
Y_tr = df[['Weekly_Sales']]
model = XGBRegressor()
model.fit(X_tr,Y_tr)

X_ts = df_ts.drop(columns=['Date'])
Yhat_ts = model.predict(X_ts)

df_ts['Weekly_Sales'] = Yhat_ts


#create the Id column as the SampleSubmission
df_ts['Date'] = df_ts['Date'].dt.strftime('%Y-%m-%d')
df_ts['Id'] = df_ts.apply(lambda x: str(x['Store']) + '_' +
                                      str(x['Dept']) + '_' +
                                      str(x['Date']), axis=1)

#create submission file
df_sb = df_ts[['Id','Weekly_Sales']]
path_sb = os.path.join(os.getcwd(),'submission.csv')
df_sb.to_csv(path_sb, sep=',', index = False)