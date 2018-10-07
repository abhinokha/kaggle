## importing libraries
import numpy as np
import pandas as pd

import sys

from sklearn.preprocessing import LabelEncoder
import json
from pandas.io.json import json_normalize
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing, cross_validation

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

PATH=""
cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']

def read_parse_dataframe(file_name):
    #full path for the data file
    path = PATH + file_name
    #read the data file, convert the columns in the list of columns to parse using json loader,
    #convert the `fullVisitorId` field as a string
    data_df = pd.read_csv(path,
                          nrows=100000,
        converters={column: json.loads for column in cols_to_parse}, 
        dtype={'fullVisitorId': 'str'})
    #parse the json-type columns
    for col in cols_to_parse:
        #each column became a dataset, with the columns the fields of the Json type object
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}.{sub_col}" for sub_col in json_col_df.columns]
        #we drop the object column processed and we add the columns created from the json fields
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)

    return data_df
    
def process_date_time(data_df):

    data_df['date'] = data_df['date'].astype(str)
    data_df["date"] = data_df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    data_df["date"] = pd.to_datetime(data_df["date"])   
    data_df["year"] = data_df['date'].dt.year
    data_df["month"] = data_df['date'].dt.month
    data_df["day"] = data_df['date'].dt.day
    data_df["weekday"] = data_df['date'].dt.weekday
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    data_df['month.unique.user.count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
    data_df['day.unique.user.count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
    data_df['weekday.unique.user.count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')

    
    return data_df

def process_format(data_df):

    for col in ['visitNumber', 'totals.hits', 'totals.pageviews']:
        data_df[col] = data_df[col].astype(float)
    data_df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    data_df['trafficSource.isTrueDirect'].fillna(False, inplace=True)

    return data_df
    
def process_device(data_df):

    data_df['browser.category'] = data_df['device.browser'] + '.' + data_df['device.deviceCategory']
    data_df['browser.os'] = data_df['device.browser'] + '.' + data_df['device.operatingSystem']

    return data_df

def process_totals(data_df):

    data_df['visitNumber'] = (data_df['visitNumber'])
    data_df['totals.hits'] = (data_df['totals.hits'])
    data_df['totals.pageviews'] = (data_df['totals.pageviews'].fillna(0))
    data_df['mean.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('mean')
    data_df['sum.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('sum')
    data_df['max.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('max')
    data_df['min.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('min')
    data_df['var.hits.per.day'] = data_df.groupby(['day'])['totals.hits'].transform('var')
    data_df['mean.pageviews.per.day'] = data_df.groupby(['day'])['totals.pageviews'].transform('mean')
    data_df['sum.pageviews.per.day'] = data_df.groupby(['day'])['totals.pageviews'].transform('sum')
    data_df['max.pageviews.per.day'] = data_df.groupby(['day'])['totals.pageviews'].transform('max')
    data_df['min.pageviews.per.day'] = data_df.groupby(['day'])['totals.pageviews'].transform('min')    

    return data_df

def process_geo_network(data_df):

    data_df['sum.pageviews.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    data_df['count.pageviews.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    data_df['mean.pageviews.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
    data_df['sum.hits.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    data_df['count.hits.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    data_df['mean.hits.per.network.domain'] = data_df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

    return data_df

def process_traffic_source(data_df):

    data_df['source.country'] = data_df['trafficSource.source'] + '.' + data_df['geoNetwork.country']
    data_df['campaign.medium'] = data_df['trafficSource.campaign'] + '.' + data_df['trafficSource.medium']
    data_df['medium.hits.mean'] = data_df.groupby(['trafficSource.medium'])['totals.hits'].transform('mean')
    data_df['medium.hits.max'] = data_df.groupby(['trafficSource.medium'])['totals.hits'].transform('max')
    data_df['medium.hits.min'] = data_df.groupby(['trafficSource.medium'])['totals.hits'].transform('min')
    data_df['medium.hits.sum'] = data_df.groupby(['trafficSource.medium'])['totals.hits'].transform('sum')

    return data_df

train = read_parse_dataframe('train.csv')
test = read_parse_dataframe('test.csv')

trn_len = train.shape[0]
train = process_date_time(train)
test = process_date_time(test)

cols_to_drop = [col for col in train.columns if train[col].nunique(dropna=False) == 1]
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop([col for col in cols_to_drop if col in test.columns], axis=1, inplace=True)

train.drop(['trafficSource.campaignCode'], axis=1, inplace=True)

train['totals.transactionRevenue'] = train['totals.transactionRevenue'].astype(float)
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'])

## Features engineering
train = process_format(train)
train = process_device(train)
train = process_totals(train)
train = process_geo_network(train)
train = process_traffic_source(train)

test = process_format(test)
test = process_device(test)
test = process_totals(test)
test = process_geo_network(test)
test = process_traffic_source(test)

print("process categorical columns ...")
num_cols = ['month.unique.user.count', 'day.unique.user.count', 'weekday.unique.user.count',
            'visitNumber', 'totals.hits', 'totals.pageviews', 
            'mean.hits.per.day', 'sum.hits.per.day', 'min.hits.per.day', 'max.hits.per.day', 'var.hits.per.day',
            'mean.pageviews.per.day', 'sum.pageviews.per.day', 'min.pageviews.per.day', 'max.pageviews.per.day',
            'sum.pageviews.per.network.domain', 'count.pageviews.per.network.domain', 'mean.pageviews.per.network.domain',
            'sum.hits.per.network.domain', 'count.hits.per.network.domain', 'mean.hits.per.network.domain',
            'medium.hits.mean','medium.hits.min','medium.hits.max','medium.hits.sum']
            
not_used_cols = ["visitNumber", "date", "fullVisitorId", "sessionId", 
        "visitId", "visitStartTime", 'totals.transactionRevenue', 'trafficSource.referralPath']
cat_cols = [col for col in train.columns if col not in num_cols and col not in not_used_cols]


merged_df = pd.concat([train, test])
print('Cat columns : ', len(cat_cols))
ohe_cols = []
for i in cat_cols:
    if len(set(merged_df[i].values)) < 100:
        ohe_cols.append(i)

print('ohe_cols : ', ohe_cols)
print(len(ohe_cols))
merged_df = pd.get_dummies(merged_df, columns = ohe_cols)
train = merged_df[:trn_len]
test = merged_df[trn_len:]
del merged_df


for col in cat_cols:
    if col in ohe_cols:
        continue
    #print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

print('FINAL train shape : ', train.shape, ' test shape : ', test.shape)


train = train.sort_values('date')
X = train.drop(not_used_cols, axis=1)
y = train['totals.transactionRevenue']
X_test = test.drop([col for col in not_used_cols if col in test.columns], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=69)

xgb_model = xgb.XGBRegressor()
parameters = {'nthread':[4], 
              'objective':['reg:linear'],
              'learning_rate': [0.01],
              'max_depth': [5,8,10],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [500],
              'missing':[-999],
              'seed': [1337]}

xgbreg_gs = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(y_train, n_folds=2, shuffle=True), 
                   scoring='neg_mean_squared_error',
                   verbose=2, refit=True)
xgbreg_gs.fit(X_train, y_train)

best_parameters, score, _ = max(xgbreg_gs.grid_scores_, key=lambda x: x[1])
print(best_parameters)

y_val_pred = xgbreg_gs.predict(X_val)

print("Train RSME" % metrics.mean_squared_error(y_train, xgbreg_gs.predict(X_train)))
print("Val RSME" % metrics.mean_squared_error(y_val, y_val_pred))