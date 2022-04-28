from pyexpat import model
import requests
import time
import pandas as pd, numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import pickle
import datetime as dt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold

import lightgbm as lgb
import optuna
import optuna.integration.lightgbm as lgb_tune
import optuna.integration.lightgbm as optuna_lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.model_selection import KFold

# LINE nitification
def line_notify(message):
    line_notify_token = 'RjPVIZw064cOaSknExnBVidycTcfZJiHdCOvB5AMS3k'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token} 
    requests.post(line_notify_api, data=payload, headers=headers)

# load data
X_train = pd.read_pickle('./data/X_train.pickle')
X_val = pd.read_pickle('./data/X_val.pickle')
y_train = pd.read_pickle('./data/y_train.pickle')
y_val = pd.read_pickle('./data/y_val.pickle')

params = {
    'objective': 'regression',
    'metric': 'mae'
}

split = 5
tscv = TimeSeriesSplit(n_splits=split)

params_list = []
models = []
score_list = []

for train, val in tscv.split(X_train):
    trn_x, val_x = X_train.iloc[train], X_train.iloc[val]
    trn_y, val_y = y_train.iloc[train], y_train.iloc[val]

    trn_ds = optuna_lgb.Dataset(trn_x, trn_y)
    val_ds = optuna_lgb.Dataset(val_x, val_y)

    best = optuna_lgb.train(
        params,
        trn_ds,
        valid_sets=val_ds,
        num_boost_round=100,
        early_stopping_rounds=50,
        show_progress_bar=False
        )

    tuned_model = lgb.train(
        best.params,
        trn_ds,
        valid_sets=val_ds,
        verbose_eval=False,
        num_boost_round=100,
        early_stopping_rounds=50
        )

    params_list.append(best.params)
    score_list.append(tuned_model.best_score)
    models.append(best)

tunde_model = models[score_list.index(min(score_list))]
print(tunde_model.params)




pickle.dump(tuned_model, open('./model/tuned_lgbm.pickle', 'wb'))

line_notify('finished')