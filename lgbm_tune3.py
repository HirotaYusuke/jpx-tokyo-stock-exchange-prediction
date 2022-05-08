from pyexpat import model
import requests
import time
import pandas as pd, numpy as np
import seaborn as sns


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
    line_notify_token = ''
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

mape = []

# 評価指標MAPEの場合(仮定)
def mean_absolute_percentage_error(Y_test, Y_pred): 
    Y_test, Y_pred = np.array(Y_test), np.array(Y_pred)
    return np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100

pred_y1 = np.zeros(len(y_val))
pred_y2 = np.zeros(len(y_val))
models = []
score_list = []

for train, val in tscv.split(X_train):
    trn_x, val_x = X_train.iloc[train], X_train.iloc[val]
    trn_y, val_y = y_train.iloc[train], y_train.iloc[val]

    trn_ds = optuna_lgb.Dataset(trn_x, trn_y)
    val_ds = optuna_lgb.Dataset(val_x, val_y)

    opt = optuna_lgb.train(
        params, trn_ds, valid_sets=val_ds,
        verbose_eval=False,
        num_boost_round=100,
        early_stopping_rounds=50,
        show_progress_bar=False,
    )

    # tuningされたパラメータ
    tuned_model = lgb.train(
        opt.params, trn_ds, valid_sets=val_ds,
        verbose_eval=False, num_boost_round=100, early_stopping_rounds=50,
    )

    # デフォルトパラメーター
    model = lgb.train(
        params, trn_ds, valid_sets=val_ds,
        verbose_eval=False, num_boost_round=100, early_stopping_rounds=50,
    )

    pred_y1 += tuned_model.predict(X_val) / split
    pred_y2 += model.predict(X_val) / split
    
    models.append(tuned_model)
    score_list.append(tuned_model.best_score)

tunde_model = models[score_list.index(min(score_list))]
print(tunde_model.params)

# Tuningあり
print("Tuning: ", np.sqrt(mean_squared_error(y_val, pred_y1)))

# Tuningなし
print("not Tuning: ", np.sqrt(mean_squared_error(y_val, pred_y2)))

line_notify('finished')

