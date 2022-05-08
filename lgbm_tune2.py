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
from lightgbm import LGBMRegressor, early_stopping
from lightgbm import log_evaluation
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# LINE nitification
def line_notify(message):
    line_notify_token = 'RjPVIZw064cOaSknExnBVidycTcfZJiHdCOvB5AMS3k'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token} 
    requests.post(line_notify_api, data=payload, headers=headers)

# load data
X = pd.read_pickle('./data/X2.pickle')
y = pd.read_pickle('./data/y2.pickle')
X_train = pd.read_pickle('./data/X_train2.pickle')
X_val = pd.read_pickle('./data/X_val2.pickle')
y_train = pd.read_pickle('./data/y_train2.pickle')
y_val = pd.read_pickle('./data/y_val2.pickle')



X_train = X_train.values
X_valid = X_val.values

y_train = y_train.values
y_valid = y_val.values

trains = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
valids = lgb.Dataset(X_val, label=y_valid, free_raw_data=False)

def objective(trial):

    # ハイパーパラメータの設定
    param = {
        "objective": "regression",
        "metric": {"rmse"},
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 50, 600),
        #"feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        #"bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        #"bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        #"min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    # 学習
    gbm = lgb.train(param, trains)

    # 予測
    preds = gbm.predict(X_valid)
    # 精度の算出
    rmse = np.sqrt(mean_squared_error(y_valid, preds))

    return rmse


if __name__ == "__main__":
    # oputuna による最適化
    study = optuna.create_study(direction="minimize")
    # 探索数(試行数)を指定する場合
    #study.optimize(objective, n_trials=1000)
    # 探索時間を指定する場合
    study.optimize(objective, timeout=60)

    #print("Number of finished trials: {}".format(len(study.trials)))

    #print("Best trial:")

    #print("  Value: {}".format(study.best_trial.value))

    #print("  Params: ")

    tuned_params = {}
    for key, value in study.best_trial.params.items():
        #print("    {}: {}".format(key, value))
        tuned_params[key] = value

    #print(tuned_params)
    
    #print(study.best_trial.value)

tuned_params = tuned_params

valid_socres = []
rmse_list = []
r2_list = []
tuned_models = []
#kf = KFold(n_splits=5, shuffle=False)
split = 3
tscv = TimeSeriesSplit(n_splits=split)

for fold, (train_index, valid_index) in enumerate(tscv.split(X_train)):

    X_tr, X_val = X_train[train_index], X_train[valid_index] 
    y_tr, y_val = y_train[train_index], y_train[valid_index]
    
    trains = lgb.Dataset(data=X_tr, label=y_tr, feature_name='auto') # dataにはテストデータ，labelには正解データ
    evals = lgb.Dataset(data=X_val, label=y_val, feature_name='auto') # feature_name=’auto’とすることで DataFrameの列名が認識される
    
    model = lgb.train(
        tuned_params,
        trains,
        valid_sets=evals,
        num_boost_round=50,
        early_stopping_rounds=10,
        verbose_eval=-1
    )

    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)
    rmse_list.append(rmse)
    r2_list.append(r2)
    valid_socres.append(pred)

    tuned_models.append(model)


print(f'rmse: {np.mean(rmse_list)}')
print(f'r2: {np.mean(r2_list)}')

X_train = pd.read_pickle('./data/X_train2.pickle')
X_val = pd.read_pickle('./data/X_val2.pickle')
y_train = pd.read_pickle('./data/y_train2.pickle')
y_val = pd.read_pickle('./data/y_val2.pickle')

X_train = X_train.values
X_valid = X_val.values

y_train = y_train.values
y_valid = y_val.values

trains = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
valids = lgb.Dataset(X_val, label=y_valid, free_raw_data=False)

tuned_models = lgb.train(
    tuned_params,
    trains,
    valid_sets=valids,
    num_boost_round=50,
    early_stopping_rounds=10,
    verbose_eval=-1
    )

print(tuned_models.best_score)