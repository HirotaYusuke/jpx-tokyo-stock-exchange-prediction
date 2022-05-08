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
    line_notify_token = ''
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

fit_params =  {
    'early_stopping_rounds': 10,
    'eval_metric':'r2',
    'eval_set':[[X_val, y_val]]
}

param_bayes = {
    'n_estimators':Integer(300, 1000),
    'max_depth':Integer(4, 32),
    'num_leaves':Integer(100, 1000),
    'reg_lambda':Real(1e-9, 1000, prior='log-uniform'),
    'reg_alpha':Real(1e-9, 1.0, prior='log-uniform'),
    'n_estimators':(100, 500),
    'learning_rate': Real(0.001, 0.5, prior='log-uniform')
}

split = 2
tscv = TimeSeriesSplit(n_splits=split)

model=LGBMRegressor(num_leaves=500, learning_rate=0.1, n_estimators=300)

bayes = BayesSearchCV(
    estimator=model,
    search_spaces=param_bayes,
    cv=tscv,
    n_jobs=-1,
    n_iter=150,
    verbose=1,
    scoring = 'r2',
    refit=True,
    fit_params=fit_params,
    return_train_score=True
    )


result = bayes.fit(X_train, y_train)

df_results = pd.DataFrame(result.cv_results_)
print(df_results.tail(5))
new_params = df_results.tail(1).params.values[0]
new_score  = df_results.tail(1).mean_test_score
print(f'Model #{len(df_results)}')
print(f'Best R2 ={new_score}')
print(f'Best params:{result.best_params_}')


R2 = result.score(X_val, y_val)
print(f'R2:{R2}')
pred = result.predict(X_val)
RMSE = mean_squared_error(pred, y_val, squared=False)
print('RMSE:{}'.format(RMSE))

line_notify('finished')