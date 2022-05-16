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
from skopt import BayesSearchCV
from lightgbm import early_stopping
from lightgbm import log_evaluation
from lightgbm import LGBMRegressor
import optuna
import optuna.integration.lightgbm as lgb_tune
import optuna.integration.lightgbm as optuna_lgb
from sklearn.model_selection import KFold

from skopt.space import Real, Categorical, Integer

# LINE nitification
def line_notify(message):
    line_notify_token = ''
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token} 
    requests.post(line_notify_api, data=payload, headers=headers)

prices = pd.read_csv("./data/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")

average = pd.DataFrame(prices.groupby("SecuritiesCode").Target.mean())
def get_avg(_id_):
    return average.loc[_id_]
prices["Avg"] = prices["SecuritiesCode"].apply(get_avg)

prices.Date = pd.to_datetime(prices.Date)
prices['Date'] = prices['Date'].dt.strftime("%Y%m%d").astype(int)
X=prices[["Date","SecuritiesCode","Avg"]]
y=prices[["Target"]]
codes = X.SecuritiesCode.unique()


model1=LGBMRegressor(num_leaves=500, learning_rate=0.1, n_estimators=300)
model1.fit(X,y)
r2_1 = model1.score(X,y)
print(f'R2(date=int):{r2_1}')

#=======================================================================

X=prices[["Date","SecuritiesCode","Open", "High", "Low", "Close", "Volume", "Avg",]]
y=prices[["Target"]]

model2=LGBMRegressor(num_leaves=500, learning_rate=0.1, n_estimators=300)
model2.fit(X,y)
r2_2 = model2.score(X,y)
print(f'R2(date=datetime){r2_2}:')

#=======================================================================
split = 5
tscv = TimeSeriesSplit(n_splits=split)

seed = 42
import optuna

code_1301=prices[prices['SecuritiesCode']==1301]
code_1332=prices[prices['SecuritiesCode']==1301]

X_train = code_1301[["Date","SecuritiesCode","Open", "High", "Low", "Close", "Volume", "Avg"]]
X_val = code_1332[["Date","SecuritiesCode","Open", "High", "Low", "Close", "Volume", "Avg"]]
y_train = code_1301['Target']
y_val = code_1332['Target']


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
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
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
    study.optimize(objective, n_trials=1000)
    # 探索時間を指定する場合
    #study.optimize(objective, timeout=60)

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

#=======================================================================
'''
fit_params =  {
    'early_stopping_rounds': 10,
    'eval_metric':'r2',
    #'eval_set':[[X_val, y_val]]
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


bayes = BayesSearchCV(
    estimator=model2,
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


result = bayes.fit(X, y)

df_results = pd.DataFrame(result.cv_results_)
print(df_results.tail(5))
new_params = df_results.tail(1).params.values[0]
new_score  = df_results.tail(1).mean_test_score
print(f'Model #{len(df_results)}')
print(f'Best R2 ={new_score}')
print(f'Best params:{result.best_params_}')


'''
'''tuned_params = result.best_params_

model3=LGBMRegressor(tuned_params)
model3.fit(X,y)
r2_3 = model3.score(X,y)
print(f'R2(date=datetime){r2_3}:')'''



line_notify('finished')