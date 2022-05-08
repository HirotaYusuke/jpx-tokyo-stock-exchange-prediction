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


model3=LGBMRegressor(
    learning_rate=0.001,
    max_depth=4,
    n_estimators=287,
    num_leaves=500,
    reg_alpha=1e-09,
    reg_lambda=6.478515977017151e-08
    )
model3.fit(X,y)
r2_3 = model3.score(X,y)
print(f'R2(date=datetime){r2_3}:')

line_notify('finished')