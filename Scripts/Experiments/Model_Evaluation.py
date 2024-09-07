#%%
from skmodel import SciKitModel
from ff.db_operations import DataManage
from ff import general as ffgeneral
import pandas as pd
import numpy as np
import os
import pickle
import datetime as dt
import gzip
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, brier_score_loss, mean_squared_error, brier_score_loss
import zModel_Functions as mf
import matplotlib.pyplot as plt
from hyperopt import Trials
from warnings import simplefilter 
from joblib import Parallel, delayed

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

def load_pickle(path, fname):
        with gzip.open(f"{path}/{fname}.p", 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object
        
def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

#%%


def r2_rmse(g):
    r2 = r2_score(g['y_act'], g['pred_fp_per_game'])
    rmse = np.sqrt(mean_squared_error(g['y_act'], g['pred_fp_per_game']))
    brier_score = brier_score_loss(g['is_million'], g['pred_million'])
    r2_score_mil = r2_score(g['is_million'], g['pred_million'])
    return pd.Series(dict(r2 = r2, rmse=rmse, brier_score=brier_score, r2_score_mil=r2_score_mil))

actuals = dm.read('''SELECT player, pos, week, year, y_act, avg_proj_points
                     FROM Backfill  
                  ''', 'Model_Features')

preds = dm.read(
                f''' 
                SELECT *
                FROM Model_Predictions
                WHERE week < 17
                ''', 'Simulation'
                )

mil_pred = dm.read(
                f''' 
                SELECT player, week, year, pred_vers, million_ens_vers, model_type, pred_fp_per_game as pred_million
                FROM Predicted_Million
                WHERE week < 17
                ''', 'Simulation'
                )

top_players = dm.read(
                f'''
                SELECT player, week, year, y_act as is_million
                FROM Top_Players
                WHERE week < 17
                ''', 'DK_Results'
)

mil_pred = pd.merge(mil_pred, top_players, on=['player', 'week', 'year'], how='left').fillna({'is_million': 0})

data = pd.merge(preds, actuals, on=['player', 'pos', 'week', 'year'])
data = pd.merge(data, mil_pred, on=['player', 'week', 'year', 'pred_vers', 'model_type'], how='inner')

data = (
     data
     .groupby(['week','year', 'pos', 'model_type', 'pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers'])
     .apply(r2_rmse)
).reset_index()

#%%
df = pd.pivot_table(data, index=['week', 'year', 'pred_vers', 'reg_ens_vers', 'million_ens_vers', 'std_dev_type'], 
                    columns=['pos', 'model_type'], values=['r2', 'r2_score_mil'])  
df = df.reset_index()
df.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in df.columns.values]

results = dm.read('''SELECT * FROM Entry_Optimize_Results''', 'Results')
results = results.groupby(['week', 'pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers']).apply(lambda x: np.mean(x['avg_winnings']))
results = results.reset_index().rename(columns={0: 'avg_winnings'})
df = pd.merge(df, results, on=['week', 'pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers'])
df = df.dropna().reset_index(drop=True)

Xy = pd.concat([df, pd.get_dummies(df['std_dev_type'])], axis=1).drop(columns=['std_dev_type'])
X_train = Xy[Xy.pred_vers != '']

test_vers = 'sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_higherkb'
X_train = Xy.loc[Xy.pred_vers!= test_vers].drop(columns=['avg_winnings', 'week', 'year', 'pred_vers', 'reg_ens_vers', 'million_ens_vers']).reset_index(drop=True)
y_train = Xy.loc[Xy.pred_vers!= test_vers, 'avg_winnings'].reset_index(drop=True)

X_test_labels = Xy.loc[Xy.pred_vers== test_vers, ['week', 'year', 'pred_vers', 'reg_ens_vers', 'million_ens_vers']]
X_test = Xy.loc[Xy.pred_vers== test_vers].drop(columns=['avg_winnings', 'week', 'year', 'pred_vers', 'reg_ens_vers', 'million_ens_vers']).reset_index(drop=True)
y_test = Xy.loc[Xy.pred_vers== test_vers, 'avg_winnings'].reset_index(drop=True)

#%%


scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
lr = ElasticNet(alpha=1, l1_ratio=1)
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
coeffs = pd.DataFrame(lr.coef_, index=X_train.columns, columns=['coeff'])
coeffs = coeffs.sort_values(by='coeff', ascending=False)
coeffs

#%%

lgbm = LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
lgbm.fit(X_train, y_train)


#%%
X_test_labels['pred'] = lgbm.predict(X_test)
X_test_labels['actual'] = y_test
X_test_labels['diff'] = X_test_labels['pred'] - X_test_labels['actual']
X_test_labels.plot.scatter(x='pred', y='actual')
# %%

coeffs = pd.DataFrame(lgbm.feature_importances_, index=X_train.columns, columns=['coeff'])
coeffs = coeffs.sort_values(by='coeff', ascending=False)
coeffs
# %%
