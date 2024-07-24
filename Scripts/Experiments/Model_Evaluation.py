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
from sklearn.metrics import r2_score, brier_score_loss, mean_squared_error
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
    return pd.Series(dict(r2 = r2, rmse = rmse))

actuals = dm.read('''SELECT player, pos, week, year, y_act, avg_proj_points
                     FROM Backfill  
                  ''', 'Model_Features')

preds = dm.read(f''' 
                SELECT *
                FROM Model_Predictions
                WHERE week < 17
                ''', 'Simulation')

df = pd.merge(preds, actuals, on=['player', 'pos', 'week', 'year'])
df['pred_diff'] = df.pred_fp_per_game - df.avg_proj_points

df = (
     df
     .groupby(['week','year', 'pos', 'model_type', 'pred_vers', 'reg_ens_vers', 'std_dev_type'])
     .apply(r2_rmse)
).reset_index()

# %%


#=================
# Compare chosen players between runs
#=================

run1 = {'week': 11,
        'pred_vers': 'fixed_model_clone',
        'ensemble_vers': 'no_weight_no_kbest_randsample_sera_logparams_include3',
        'std_dev_type': 'pred_spline',
        'sim_type': 'v2'}

run2 = {'week': 11,
        'pred_vers': 'fixed_model_clone_proba_sera_perc',
        'ensemble_vers': 'no_weight_no_kbest_randsample_sera_logparams_include3',
        'std_dev_type': 'pred_spline',
        'sim_type': 'v2'
        }

def get_lineups(r, label):
    df = dm.read(f'''SELECT * 
                     FROM Lineups_Optimize
                     WHERE week={r['week']}
                             AND pred_vers='{r['pred_vers']}'
                             AND ensemble_vers='{r['ensemble_vers']}'
                             AND std_dev_type='{r['std_dev_type']}' 
                           --  AND sim_type='{r['sim_type']}'
                            ''', 'Results')

    df = pd.DataFrame(pd.melt(df.iloc[:, :8]).value.value_counts() / df.shape[0]).reset_index()
    df.columns = ['player', label]


    df = pd.merge(df, pred, on='player')

    return df

def add_actual(r, set_pos):

    if set_pos=='Defense': pl = 'defTeam'
    else: pl = 'player'
    actual_pts = dm.read(f'''SELECT {pl} player, fantasy_pts actual_pts
                            FROM {set_pos}_Stats 
                            WHERE week={r['week']} 
                                and season=2021''', 'FastR')
    return actual_pts

pts = pd.DataFrame()
for p in ['QB', 'RB','WR', 'TE', 'Defense']:
    pts = pd.concat([pts, add_actual(run1, p)])
 
df = get_lineups(run1, 'best_run')
df2 = get_lineups(run2, 'recent_run')

df3 = pd.merge(df, df2, on=['player'])
df3 = pd.merge(df3, pts, on='player')

df3['pct_diff'] = df3.best_run - df3.recent_run
df3 = df3[['player', 'best_run', 'recent_run','pct_diff',
           'pred_fp_per_game_best_run', 'pred_fp_per_game_recent_run',
           'std_dev_best_run', 'std_dev_recent_run', 'actual_pts']]
df3.sort_values(by='best_run', ascending=False).iloc[:50]

# %%

df3.sort_values(by='pct_diff', ascending=False).iloc[:50]

# %%
df3.sort_values(by='pct_diff', ascending=True).iloc[:50]




# %%
