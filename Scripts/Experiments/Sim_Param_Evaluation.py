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
from sklearn.metrics import r2_score, brier_score_loss
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
#======================================================================================================================

#==================================================================
# Regression for Winnings Optimization
#==================================================================

from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import shap

model_type = {
 'enet': ElasticNet(alpha=5, l1_ratio=0.1),
 'ridge': Ridge(alpha=100),
 'rf': RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=10, n_jobs=-1),
 'lgbm': LGBMRegressor(n_estimators=50, max_depth=5, min_samples_leaf=5, n_jobs=-1)

}


def winnings_importance(df):

    df.loc[df.Contest=='ThreePointStance', ['total_winnings', 'max_winnings']] = df.loc[df.Contest=='ThreePointStance', ['total_winnings', 'max_winnings']] * 20 / 33
    df.loc[df.Contest=='ScreenPass', ['total_winnings', 'max_winnings']] = df.loc[df.Contest=='ScreenPass', ['total_winnings', 'max_winnings']] * 20 / 15

    X = df[['adjust_pos_counts', 'drop_player_multiple',  'drop_team_frac', 'top_n_choices', 
            'week', 'covar_type', 'full_model_rel_weight', 'min_player_same_team', 'num_iters',
            'pred_proba', 'pred_sera', 'pred_brier', 'pred_lowsample', 'proper_ensemble', 'pred_perc',
            'pred_sera_wt', 'pred_rsq_wt', 'pred_matt_wt', 'pred_brier_wt', 'pred_calibrate',
            'ens_sample_weights', 'ens_kbest', 'ens_randsample', 'ens_sera', 'ens_sera_wt', 'ens_rsq_wt',
            'std_dev_type', 'sim_type',
            'std_spline', 'std_quantile', 'std_experts', 'std_actuals', 'std_splquantile', 
            'std_predictions', 'std_coef', 'std_isotonic', 'std_calibrate', 'std_class',
            'std_matt_wt', 'std_brier_wt',
            'Contest']].copy()

    X['std_class'] = 0
    X.loc[X.std_dev_type.str.contains('class'), 'std_class'] = 1

    X['num_include'] = 1
    X.loc[X.std_dev_type.str.contains('include2'), 'num_include'] = 2
    X.loc[X.std_dev_type.str.contains('include3'), 'num_include'] = 3

    # X.loc[X.min_player_same_team == 'Auto', 'min_player_same_team'] = 2.5
    # X.min_player_same_team = X.min_player_same_team.astype('float')
    X.drop_player_multiple = X.drop_player_multiple.astype('object')
    def one_hot(X):
        for c in ['week', 'covar_type', 'std_dev_type', 'sim_type', 'Contest', 'min_player_same_team', 'drop_player_multiple']:
            X = pd.concat([X, pd.get_dummies(X[c], prefix=c, drop_first=False)], axis=1)
            if c!='week':
                X = X.drop(c, axis=1)
        return X

    X = one_hot(X).fillna(0)
    y = df.max_winnings

    return X, y


def get_model_coef(X, y, m):

    if type(m) == sklearn.linear_model._ridge.Ridge or type(m) == sklearn.linear_model._coordinate_descent.ElasticNet:
        sc = StandardScaler()
        sc.fit(X)
        X = pd.DataFrame(sc.transform(X), columns=X.columns)

    scores = cross_val_score(m, X, y, cv=5, scoring='neg_mean_squared_error')
    scores = np.sqrt(-np.mean(scores))
    print(scores)
    m.fit(X,y)

    try:
        coef_vals = pd.DataFrame(m.coef_, index=X.columns, columns=[f'week_{w}']).reset_index()
        coef_vals = coef_vals.rename(columns={'index': 'metric'})
    except:
        
        shap_values = shap.TreeExplainer(m).shap_values(X)
        coef_vals = pd.DataFrame(shap_values, columns=X.columns)
    
    return coef_vals, X


def join_coef(i, all_coef, coef_vals, X_all, X, m):

    if m in ('ridge', 'enet'):
        if i==0: 
            all_coef = coef_vals.copy()
            X_all=None
        else: 
            all_coef = pd.merge(all_coef, coef_vals, on='metric', how='outer').fillna(0)
            X_all = None

    if m in ('lgbm', 'rf'):
        if i==0: 
            all_coef = coef_vals.copy()
            X_all = X.copy()
        else: 
            all_coef = pd.concat([all_coef, coef_vals], axis=0, sort=False).fillna(0)
            X_all = pd.concat([X, X_all], sort=False, axis=0)
    return all_coef, X_all

def show_coef(all_coef, X_all):
    try:
        all_coef = pd.Series(all_coef.mean(axis=1).values, index=all_coef.metric)
        all_coef = all_coef[~all_coef.index.str.contains('entry_type')]
        all_coef[abs(all_coef) > 0.005].sort_values().plot.barh(figsize=(10,18))
    except:
        all_coef = all_coef[[c for c in all_coef.columns if 'week' not in c]]
        X_all = X_all[[c for c in X_all.columns if 'week' not in c]]
        shap.summary_plot(all_coef.values, X_all, feature_names=X_all.columns, plot_size=(18,10), max_display=40, show=False)

def entry_optimize_params(df, max_adjust, model_name):

    adjust_winnings = df.groupby(['trial_num', 'entry_type']).agg(max_lineup_num=('lineup_num', 'max')).reset_index()

    adjust_winnings.loc[adjust_winnings.entry_type=='millions_only', 'max_lineup_num'] = \
        13 / (adjust_winnings.loc[adjust_winnings.entry_type=='millions_only', 'max_lineup_num'] + 1)
    
    adjust_winnings.loc[adjust_winnings.entry_type=='millions_playaction', 'max_lineup_num'] = \
        30 / (adjust_winnings.loc[adjust_winnings.entry_type=='millions_playaction', 'max_lineup_num'] + 1)
    
    df = pd.merge(df, adjust_winnings.drop('entry_type', axis=1), on='trial_num')
    df.winnings = df.winnings / df.max_lineup_num
    
    df.loc[df.winnings >= max_adjust, 'winnings'] = max_adjust
    df.loc[(df.winnings >= 500) & (df.week==8) & (df.year==2022), 'winnings'] = 500

    df.loc[df.trial_num < 520, 'player_drop_multiple'] = 0

    str_cols = ['week', 'year', 'pred_vers', 'reg_ens_vers', 'million_ens_vers', 'std_dev_type']
    if model_name in ('enet', 'lasso',' ridge'):
        str_cols.extend( ['player_drop_multiple','top_n_choices', 'matchup_drop', 'adjust_pos_counts', 
                         'full_model_weight', 'max_lineup_num', 'use_ownership', 'own_neg_frac',
                         'num_top_players', 'static_top_players', 'num_iters',
                         'qb_min_iter', 'qb_solo_start', 'qb_set_max_team', 'num_avg_pts',
                         'qb_stack_wt'])
    df[str_cols] = df[str_cols].astype('str')

    df = df.drop(['trial_num', 'lineup_num'], axis=1)

    df.max_salary_remain = df.max_salary_remain.fillna(5000).astype('float').astype('int').astype('str')
    for c in df.columns:
        if df.dtypes[c] == 'object': 
            df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1).drop(c, axis=1)

    X = df.drop(['repeat_num', 'winnings'], axis=1)
    y = df.winnings

    return X, y
#%%
df = dm.read('''SELECT *  
                FROM Entry_Optimize_Params_Detail 
                JOIN (
                     SELECT week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, entry_type, trial_num, repeat_num
                      FROM Entry_Optimize_Results
                      ) USING (week, year, trial_num, repeat_num)
                WHERE trial_num >= 655
                     -- AND pred_vers = 'sera0_rsq0_mse1_brier1_matt1_bayes'
                      AND week < 17
                    --  AND NOT (week=8 AND year=2022)
                    --  AND (reg_ens_vers LIKE '%team_stats%' OR million_ens_vers LIKE '%team_stats%')
             
                ''', 'Results')

df['week'] = df.week.astype(str) + '_' + df.year.astype(str)
df.loc[df.week!='8_2022', 'winnings'] = df.loc[df.week!='8_2022', 'winnings']*2

model_type = {
 'enet': ElasticNet(alpha=1, l1_ratio=0.1),
 'lasso': Lasso(alpha=0.1),
 'ridge': Ridge(alpha=100),
 'rf': RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=10, n_jobs=-1),
 'lgbm': LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.05, min_samples_leaf=5, n_jobs=-1)

}
w=1
model_name='lgbm'
m = model_type[model_name] 
X, y = entry_optimize_params(df, max_adjust=10000, model_name=model_name)
coef_vals, X = get_model_coef(X, y, m)
show_coef(coef_vals, X)

#%%

weeks = [
         1, 2, 3, 4, 5, 6, 7, 8, 
         9, 10, 11, 12, 13, 14, 15, 16,
         1, 2, 3, 4, 5, 6,
          # 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
           ]
years = [
          2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 
          2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 
          2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023,
          2023, 2023, 2023, 2023, 2023, 2023]

i=0
all_coef = None; X_all = None
for w, yr in zip(weeks, years):
    df = dm.read(f'''SELECT *  
                     FROM Entry_Optimize_Params_Detail 
                     JOIN (
                            SELECT week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, entry_type, trial_num, repeat_num
                            FROM Entry_Optimize_Results          
                          ) USING (week, year, trial_num, repeat_num)
                     WHERE trial_num >= 655
                         --  AND pred_vers = 'sera0_rsq0_mse1_brier1_matt1_bayes'
                        --   AND (reg_ens_vers LIKE '%team_stats%' OR million_ens_vers LIKE '%team_stats%')
                           AND week = {w}
                           AND year = {yr}
                     ''', 'Results')
    df['week'] = df.week.astype(str) + '_' + df.year.astype(str)
    df.loc[df.week!='8_2022', 'winnings'] = df.loc[df.week!='8_2022', 'winnings']*2

    model_name = 'enet'
    m = model_type[model_name]
    if w == 8 and yr==2022: max_adjust = 1000
    else: max_adjust = 10000
    X, y = entry_optimize_params(df, max_adjust=max_adjust, model_name=model_name)
    coef_vals, X = get_model_coef(X, y, m)
    all_coef, X_all = join_coef(i, all_coef, coef_vals, X_all, X, model_name); i+=1

show_coef(all_coef, X_all)


#%%

def entry_optimize_bayes(df):

    for c in df.columns:
        if df.dtypes[c] == 'object': 
            df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1).drop(c, axis=1)

    X = df.drop('loss', axis=1)
    y = -df.loss

    return X, y

df = dm.read('''SELECT *  
                FROM Entry_Optimize_Bayes
                ''', 'Results')

model_type = {
 'enet': ElasticNet(alpha=1, l1_ratio=0.1),
 'lasso': Lasso(alpha=0.1),
 'ridge': Ridge(alpha=100),
 'rf': RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=10, n_jobs=-1),
 'lgbm': LGBMRegressor(n_estimators=50, max_depth=5, min_samples_leaf=5, n_jobs=-1)

}

model_name='enet'
m = model_type[model_name] 
X, y = entry_optimize_bayes(df)
coef_vals, X = get_model_coef(X, y, m)
show_coef(coef_vals, X)


#%%

df_all = dm.read('''SELECT *  
                     FROM Entry_Optimize_Bayes
                    ''', 'Results')

for tn in df_all.trial_name.unique():
    df = df_all[df_all.trial_name == tn].copy().reset_index(drop=True)

    model_name = 'enet'
    m = model_type[model_name]
    X, y = entry_optimize_bayes(df)
    coef_vals, X = get_model_coef(X, y, m)
    all_coef, X_all = join_coef(i, all_coef, coef_vals, X_all, X, model_name); i+=1

show_coef(all_coef, X_all)


#%%
#
