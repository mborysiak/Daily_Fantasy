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

    
    df.loc[df.winnings >= max_adjust, 'winnings'] = max_adjust
    # df.loc[(df.winnings >= 500) & (df.week==8) & (df.year==2022), 'winnings'] = 500

    df.loc[df.use_ownership==1, 'ownership_vers'] = 'None'

    df['max_overlap_effective_non_qb'] = df.max_overlap
    df.loc[df.overlap_constraint=='plus_wts', 'max_overlap_effective_non_qb'] = \
        df.loc[df.overlap_constraint=='plus_wts', 'max_overlap'] + \
            df.loc[df.overlap_constraint=='plus_wts', 'prev_qb_wt'] + \
                df.loc[df.overlap_constraint=='plus_wts', 'prev_def_wt']

    df.loc[df.overlap_constraint=='minus_one', 'max_overlap_effective_non_qb'] = \
        df.loc[df.overlap_constraint=='minus_one', 'max_overlap'] + \
            df.loc[df.overlap_constraint=='minus_one', 'prev_qb_wt'] + \
                df.loc[df.overlap_constraint=='minus_one', 'prev_def_wt'] - 2

    df.loc[df.overlap_constraint=='div_two', 'max_overlap_effective_non_qb'] = \
        df.loc[df.overlap_constraint=='div_two', 'max_overlap'] + \
            (df.loc[df.overlap_constraint=='div_two', 'prev_qb_wt']/2) + \
                (df.loc[df.overlap_constraint=='div_two', 'prev_def_wt']/2)

    df = df[df.max_overlap_effective_non_qb < 9].reset_index(drop=True)
    
    str_cols = ['week', 'year', 'pred_vers', 'reg_ens_vers', 'million_ens_vers', 'std_dev_type']
    if model_name in ('enet', 'lasso',' ridge'):
        str_cols.extend( ['full_model_rel_weight', 'max_overlap', 'max_salary_remain', 'max_teams_lineup',
                          'min_opp_team', 'num_avg_pts', 'num_options', 'prev_qb_wt', 'qb_te_stack', 'qb_wr_stack',
                          'wr_flex_pct', 'rb_flex_pct', 'use_ownership', 'max_overlap_effective_non_qb'])
    df[str_cols] = df[str_cols].astype('str')

    df = df.drop(['trial_num'], axis=1)

    df.max_salary_remain = df.max_salary_remain.fillna(5000).astype('float').astype('int').astype('str')
    for c in df.columns:
        if df.dtypes[c] == 'object': 
            df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1).drop(c, axis=1)

    X = df.drop(['repeat_num','winnings'], axis=1)
    y = df.winnings

    return X, y

#%%

weeks = [
         2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
         1,2,3,4,5,7,8, 9, 
         6,
         1,
         8
           ]
years = [
          2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 
          2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023,
          2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024,
          2024,
          2022,
          2022
          ]

i=0
all_coef = None; X_all = None
for w, yr in zip(weeks, years):
    df = dm.read(f'''SELECT *  
                     FROM Entry_Optimize_Params_Detail 
                     JOIN (
                            SELECT week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, entry_type, trial_num, repeat_num
                            FROM Entry_Optimize_Results          
                          ) USING (week, year, trial_num, repeat_num)
                     WHERE week = {w}
                           AND year = {yr}
                           AND trial_num > 57
                     ''', 'ResultsNew')
    df['week'] = df.week.astype(str) + '_' + df.year.astype(str)
    # df.loc[df.week!='8_2022', 'winnings'] = df.loc[df.week!='8_2022', 'winnings']*2

    model_name = 'enet'
    m = model_type[model_name]
    # if w == 8 and yr==2022: max_adjust = 1000
    # else: max_adjust = 10000
    max_adjust = 15000
    X, y = entry_optimize_params(df, max_adjust=max_adjust, model_name=model_name)
    coef_vals, X = get_model_coef(X, y, m)
    all_coef, X_all = join_coef(i, all_coef, coef_vals, X_all, X, model_name); i+=1

show_coef(all_coef, X_all)

#%%
df = dm.read('''SELECT *  
                FROM Entry_Optimize_Params_Detail 
                JOIN (
                     SELECT week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, entry_type, trial_num, repeat_num
                     FROM Entry_Optimize_Results
                      ) USING (week, year, trial_num, repeat_num)
                     WHERE week < 17

                ''', 'ResultsNew')

df.ownership_vers = df.ownership_vers.apply(lambda x: x.replace('{', '').replace('}', '').replace('0. ', '0.0 ').replace(':', '').replace(',',''))

df['week'] = df.week.astype(str) + '_' + df.year.astype(str)
# df.loc[df.week!='8_2022', 'winnings'] = df.loc[df.week!='8_2022', 'winnings']*2

model_type = {
 'enet': ElasticNet(alpha=1, l1_ratio=0.1),
 'lasso': Lasso(alpha=0.1),
 'ridge': Ridge(alpha=100),
 'rf': RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=10, n_jobs=-1),
 'lgbm': LGBMRegressor(n_estimators=50, max_depth=10, learning_rate=0.01, min_samples_leaf=25, n_jobs=-1)

}
w=1
model_name='lgbm'
m = model_type[model_name] 
X, y = entry_optimize_params(df, max_adjust=15000, model_name=model_name)
coef_vals, X = get_model_coef(X, y, m)
show_coef(coef_vals, X)

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
