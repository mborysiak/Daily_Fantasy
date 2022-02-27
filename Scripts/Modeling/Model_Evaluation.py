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

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


def load_pickle(path, fname):
        with gzip.open(f"{path}/{fname}.p", 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object

def load_all_pickles(model_output_path, label):
    pred = load_pickle(model_output_path, f'{label}_pred')
    actual = load_pickle(model_output_path, f'{label}_actual')
    models = load_pickle(model_output_path, f'{label}_models')
    scores = load_pickle(model_output_path, f'{label}_scores')
    try: full_hold = load_pickle(model_output_path, f'{label}_full_hold')
    except: full_hold = None
    return pred, actual, models, scores, full_hold

def load_data(model_type, set_pos):

    # load data and filter down
    if model_type=='full_model': df = dm.read(f'''SELECT * FROM {set_pos}_Data''', 'Model_Features')
    elif model_type=='backfill': df = dm.read(f'''SELECT * FROM Backfill WHERE pos='{set_pos}' ''', 'Model_Features')

    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * FROM {set_pos}_Data2''', 'Model_Features')
        df = pd.concat([df, df2], axis=1)
    df = df.sort_values(by=['year', 'week']).reset_index(drop=True)

    drop_cols = list(df.dtypes[df.dtypes=='object'].index)
    print(drop_cols)

    return df, drop_cols


def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))


def create_game_date(df):
    
    # set up the date column for sorting
    df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
    cv_time_input = int(dt.datetime(val_year_min, 1, val_week_min).strftime('%Y%m%d'))
    train_time_split = int(dt.datetime(set_year, 1, set_week).strftime('%Y%m%d'))

    return df, cv_time_input, train_time_split


def train_predict_split(df, train_time_split, cv_time_input):

    # # get the train / predict dataframes and output dataframe
    df_train = df[df.game_date < train_time_split].reset_index(drop=True)
    df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)

    df_predict = df[df.game_date == train_time_split].reset_index(drop=True)
    output_start = df_predict[['player', 'dk_salary', 'fantasyPoints', 'projected_points', 'ProjPts']].copy().drop_duplicates()

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.game_date < cv_time_input].shape[0])  
    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, output_start, min_samples

#%%

#======================================================================================================================


#==================================================================
# SHAP Plots for Best Models
#==================================================================

# set year to analyze
set_year = 2021
set_week = 10

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 10

# set the model version
vers = 'standard_proba_sera_brier'
ensemble_vers = 'no_weight_yes_kbest_sera'
model_type ='full_model'
set_pos = 'RB'

     
pkey = f'{set_pos}_year{set_year}_week{set_week}_{model_type}{vers}'

model_output_path = f'{root_path}/Model_Outputs/{set_year}/{pkey}/'
if not os.path.exists(model_output_path): os.makedirs(model_output_path)

#==========
# Pull and clean compiled data
#==========

df, drop_cols = load_data(model_type, set_pos)
df, cv_time_input, train_time_split = create_game_date(df)
df_train, df_predict, output_start, min_samples = train_predict_split(df, train_time_split, cv_time_input)

skm = SciKitModel(df_train, model_obj='reg')
X_all, y = skm.Xy_split('y_act', drop_cols)
X = X_all.sample(frac=0.25, axis=1)
if 'game_date' not in X.columns:
    X['game_date'] = X_all.game_date

#------------
# Get Regression Data
#------------

from sklearn.pipeline import Pipeline

# set up the model pipe and get the default search parameters
pipe = skm.model_pipe([skm.piece('std_scale'), 
                       skm.piece('k_best'),
                       skm.piece('lgbm')])

# set params
params = skm.default_params(pipe, 'rand')
params['k_best__k'] = range(25, 200, 5)
# run the model with parameter search
best_models, oof_data, param_scores = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                        col_split='game_date', 
                                                        time_split=cv_time_input,
                                                        bayes_rand='custom_rand',
                                                        sample_weight=False,
                                                        random_seed=1234)

m = best_models[1]
transformer = Pipeline(m.steps[:-1])
X_shap = transformer.transform(X)
cols = X.columns[transformer['k_best'].get_support()]
X_shap = pd.DataFrame(X_shap, columns=cols)

import shap
shap_values = shap.TreeExplainer(m.steps[-1][1]).shap_values(X_shap)
shap.summary_plot(shap_values, X_shap, feature_names=X_shap.columns, plot_size=(8,15), max_display=30, show=False)

#%%

#======================================================================================================================

#==================================================================
# Regression for Winnings Optimization
#==================================================================

from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

df = dm.read('''SELECT * 
                FROM Winnings_Optimize
                ORDER BY year, week''', 'Results')

X = df[['adjust_pos_counts', 'drop_player_multiple',  'drop_team_frac', 'top_n_choices', 
        'week', 'covar_type', 'full_model_rel_weight', 'min_player_same_team', 'num_iters',
        'pred_proba', 'pred_sera', 'pred_brier', 'pred_lowsample', 
        'ens_sample_weights', 'ens_kbest', 'ens_randsample', 'ens_sera', 
        'std_spline', 'std_quantile']]

def one_hot(X):
    for c in ['week', 'covar_type']:
        X = pd.concat([X, pd.get_dummies(X[c], prefix=c, drop_first=True)], axis=1)
        if c!='week':
            X = X.drop(c, axis=1)
    return X


X = one_hot(X)
y = df.total_winnings

# m = Ridge(alpha=100)
m = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2)
# m = LGBMRegressor(n_estimators=25, max_depth=5, min_samples_leaf=1)

if type(m) == sklearn.linear_model._ridge.Ridge:
    sc = StandardScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X), columns=X.columns)

scores = cross_val_score(m, X, y, cv=5, scoring='neg_mean_squared_error')
scores = np.sqrt(-np.mean(scores))
print(scores)
m.fit(X,y)

try:
    pd.Series(m.coef_, index=X.columns).sort_values().plot.barh(figsize=(10,10))

except:
    import shap
    shap_values = shap.TreeExplainer(m).shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=X.columns, plot_size=(8,10), max_display=20, show=False)


# %%

# X_pred = pd.DataFrame({
#  'adjust_pos_counts': [1], 
#  'drop_player_multiple': [0], 
#  'drop_team_frac': [0],
#  'top_n_choices': [0], 
#  'week': [19], 
#  'full_model_rel_weight': [0], 
# #  'week_3': [0], 
#  'week_4': [0],
#  'week_5': [0], 
#  'week_6': [0], 
#  'week_7': [0], 
#  'week_8': [0], 
#  'week_9': [0], 
#  'week_10': [0], 
#  'week_11': [0],
#  'week_12': [0], 
#  'week_13': [1], 
#  'week_14': [0], 
#  'week_15': [0], 
#  'week_16': [0], 
#  'week_17': [0],
#  'week_18': [0], 
#  'pred_vers_standard_proba': [0],
#  'pred_vers_standard_proba_quant': [0],
#  'pred_vers_standard_proba_sweight': [1],
#  'ensemble_vers_linear_weight': [0],
#  'ensemble_vers_no_weight': [0],
#  'covar_type_team_points': [0], 
#  'std_dev_type_spline': [1],
#  }, index=[0])

# if type(m) == sklearn.linear_model._ridge.Ridge:
#     X_pred = pd.DataFrame(sc.transform(X_pred), columns=X_pred.columns)

# print('Optimal Avg Winnings:', m.predict(X_pred)[0])

# my_avg_winnings = dm.read('''SELECT DISTINCT week, year, my_total_winnings 
#                              FROM Winnings_Optimize''', 'Simulation').my_total_winnings.mean()
# print('My Avg Winnings:', my_avg_winnings)

#%%

#======================================================================================================================

#==================================================================
# Look at Hyperparameter Optimization
#==================================================================

reg_or_class = 'reg'
model_type = 'lgbm'

df = dm.read(f'''SELECT * 
                 FROM {reg_or_class}_{model_type}
                 WHERE scores < 10000''', 'Results')
df = df.drop(['model'], axis=1)

if reg_or_class == 'reg':
    df['input_features'] = df.feature_union__k_best__k + df.feature_union__agglomeration__n_clusters
    df.loc[df.k_best__k < df.input_features, 'k_best__k'] = df.input_features
    df.select_perc__percentile = df.select_perc__percentile.fillna(100)

else:
    df['input_features'] = df.feature_union__k_best_c__k + df.feature_union__agglomeration__n_clusters
    df.loc[df.k_best_c__k < df.input_features, 'k_best_c__k'] = df.input_features
    # df.select_perc_c__percentile = df.select_perc_c__percentile.fillna(100)


df = df.drop([ 'week'], axis=1)


def one_hot(X):
    # for c in [ 'pos', 'model_type', 'knn__weights', 'knn__algorithm']:
    for c in [ 'pos', 'model_type']:
        X = pd.concat([X, pd.get_dummies(X[c], prefix=c, drop_first=True)], axis=1).drop(c, axis=1)
    return X



X = one_hot(df)
X = X.drop('scores', axis=1)
y = df.scores

# m = Ridge(alpha=100)
# m = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2)
m = LGBMRegressor(n_estimators=50, max_depth=5, min_samples_leaf=1)

if type(m) == sklearn.linear_model._ridge.Ridge:
    sc = StandardScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X), columns=X.columns)

scores = cross_val_score(m, X, y, cv=5, scoring='neg_mean_squared_error')
scores = np.sqrt(-np.mean(scores))
print(scores)
m.fit(X,y)

try:
    pd.Series(m.coef_, index=X.columns).sort_values().plot.barh(figsize=(10,10))

except:
    import shap
    shap_values = shap.TreeExplainer(m).shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=X.columns, plot_size=(8,10), max_display=20, show=False)


# %%
