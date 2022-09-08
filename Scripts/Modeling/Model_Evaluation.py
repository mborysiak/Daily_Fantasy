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
from sklearn.linear_model import Ridge, ElasticNet
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

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
set_week = 17

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

df_train['y_act'] = ( df_train.dk_salary / (df_train.y_act+2.020403))

skm = SciKitModel(df_train, model_obj='reg')
X_all, y = skm.Xy_split('y_act', drop_cols)

X = X_all.sample(frac=1, axis=1)
if 'game_date' not in X.columns:
    X['game_date'] = X_all.game_date

#------------
# Get Regression Data
#------------

from sklearn.pipeline import Pipeline

# set up the model pipe and get the default search parameters
pipe = skm.model_pipe([skm.piece('std_scale'), 
                       skm.piece('k_best'),
                       skm.piece('xgb')])

# set params
params = skm.default_params(pipe, 'rand')
params['k_best__k'] = range(20, 200, 5)


# run the model with parameter search
best_models, oof_data, param_scores = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                        col_split='game_date', 
                                                        time_split=cv_time_input,
                                                        bayes_rand='custom_rand',
                                                        sample_weight=False,
                                                        random_seed=1234)

oof_data['full_hold'].plot.scatter(x='pred', y='y_act')

#%%
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
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

for i, w in enumerate([8, 9, 10, 11, 12, 13, 14, 15, 16]):

    df = dm.read(f'''SELECT * 
                    FROM Winnings_Optimize
                    WHERE week={w}
                    ORDER BY year, week''', 'Results')

    X = df[['adjust_pos_counts', 'drop_player_multiple',  'drop_team_frac', 'top_n_choices', 
            'week', 'covar_type', 'full_model_rel_weight', 'min_player_same_team', 'num_iters',
            'pred_proba', 'pred_sera', 'pred_brier', 'pred_lowsample', 'proper_ensemble', 'pred_perc',
            'ens_sample_weights', 'ens_kbest', 'ens_randsample', 'ens_sera', 
            'std_dev_type', 'sim_type',
            'std_spline', 'std_quantile', 'std_experts', 'std_actuals', 'std_splquantile', 
            'std_predictions', 'std_coef', 'std_isotonic']]

    X.loc[X.min_player_same_team == 'Auto', 'min_player_same_team'] = 2.5
    X.min_player_same_team = X.min_player_same_team.astype('float')
    def one_hot(X):
        for c in ['week', 'covar_type', 'std_dev_type', 'sim_type']:
            X = pd.concat([X, pd.get_dummies(X[c], prefix=c, drop_first=False)], axis=1)
            if c!='week':
                X = X.drop(c, axis=1)
        return X


    X = one_hot(X).fillna(0)
    y = df.max_winnings

    # m = ElasticNet(alpha=5, l1_ratio=0.1)
    # m = Ridge(alpha=100)
    # m = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=10, n_jobs=-1)
    m = LGBMRegressor(n_estimators=50, max_depth=5, min_samples_leaf=5, n_jobs=-1)

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
        # coef_vals[abs(coef_vals) > 0.01].sort_values().plot.barh(figsize=(10,10))

    except:
        import shap
        shap_values = shap.TreeExplainer(m).shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=X.columns, plot_size=(8,10), max_display=30, show=False)

    if i==0: all_coef = coef_vals.copy()
    else: all_coef = pd.merge(all_coef, coef_vals, on='metric', how='outer').fillna(0)

coef_vals = pd.Series(all_coef.mean(axis=1).values, index=all_coef.metric)
coef_vals[abs(coef_vals) > 0.01].sort_values().plot.barh(figsize=(10,10))
# %%

skm = SciKitModel(pd.DataFrame({'x': [1, 2]}))
df = dm.read("SELECT * FROM Model_Validations", 'Simulation')
gcols = ['set_week', 'set_year', 'pos', 'pred_version', 'ensemble_vers', 'model_type']
df = df.groupby(gcols).apply(lambda x: skm.test_scores(x['y_act'], x['pred_fp_per_game'])[0]).reset_index()
df

#%%

skm = SciKitModel(pd.DataFrame({'x': [1, 2]}))
df = dm.read("SELECT * FROM Model_Test_Validations WHERE model_type='full_model'", 'Simulation')
gcols = ['set_week', 'set_year', 'pos', 'pred_version', 'ensemble_vers', 'model_type']
df = df.groupby(gcols).apply(lambda x: skm.test_scores(x['actual_pts'], x['pred_fp_per_game'])[0]).reset_index()
df.sort_values(by=['set_year', 'set_week', 'pos', 0],
               ascending=[True, True, False, False])

#%%

#======================================================================================================================

#==================================================================
# Look at Hyperparameter Optimization
#==================================================================



reg_or_class = 'reg'
model_type = 'enet'

df = dm.read(f'''SELECT * 
                 FROM {reg_or_class}_{model_type}
                 WHERE scores < 1
                        ''', 'Results')
df = df.drop(['model'], axis=1).dropna()

if reg_or_class == 'reg':
    df['input_features'] = df.feature_union__k_best__k + df.feature_union__agglomeration__n_clusters
    df.loc[df.k_best__k < df.input_features, 'k_best__k'] = df.input_features
    df.select_perc__percentile = df.select_perc__percentile.fillna(100)

else:
    df['input_features'] = df.feature_union__k_best_c__k + df.feature_union__agglomeration__n_clusters
    df.loc[df.k_best_c__k < df.input_features, 'k_best_c__k'] = df.input_features
    # df.select_perc_c__percentile = df.select_perc_c__percentile.fillna(100)

if model_type == 'knn':
    df = pd.concat([df, pd.get_dummies(df.knn__weights), pd.get_dummies(df.knn__algorithm)], axis=1).drop(['knn__weights', 'knn__algorithm'], axis=1)

df = df.drop([ 'week'], axis=1)


def one_hot(X):
    # for c in [ 'pos', 'model_type', 'knn__weights', 'knn__algorithm']:
    for c in [ 'pos', 'model_type']:
        X = pd.concat([X, pd.get_dummies(X[c], prefix=c, drop_first=True)], axis=1).drop(c, axis=1)
    return X



X = one_hot(df)
X = X.drop('scores', axis=1)
y = -df.scores

# m = ElasticNet(alpha=0.01, l1_ratio=0.05)
# m = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2)
m = LGBMRegressor(n_estimators=50, max_depth=5, min_samples_leaf=1)

if type(m) == sklearn.linear_model._ridge.Ridge or type(m)==sklearn.linear_model._coordinate_descent.ElasticNet:
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
    pred = dm.read(f''' 
                     SELECT player, 
                            AVG(pred_fp_per_game) pred_fp_per_game_{label},
                            AVG(std_dev) std_dev_{label}
                          --  AVG(max_score) max_score_{label}, 
                          --  1000 *AVG(pred_fp_per_game) / AVG(dk_salary) pts_salary_{label}
                     FROM Model_Predictions
                     WHERE week={r['week']}
                             AND version='{r['pred_vers']}'
                             AND ensemble_vers='{r['ensemble_vers']}'
                             AND std_dev_type='{r['std_dev_type']}'
                         --    AND sim_type='{r['sim_type']}'
                     GROUP BY player
    ''', 'Simulation')

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
