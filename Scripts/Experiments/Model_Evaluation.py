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
from sklearn.metrics import r2_score
import zModel_Functions as mf

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


def load_data(model_type, set_pos, run_params):

    # load data and filter down
    if model_type=='full_model': df = dm.read(f'''SELECT * 
                                                  FROM {set_pos}_Data{run_params['rush_pass']}
                                                ''', 'Model_Features')
    elif model_type=='backfill': df = dm.read(f'''SELECT * 
                                                  FROM Backfill 
                                                  WHERE pos='{set_pos}' ''', 'Model_Features')

    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * 
                          FROM {set_pos}_Data{run_params['rush_pass']}2
                       ''', 'Model_Features')
        df = pd.concat([df, df2], axis=1)

    df = df.sort_values(by=['year', 'week']).reset_index(drop=True)
    if set_pos == 'Defense':
        df['team'] = df.player
        
    drop_cols = list(df.dtypes[df.dtypes=='object'].index)
    run_params['drop_cols'] = drop_cols
    print(drop_cols)

    return df, run_params

def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))


def get_cv_time_input(df, back_weeks):
    max_date = str(df.game_date.max())
    year = int(max_date[:4])
    week = int(max_date[-2:])

    for i in range(back_weeks):
        if week > 1:
            week -= 1
        else:
            year -= 1
            week = 17
    cv_time_input = int(dt.datetime(year, 1, week).strftime('%Y%m%d'))
    print(f'Begin Validation on Week {week}, {year}')
    return cv_time_input


def create_game_date(df, run_params):
    
    # set up the date column for sorting
    df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
    cv_time_input = get_cv_time_input(df, run_params['back_weeks'][set_pos])
    train_time_split = int(dt.datetime(run_params['set_year'], 1, run_params['set_week']).strftime('%Y%m%d'))

    run_params['cv_time_input'] = cv_time_input
    run_params['train_time_split'] = train_time_split

    return df, run_params


def train_predict_split(df, run_params):

    # # get the train / predict dataframes and output dataframe
    df_train = df[df.game_date < run_params['train_time_split']].reset_index(drop=True)
    df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)

    df_predict = df[df.game_date == run_params['train_time_split']].reset_index(drop=True)
    output_start = df_predict[['player', 'dk_salary', 'fantasyPoints', 'projected_points', 'ProjPts']].copy().drop_duplicates()

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.game_date < run_params['cv_time_input']].shape[0])  
    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, output_start, min_samples

def get_class_data(df, cut, run_params):

    # set up the training and prediction datasets for the classification 
    df_train_class = df[df.game_date < run_params['train_time_split']].reset_index(drop=True)
    df_predict_class = df[df.game_date == run_params['train_time_split']].reset_index(drop=True)

    # set up the target variable to be categorical based on Xth percentile
    cut_perc = df_train_class.groupby('game_date')['y_act'].apply(lambda x: np.percentile(x, cut))
    df_train_class = pd.merge(df_train_class, cut_perc.reset_index().rename(columns={'y_act': 'cut_perc'}), on=['game_date'])
    
    if cut > 33:
        df_train_class['y_act'] = np.where(df_train_class.y_act >= df_train_class.cut_perc, 1, 0)
    else: 
        df_train_class['y_act'] = np.where(df_train_class.y_act <= df_train_class.cut_perc, 1, 0)

    df_train_class = df_train_class.drop('cut_perc', axis=1)

    return df_train_class, df_predict_class

#%%

#======================================================================================================================


#==================================================================
# SHAP Plots for Best Models
#==================================================================

# set year to analyze


# set the model version
model_type ='full_model'
set_pos = 'Defense'

run_params = {
    
    # set year and week to analyze
    'set_year': 2022,
    'set_week': 16,

    # set beginning of validation period
    'val_year_min': 2020,
    'val_week_min': 14,

    # opt params
    'n_iters': 25,
    'n_splits': 5,

    # other parameters
    'use_sample_weight': False,
    'opt_type': 'custom_rand',
    'cuts': [33, 80, 95],
    'met': 'y_act',

    # set number of weeks back to begin validation
    'back_weeks': {
        'QB': 28,
        'RB': 32,
        'WR': 24,
        'TE': 28,
        'Defense': 28
    },

    'rush_pass': ''
}


#==========
# Pull and clean compiled data
#==========

df, run_params = load_data(model_type, set_pos, run_params)
df, run_params = create_game_date(df, run_params)
df['y_act'] = df.y_act# 1000*df.y_act / df.dk_salary

df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

cut = 95
df_train_class, df_predict_class = get_class_data(df, cut, run_params)

skm = SciKitModel(df_train, model_obj='reg', sera_wt=1, r2_wt=1)
X_all, y = skm.Xy_split('y_act', run_params['drop_cols'])

X = X_all.sample(frac=1, axis=1)
if 'game_date' not in X.columns:
    X['game_date'] = X_all.game_date

#------------
# Get Regression Data
#------------

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LassoLars,HuberRegressor, QuantileRegressor
from sklearn_quantile import RandomForestQuantileRegressor, KNeighborsQuantileRegressor, SampleRandomForestQuantileRegressor
from category_encoders.cat_boost import CatBoostEncoder

pipe = skm.model_pipe([ #('cbe', CatBoostEncoder()),
                        skm.piece('random_sample'),
                        skm.piece('std_scale'), 
                         skm.piece('select_perc'),
                        skm.feature_union([
                                        skm.piece('agglomeration'), 
                                        skm.piece('k_best'),
                                        skm.piece('pca')
                                        ]),
                        skm.piece('k_best'),
                        skm.piece('enet')
                        
                     ])

# pipe.steps[-1][-1].quantile = 0.95
# pipe.steps[-1][-1].q = 0.8
# set params
params = skm.default_params(pipe, 'rand')


# params = {'random_sample__frac': np.array([0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45,
#         0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67,
#         0.69]),
#  'random_sample__seed': range(0, 10000, 1000),
#  'select_perc__percentile': range(20, 61, 5),
#  'feature_union__agglomeration__n_clusters': range(2, 25, 2),
#  'feature_union__k_best__k': range(20, 125, 5),
#  'feature_union__pca__n_components': range(2, 20, 2),
#  'k_best__k': range(20, 125, 5),
#  'knn__n_neighbors': range(20, 80),
#  'knn__weights': [ 'uniform'],
#  'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}


# run the model with parameter search
best_models, oof_data, param_scores = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                        col_split='game_date',n_splits=5,
                                                        time_split=run_params['cv_time_input'],
                                                        bayes_rand='custom_rand',
                                                        sample_weight=False,
                                                        random_seed=12345)


mf.show_scatter_plot(oof_data['full_hold']['pred'], oof_data['full_hold']['y_act'])

#%%
i=2
best_models[i].fit(X,y)
skm.print_coef(best_models[i])

#%%

df = param_scores.copy()

df['input_features'] = df.feature_union__k_best__k + df.feature_union__agglomeration__n_clusters
df.loc[df.k_best__k < df.input_features, 'k_best__k'] = df.loc[df.k_best__k < df.input_features, 'input_features']
df.select_perc__percentile = df.select_perc__percentile.fillna(100)

X = df.drop('scores', axis=1)

for c in X.dtypes[X.dtypes=='object'].index:
    X = pd.concat([X, pd.get_dummies(X)], axis=1).drop(c, axis=1)
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
        all_coef[abs(all_coef) > 0.005].sort_values().plot.barh(figsize=(10,18))
    except:
        shap.summary_plot(all_coef.values, X_all, feature_names=X_all.columns, plot_size=(18,10), max_display=40, show=False)

def entry_optimize_params(df, max_adjust, model_name):

    adjust_winnings = df.groupby(['trial_num']).agg(max_lineup_num=('lineup_num', 'max')).reset_index()
    adjust_winnings.max_lineup_num = 30 / (adjust_winnings.max_lineup_num+1)
    
    df = pd.merge(df, adjust_winnings, on='trial_num')
    df.winnings = df.winnings / df.max_lineup_num
    df.loc[df.winnings >= max_adjust, 'winnings'] = max_adjust
    
    df.loc[df['max_lineup_num']==1, ['player_drop_multiple']] = 0

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
                     SELECT week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, trial_num, repeat_num
                      FROM Entry_Optimize_Results
                      ) USING (week, year, trial_num, repeat_num)
                WHERE trial_num >= 269
                      AND pred_vers = 'sera0_rsq0_mse1_brier1_matt1_bayes'
                      AND week < 17
                      AND week != 8
                    --  AND week != 14
                     -- AND reg_ens_vers IN ('random_kbest_sera0_rsq0_mse1_include2_kfold3', 'random_sera0_rsq0_mse1_include2_kfold3')
                     --AND reg_ens_vers='random_full_stack_sera0_rsq0_mse1_include2_kfold3'
                  --  AND million_ens_vers='kbest_matt0_brier1_include2_kfold3'
                ''', 'Results')

model_type = {
 'enet': ElasticNet(alpha=1, l1_ratio=0.1),
 'lasso': Lasso(alpha=0.1),
 'ridge': Ridge(alpha=100),
 'rf': RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=10, n_jobs=-1),
 'lgbm': LGBMRegressor(n_estimators=50, max_depth=5, min_samples_leaf=5, n_jobs=-1)

}
w=1
model_name='lgbm'
m = model_type[model_name] 
X, y = entry_optimize_params(df, max_adjust=5000, model_name=model_name)
coef_vals, X = get_model_coef(X, y, m)
show_coef(coef_vals, X)

#%%

weeks = [1, 2, 3, 4, 5, 6, 7, #8, 
         9, 10, 11, 12, 13, 14, 15, 16,
         1]
years = [2022, 2022, 2022, 2022, 2022, 2022, 2022,# 2022, 
         2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 
         2023]

i=0
all_coef = None; X_all = None
for w, yr in zip(weeks, years):
    df = dm.read(f'''SELECT *  
                     FROM Entry_Optimize_Params_Detail 
                     JOIN (
                            SELECT week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, trial_num, repeat_num
                            FROM Entry_Optimize_Results          
                          ) USING (week, year, trial_num, repeat_num)
                     WHERE trial_num >= 269
                           AND pred_vers = 'sera0_rsq0_mse1_brier1_matt1_bayes'
                           --AND reg_ens_vers IN ('random_kbest_sera0_rsq0_mse1_include2_kfold3', 'random_sera0_rsq0_mse1_include2_kfold3')
                           --AND reg_ens_vers='random_full_stack_sera0_rsq0_mse1_include2_kfold3'          
                           --AND million_ens_vers IN ('random_matt0_brier1_include2_kfold3', 'random_kbest_matt0_brier1_include2_kfold3')
                           AND week = {w}
                           AND year = {yr}
                     ''', 'Results')

    model_name = 'enet'
    m = model_type[model_name]
    X, y = entry_optimize_params(df, max_adjust=5000, model_name=model_name)
    coef_vals, X = get_model_coef(X, y, m)
    all_coef, X_all = join_coef(i, all_coef, coef_vals, X_all, X, model_name); i+=1

show_coef(all_coef, X_all)

 # %%

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

set_week = 16
set_year = 2022
pos = 'QB'
# pred_vers = 'sera1_rsq0_brier1_matt1_lowsample_perc'
pred_vers = 'sera1_rsq0_brier1_matt1_lowsample_perc_ffa_fc'

# ens_vers = 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3'
ens_vers = 'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3_fullstack'

std_dev_type = 'pred_spline_class80_q80_matt1_brier1_kfold3'
model_type = 'backfill'

skm = SciKitModel(pd.DataFrame({'x': [1, 2]}))
df = dm.read(f'''SELECT * 
                 FROM Model_Validations 
                 WHERE model_type='{model_type}'
                       AND pred_version='{pred_vers}'
                       AND ensemble_vers='{ens_vers}'
                       AND std_dev_type='{std_dev_type}'
                       AND pos='{pos}' 
                       AND set_year={set_year}
                       AND set_week={set_week}''', 'Simulation')

print(pred_vers, '\n', ens_vers)
metrics = skm.test_scores(df.y_act, df.pred_fp_per_game)

df['error'] = df.y_act - df.pred_fp_per_game
# df.sort_values(by='error').iloc[:50]


#%%

actual_pts = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
    if pos=='Defense': pl = 'defTeam'
    else: pl = 'player'

    actual_pts_cur = dm.read(f'''SELECT {pl} player, week, season year, fantasy_pts y_act
                                    FROM {pos}_Stats 
                                    WHERE week>=1
                                        and week < 18
                                        and season >= 2020
                                        ''', 'FastR')
    actual_pts = pd.concat([actual_pts, actual_pts_cur], axis=0)

if len(actual_pts) > 0:
    df = pd.merge(df, actual_pts, on=['player', 'week', 'year'], how='left')

#%%
from sklearn.metrics import r2_score, mean_squared_error
skm = SciKitModel(pd.DataFrame({'x': [1, 2]}))
df = dm.read(f'''SELECT * 
                FROM Model_Validations 
                WHERE model_type='full_model' 
                      AND set_year=2022
                      AND set_week <= 15
                      AND (pred_version LIKE '%ffa%' 
                           OR pred_version LIKE '%bayes%')
             ''', 'Validations')

df = df.drop('y_act', axis=1)
df = pd.merge(df, actual_pts, on=['player', 'week', 'year'], how='left').dropna().reset_index(drop=True)
gcols = ['set_week', 'set_year', 'pos', 'pred_version', 'ensemble_vers', ]
df = df.groupby(gcols).apply(lambda x: mean_squared_error(x['y_act'], x['pred_fp_per_game'])).reset_index()
display(df.sort_values(by=['set_year', 'set_week', 'pos', 0],
               ascending=[True, True, False, False]).iloc[:50])

#%%

df.groupby(['pos', 'pred_version', 'ensemble_vers']).agg({0: 'mean'}).sort_values(by=['pos', 0],
               ascending=[False, True])

#%%

skm = SciKitModel(pd.DataFrame({'x': [1, 2]}))
df = dm.read(f'''SELECT * 
                FROM Model_Test_Validations 
                WHERE model_type='full_model' 
                      AND set_year=2022
                      AND (pred_version LIKE '%ffa%' 
                           OR pred_version LIKE '%bayes%')
             ''', 'Validations')

gcols = ['set_week', 'set_year', 'pos', 'pred_version', 'ensemble_vers', ]
df = df.groupby(gcols).apply(lambda x:r2_score(x['actual_pts'], x['pred_fp_per_game'])).reset_index()
display(df.sort_values(by=['set_year', 'set_week', 'pos', 0],
               ascending=[True, True, False, False]).iloc[:50])

#%%

df = dm.read('''SELECT * 
                FROM Model_Validations 
                JOIN (SELECT player, salary, year, league week FROM Salaries) USING (player, week, year)
                WHERE set_year=2022''', 'Simulation')

df = df[(df.pred_version=='fixed_model_clone_proba_sera_brier_lowsample_perc') & \
    (df.ensemble_vers=='no_weight_yes_kbest_randsample_sera_include2') & \
        (df.model_type=='backfill') & \
            (df.set_week==1) & \
                (df.set_year==2022) & \
                    (df.pos=='TE')]

df['y_act'] = df.y_act
df['pred_fp_per_game'] = df.pred_fp_per_game
df.corr()['y_act']

#%%

#======================================================================================================================

#==================================================================
# Look at Hyperparameter Optimization
#==================================================================



reg_or_class = 'reg'
model_type = 'lgbm'

df = dm.read(f'''SELECT * 
                 FROM {reg_or_class}_{model_type}
                 WHERE scores > 0
                       AND pos='WR'
                       and week=1
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

m = ElasticNet(alpha=0.01, l1_ratio=0.05)
# m = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2)
# m = LGBMRegressor(n_estimators=50, max_depth=5, min_samples_leaf=1)

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
