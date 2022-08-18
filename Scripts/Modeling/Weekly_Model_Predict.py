#%%
# core packages
from random import Random, sample
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt
import matplotlib.pyplot as plt
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from skmodel import SciKitModel
from Fix_Standard_Dev import *
from sklearn.metrics import r2_score
import zModel_Functions as mf

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

import pandas_bokeh
pandas_bokeh.output_notebook()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

pd.set_option('display.max_columns', 999)

from sklearn import set_config
set_config(display='diagram')

#====================
# Data Loading Functions
#====================

def create_pkey_output_path(set_pos, run_params, model_type, vers):

    pkey = f"{set_pos}_year{run_params['set_year']}_week{run_params['set_week']}_{model_type}{vers}{run_params['rush_pass']}"
    db_output = {
                'pkey': [], 
                'set_pos': [], 
                'set_year': [], 
                'set_week':[], 
                'model_type':[],
                'model': [],
                'validation_score': [],
                'test_score': []
                }

    model_output_path = f"{root_path}/Model_Outputs/{run_params['set_year']}/{pkey}/"
    if not os.path.exists(model_output_path): os.makedirs(model_output_path)
    
    return pkey, db_output, model_output_path

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

def create_game_date(df, run_params):
    
    # set up the date column for sorting
    df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
    cv_time_input = int(dt.datetime(run_params['val_year_min'], 1, run_params['val_week_min']).strftime('%Y%m%d'))
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


#====================
# Stacking Functions
#====================


def get_skm(skm_df, model_obj, to_drop):
    
    skm = SciKitModel(skm_df, model_obj=model_obj)
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10):

    if m == 'adp':
        
        # set up the ADP model pipe
        pipe = skm.model_pipe([skm.piece('feature_select'), 
                               skm.piece('std_scale'), 
                               skm.piece('lr')])

    elif stack_model:
        pipe = skm.model_pipe([
                            skm.piece('std_scale'),
                            skm.piece('k_best'), 
                            skm.piece(m)
                        ])

    elif skm.model_obj == 'reg':
        pipe = skm.model_pipe([skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best'),
                                              #  skm.piece('pca')
                                                ]),
                                skm.piece('k_best'),
                                skm.piece(m)])

    elif skm.model_obj == 'class':
        pipe = skm.model_pipe([skm.piece('random_sample'),
                               skm.piece('std_scale'), 
                               skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best_c'),
                                                ]),
                               skm.piece('k_best_c'),
                               skm.piece(m)])
    
    

    elif skm.model_obj == 'quantile':
        pipe = skm.model_pipe([
                                skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.piece('k_best'), 
                                skm.piece(m)
                                ])
        pipe.steps[-1][-1].alpha = alpha


    # get the params for the current pipe and adjust if needed
    params = skm.default_params(pipe, 'rand')
    if m=='adp': params['feature_select__cols'] = [
                                                    ['game_date', 'ProjPts', 'dk_salary', 'fantasyPoints', 'year', 'week'],
                                                    ['ProjPts', 'dk_salary', 'fantasyPoints'],
                                                ]
    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)
    if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)
    if stack_model: params['k_best__k'] = range(2, 40)

    return pipe, params

def load_all_stack_pred(model_output_path):

    # load the regregression predictions
    pred, actual, models_reg, _, full_hold_reg = mf.load_all_pickles(model_output_path, 'reg')
    X_stack, y_stack = mf.X_y_stack('reg', full_hold_reg, pred, actual)

    # load the class predictions
    pred_class, actual_class, models_class, _, full_hold_class = mf.load_all_pickles(model_output_path, 'class')
    X_stack_class, _ = mf.X_y_stack('class', full_hold_class, pred_class, actual_class)

    # load the quantile predictions
    pred_quant, actual_quant, models_quant, _, full_hold_quant = mf.load_all_pickles(model_output_path, 'quant')
    X_stack_quant, _ = mf.X_y_stack('quant', full_hold_quant, pred_quant, actual_quant)

    # concat all the predictions together
    X_stack = pd.concat([X_stack, X_stack_quant, X_stack_class], axis=1)

    return X_stack, y_stack, models_reg, models_class, models_quant


def run_stack_models(final_m, i, X_stack, y_stack, best_models, scores, stack_val_pred, show_plots=True):

    print(f'\n{final_m}')

    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), 'reg', to_drop=[])
    pipe, params = get_full_pipe(skm, final_m, stack_model=True)

    best_model, stack_scores, stack_pred = skm.best_stack(pipe, params,
                                                          X_stack, y_stack, n_iter=100, 
                                                          run_adp=True, print_coef=True,
                                                          sample_weight=False, random_state=(i*12)+(i*17))
    best_models.append(best_model)
    scores.append(stack_scores['stack_score'])
    stack_val_pred = pd.concat([stack_val_pred, pd.Series(stack_pred['stack_pred'], name=final_m)], axis=1)

    if show_plots:
        mf.show_scatter_plot(stack_pred['stack_pred'], stack_pred['y'], r2=True)
        mf.top_predictions(stack_pred['stack_pred'], stack_pred['y'], r2=True)

    return best_models, scores, stack_val_pred
    
def create_stack_predict(df_predict, models, X, y, proba=False):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
        
        predictions = pd.DataFrame()
        for m in ind_models:

            m.fit(X, y)
            if proba: cur_predict = m.predict_proba(df_predict[X.columns])[:,1]
            else: cur_predict = m.predict(df_predict[X.columns])
            predictions = pd.concat([predictions, pd.Series(cur_predict)], axis=1)
            
        predictions = predictions.mean(axis=1)
        predictions = pd.Series(predictions, name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict

def get_stack_predict_data(df_train, df_predict, df, run_params, 
                           models_reg, models_class, models_quant):

    _, X, y = get_skm(df_train, 'reg', to_drop=run_params['drop_cols'])
    X_predict = create_stack_predict(df_predict, models_reg, X, y)
    X_predict_quant = create_stack_predict(df_predict, models_quant, X, y)
    X_predict = pd.concat([X_predict, X_predict_quant], axis=1)

    for cut in run_params['cuts']:
        df_train_class, df_predict_class = get_class_data(df, cut, run_params)
        _, X, y = get_skm(df_train_class, 'class', to_drop=run_params['drop_cols'])
        X_predict_class = create_stack_predict(df_predict_class, models_class, X, y, proba=True)
        X_predict_class = X_predict_class[[c for c in X_predict_class.columns if str(cut) in c]]
        X_predict = pd.concat([X_predict, X_predict_class], axis=1)

    return X_predict


def average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, show_plot=True):
    
    skm_stack, _, _ = get_skm(df_train, 'reg', to_drop=[])
    best_val, best_predictions = mf.best_average_models(skm_stack, scores, final_models, y_stack, stack_val_pred, predictions)
    
    if show_plot:
        mf.show_scatter_plot(best_val.mean(axis=1), y_stack, r2=True)
    return best_val, best_predictions


def create_output(output_start, predictions):

    output = output_start.copy()
    output['pred_fp_per_game'] = predictions.mean(axis=1)
    output = output.sort_values(by='dk_salary', ascending=False)
    output['dk_rank'] = range(len(output))
    output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

    return output


def validation_compare_df(model_output_path, best_val):

    _, _, _, _, oof_data = mf.load_all_pickles(model_output_path, 'reg')
    oof_data = oof_data['reg_adp'][['player', 'team', 'year', 'y_act']].reset_index(drop=True)
    best_val = pd.Series(best_val.mean(axis=1), name='prediction')
    val_compare = pd.concat([oof_data, best_val], axis=1).rename(columns={'year': 'season'})
    
    return val_compare


def std_dev_features(cur_df, model_name, run_params, show_plot=True):

    skm, X, y = get_skm(cur_df, 'reg', to_drop=run_params['drop_cols'])
    pipe, params = get_full_pipe(skm, model_name, stack_model=True)

    # fit and append the ADP model
    best_models, _, _ = skm.time_series_cv(pipe, X, y, params, n_iter=run_params['n_iters'], n_splits=run_params['n_splits'],
                                           col_split='game_date', time_split=run_params['cv_time_input'],
                                           bayes_rand='custom_rand', random_seed=1234)

    for bm in best_models: bm.fit(X, y)
    if show_plot:
        mf.shap_plot(best_models, X, model_num=0)
        plt.show()

    return best_models, X


def add_std_dev_max(df_train, df_predict, output, set_pos, num_cols=5):

    std_dev_models, X = std_dev_features(df_train, 'enet', set_pos, show_plot=True)
    sd_cols, df_train, df_predict = mf.get_sd_cols(df_train, df_predict, X, std_dev_models, num_cols=num_cols)
    sd_spline, max_spline, min_spline = get_std_splines(df_train, sd_cols, show_plot=True, k=1, 
                                                        min_grps_den=int(df_train.shape[0]*0.3), 
                                                        max_grps_den=int(df_train.shape[0]*0.1))

    output = assign_sd_max(output, df_predict, df_train, sd_cols, sd_spline, max_spline, min_spline)

    return output


def assign_sd_max(output, df_predict, sd_df, sd_cols, sd_spline, max_spline, min_spline):
    
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(sd_df[list(sd_cols.keys())])

    df_predict = df_predict.set_index('player')
    df_predict = df_predict.reindex(index=output['player'])
    df_predict = df_predict.reset_index()

    pred_sd_max = pd.DataFrame(sc.transform(df_predict[list(sd_cols.keys())])) * list(sd_cols.values())
    pred_sd_max = pred_sd_max.mean(axis=1)

    output['std_dev'] = sd_spline(pred_sd_max)
    output['max_score'] = max_spline(pred_sd_max)
    output['min_score'] = min_spline(pred_sd_max)
    
    return output


def add_actual(df):
    if set_pos=='Defense': pl = 'defTeam'
    else: pl = 'player'
    actual_pts = dm.read(f'''SELECT {pl} player, fantasy_pts actual_pts
                            FROM {set_pos}_Stats 
                            WHERE week={run_params['set_week']} 
                                and season={run_params['set_year']}''', 'FastR')
    df = pd.merge(df, actual_pts, on='player')
    return df


def save_output_to_db(output, run_params):

    output['pos'] = set_pos
    output['version'] = vers
    output['ensemble_vers'] = ensemble_vers
    output['std_dev_type'] = std_dev_type
    output['model_type'] = model_type
    output['week'] = run_params['set_week']
    output['year'] = run_params['set_year']
    output['min_score'] = 0

    output = output[['player', 'dk_salary', 'pred_fp_per_game', 'std_dev',
                        'dk_rank', 'pos', 'version', 'model_type', 'max_score', 'min_score',
                        'week', 'year', 'ensemble_vers', 'std_dev_type']]

    del_str = f'''pos='{set_pos}' 
                AND version='{vers}'
                AND ensemble_vers='{ensemble_vers}' 
                AND std_dev_type='{std_dev_type}'
                AND week={run_params['set_week']} 
                AND year={run_params['set_year']}
                AND model_type='{model_type}'
                '''
    dm.delete_from_db('Simulation', 'Model_Predictions', del_str)
    dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')

#%%
#==========
# General Setting
#==========

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#---------------
# Settings
#---------------

run_params = {
    
    # set year and week to analyze
    'set_year': 2021,
    'set_week': 12,

    # set beginning of validation period
    'val_year_min': 2020,
    'val_week_min': 10,

    # opt params
    'n_iters': 25,
    'n_splits': 5,

    # other parameters
    'use_sample_weight': False,
    'opt_type': 'custom_rand',
    'cuts': [33, 80, 95],
    'met': 'y_act',

    'rush_pass': ''
}

# set position and model type
set_pos = 'QB'
model_type = 'backfill'
std_dev_type = 'spline_enet_coef'
ensemble_vers = 'no_weight_yes_kbest'

# set version and iterations
vers = 'fixed_model_clone'

for set_pos in ['QB', 'RB', 'WR', 'TE']:

    # load data and filter down
    pkey, db_output, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
    df, run_params = load_data(model_type, set_pos, run_params)
    df, run_params = create_game_date(df, run_params)
    df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

    #------------
    # Run the Stacking Models and Generate Output
    #------------

    # get the training data for stacking and prediction data after stacking
    X_stack, y_stack, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path)
    X_predict = get_stack_predict_data(df_train, df_predict, df, run_params, 
                                    models_reg, models_class, models_quant)

    # create the stacking models
    final_models = ['ridge', 'lasso', 'lgbm', 'xgb', 'rf', 'bridge', 'gbm']
    stack_val_pred = pd.DataFrame(); scores = []; best_models = []
    for i, fm in enumerate(final_models):
        best_models, scores, stack_val_pred = run_stack_models(fm, i, X_stack, y_stack, best_models, scores, stack_val_pred, show_plots=True)

    # get the best stack predictions and average
    predictions = mf.stack_predictions(X_predict, best_models, final_models)
    best_val, best_predictions = average_stack_models(df_train, scores, final_models, y_stack, stack_val_pred, predictions, show_plot=True)

    # create the output and add standard devations / max scores
    output = create_output(output_start, best_predictions)
    output = add_std_dev_max(df_train, df_predict, output, run_params, num_cols=10)

    try:  
        output = add_actual(output)
        display(output.loc[:50, ['player', 'dk_salary','dk_rank', 'pred_fp_per_game', 'actual_pts', 'std_dev', 'max_score']])
        mf.show_scatter_plot(output.pred_fp_per_game, output.actual_pts)
        output = output.drop('actual_pts', axis=1)
    except:
        display(output.loc[:50, ['player', 'dk_salary','dk_rank', 'pred_fp_per_game', 'std_dev', 'max_score']])

    save_output_to_db(output, run_params)

#%%

# # save out final results


#%%
# both_compare = validation_compare_df(model_output_path, best_val)
# rush_val_compare = validation_compare_df(model_output_path, best_val)
# pass_val_compare = validation_compare_df(model_output_path, best_val)
# pass_val_compare = pass_val_compare.rename(columns={'prediction': 'pass_prediction', 'y_act': 'pass_y_act'})
# rush_val_compare = rush_val_compare.rename(columns={'prediction': 'rush_prediction', 'y_act': 'rush_y_act'})

# skm, _, _ = get_skm(df_train, 'reg', [])
# from sklearn.metrics import mean_squared_error

# rp = pd.concat([both_compare, 
#                 pass_val_compare[['pass_prediction', 'pass_y_act']],
#                 rush_val_compare[['rush_prediction', 'rush_y_act']]], axis=1)
# rp['rp_pred'] = rp.pass_prediction + rp.rush_prediction
# rp['rp_y_act'] = rp.pass_y_act + rp.rush_y_act

# rp = rp[(rp.rp_pred < 29) & (rp.prediction > 14)].reset_index(drop=True)

# mf.show_scatter_plot(rp.rp_pred, rp.y_act, r2=True)
# mf.show_scatter_plot(rp.prediction, rp.y_act, r2=True)


# print('\nRP MSE:', np.sqrt(mean_squared_error(rp.rp_pred, rp.y_act)))
# print('Both MSE:', np.sqrt(mean_squared_error(rp.prediction, rp.y_act)))
# print('\nRP Sera:', skm.sera_loss(rp.y_act, rp.rp_pred))
# print('Both Sera:', skm.sera_loss(rp.y_act, rp.prediction), '\n')
# print(rp[abs(rp.y_act - rp.rp_y_act) > 0.001])

# %%

def plot_distribution(estimates, player_name):

    from IPython.core.pylabtools import figsize
    import seaborn as sns
    import matplotlib.pyplot as plt

    print('\n', player_name)

    # Plot all the estimates
    plt.figure(figsize(8, 8))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws = {'linewidth' : 4},
                 label = 'Estimated Dist.')

    # Plot the mean estimate
    plt.vlines(x = estimates.mean(), ymin = 0, ymax = 0.01, 
                linestyles = '--', colors = 'red',
                label = 'Pred Estimate',
                linewidth = 2.5)

    plt.legend(loc = 1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel('Grade'); plt.ylabel('Density');

    # Prediction information
    sum_stats = (np.percentile(estimates, 5), np.percentile(estimates, 95), estimates.std() /estimates.mean())
    print('Average Estimate = %0.4f' % estimates.mean())
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f    Std Error = %0.4f' % sum_stats)

def trunc_normal(player_data, num_samples=1000):

    import scipy.stats as stats

    # create truncated distribution
    lower, upper = player_data.min_score,  player_data.max_score
    lower_bound = (lower - player_data.pred_fp_per_game) / player_data.std_dev, 
    upper_bound = (upper - player_data.pred_fp_per_game) / player_data.std_dev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc= player_data.pred_fp_per_game, scale= player_data.std_dev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


def trunc_normal_dist(self, num_options=500):
    predictions = pd.DataFrame()
    for _, row in self.player_data.iterrows():
        dists = pd.DataFrame(self.trunc_normal(row, num_options)).T
        predictions = pd.concat([predictions, dists], axis=0)
    
    return predictions.reset_index(drop=True)

output['min_score'] = 8
cur_player = output.iloc[0]
estimates= trunc_normal(cur_player)
plot_distribution(estimates, cur_player.player)
# %%
