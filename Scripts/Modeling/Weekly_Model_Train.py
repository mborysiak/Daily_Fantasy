#%%
# core packages
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt
import matplotlib.pyplot as plt

from ff.db_operations import DataManage
from ff import general as ffgeneral
from skmodel import SciKitModel
import zModel_Functions as mf

import pandas_bokeh
pandas_bokeh.output_notebook()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

pd.set_option('display.max_columns', 999)
# from sklearn import set_config
# set_config(display='diagram')

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

run_weeks = [6]

run_params = {
    
    # set year and week to analyze
    'set_year': 2022,

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

    'rush_pass': ''
}

# set position and model type
set_pos = 'RB'
model_type = 'full_model'

# set weights for running model
r2_wt = 0
sera_wt = 1
matt_wt = 1
brier_wt = 2

use_calibrate = True

# set version and iterations
vers = 'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate'

#----------------
# Data Loading
#----------------

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
    

#----------------
# Modeling Functions
#----------------

def output_dict():
    return {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}


def update_output_dict(label, m, suffix, out_dict, oof_data, best_models):

    # append all of the metric outputs
    lbl = f'{label}_{m}{suffix}'
    out_dict['pred'][lbl] = oof_data['hold']
    out_dict['actual'][lbl] = oof_data['actual']
    out_dict['scores'][lbl] = oof_data['scores']
    out_dict['models'][lbl] = best_models
    out_dict['full_hold'][lbl] = oof_data['full_hold']

    return out_dict


def get_skm(skm_df, model_obj, to_drop):
    
    skm_options = {
        'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt),
        'class': SciKitModel(skm_df, model_obj='class', brier_wt=brier_wt, matt_wt=matt_wt),
        'quantile': SciKitModel(skm_df, model_obj='quantile')
    }
    
    skm = skm_options[model_obj]
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10):

    if m == 'adp':
        
        
        # set up the ADP model pipe
        pipe = skm.model_pipe([skm.piece('feature_select'), 
                               skm.piece('std_scale'), 
                               skm.piece('k_best'),
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
                                skm.piece('select_perc'),
                                skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best'),
                                                skm.piece('pca')
                                                ]),
                                skm.piece('k_best'),
                                skm.piece(m)])

    elif skm.model_obj == 'class':
        pipe = skm.model_pipe([skm.piece('random_sample'),
                               skm.piece('std_scale'), 
                               skm.piece('select_perc_c'),
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
        if m == 'qr_q': pipe.steps[-1][-1].quantile = alpha
        else: pipe.steps[-1][-1].alpha = alpha


    # get the params for the current pipe and adjust if needed
    params = skm.default_params(pipe, 'rand')
    if m=='adp': 
        params['feature_select__cols'] = [
                                            ['game_date', 'year', 'week', 'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints'],
                                            ['year', 'week',  'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints'],
                                            [ 'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints']
                                        ]
        params['k_best__k'] = range(1, 9)
    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)
    if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)
    if stack_model: params['k_best__k'] = range(2, 40)

    return pipe, params


def get_model_output(model_name, cur_df, model_obj, out_dict, run_params, i, min_samples=10, alpha=''):

    print(f'\n{model_name}\n============\n')

    skm, X, y = get_skm(cur_df, model_obj, to_drop=run_params['drop_cols'])
    pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples)

    if model_obj == 'class': 
        proba = True
        alpha = f'_{cut}' 
        calibrate=use_calibrate
    else: 
        proba = False
        calibrate=False

    # fit and append the ADP model
    best_models, oof_data, param_scores = skm.time_series_cv(pipe, X, y, params, n_iter=run_params['n_iters'], n_splits=run_params['n_splits'],
                                                             col_split='game_date', time_split=run_params['cv_time_input'],
                                                             bayes_rand=run_params['opt_type'], proba=proba, calibrate=calibrate,
                                                             random_seed=(i+7)*19+(i*12)+6, alpha=alpha)
    
    out_dict = update_output_dict(model_obj, model_name, str(alpha), out_dict, oof_data, best_models)
    # db_output = add_result_db_output('reg', m, oof_data['scores'], db_output, run_params)
    try: save_param_scores(param_scores, model_obj, model_name, run_params)
    except: print(f'Param save for {model_name} failed')

    return out_dict, best_models, oof_data


#-----------------
# Saving Data / Handling
#-----------------


def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def add_result_db_output(model_type, model, results, db_output, run_params):
    db_output['pkey'].append(pkey)
    db_output['set_pos'].append(set_pos)
    db_output['set_year'].append(run_params['set_year'])
    db_output['set_week'].append(run_params['set_week'])
    db_output['model_type'].append(model_type)
    db_output['model'].append(model)
    db_output['validation_score'].append(results[0])
    db_output['test_score'].append(results[1])

    return db_output

def save_param_scores(df, obj_type, model, run_params):
    for c in df.columns:
        if 'class_weight' in c:
            df[c] = df[c].apply(lambda x: x[0])

    df = df.assign(model=model, year=run_params['set_year'], week=run_params['set_week'], pos=set_pos, model_type=model_type)
    exist = dm.read(f"SELECT * FROM {obj_type}_{model}", 'Results')
    df = pd.concat([exist, df], axis=0, sort=False)

    dm.write_to_db(df, 'Results', f'{obj_type}_{model}', 'replace')

def save_output_dict(out_dict, model_output_path, label):

    save_pickle(out_dict['pred'], model_output_path, f'{label}_pred')
    save_pickle(out_dict['actual'], model_output_path, f'{label}_actual')
    save_pickle(out_dict['models'], model_output_path, f'{label}_models')
    save_pickle(out_dict['scores'], model_output_path, f'{label}_scores')
    save_pickle(out_dict['full_hold'], model_output_path, f'{label}_full_hold')

#====================
# Stacking Functions
#====================

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
    output['dk_salary_rank'] = range(len(output))
    output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

    return output



#%%
run_list = [
            # ['QB', '', 'full_model'],
            # ['RB', '', 'full_model'],
            # ['WR', '', 'full_model'],
            # ['TE', '', 'full_model'],
            # ['Defense', '', 'full_model'],
            # ['QB', '', 'backfill'],
            # ['RB', '', 'backfill'],
            ['WR', '', 'backfill'],
            ['TE', '', 'backfill'],
]

for w in run_weeks:
    run_params['set_week'] = w

    for set_pos, rush_pass, model_type in run_list:

        run_params['rush_pass'] = rush_pass

        print(f"\n==================\n{set_pos} {model_type} {run_params['set_year']} {run_params['set_week']} {vers}\n====================")

        #==========
        # Pull and clean compiled data
        #==========

        # load data and filter down
        pkey, db_output, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
        df, run_params = load_data(model_type, set_pos, run_params)
        df, run_params = create_game_date(df, run_params)
        df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

        #=========
        # Run Models
        #=========


        # set up blank dictionaries for all metrics
        out_reg, out_class, out_quant = output_dict(), output_dict(), output_dict()

        # run all other models
        model_list = ['adp', 'huber', 'lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'gbmh', 'rf']
        for i, m in enumerate(model_list):
            out_reg, _, _ = get_model_output(m, df_train, 'reg', out_reg, run_params, i, min_samples)
        save_output_dict(out_reg, model_output_path, 'reg')

        # run all other models
        model_list = ['lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c', 'gbmh_c']
        for cut in run_params['cuts']:
            print(f"\n--------------\nPercentile {cut}\n--------------\n")
            df_train_class, df_predict_class = get_class_data(df, cut, run_params)    
            for i, m in enumerate(model_list):
                out_class, _, _= get_model_output(m, df_train_class, 'class', out_class, run_params, i, min_samples)
        save_output_dict(out_class, model_output_path, 'class')

        # # run all other models
        # model_list = ['gbm_q', 'lgbm_q', 'qr_q']
        # for i, m in enumerate(model_list):
        #     for alph in [0.8, 0.95]:
        #         out_quant, _, _ = get_model_output(m, df_train, 'quantile', out_quant, run_params, i, alpha=alph)
        # save_output_dict(out_quant, model_output_path, 'quant')


#%%

# db_output = add_result_db_output('final', final_m, 
#                                 [stack_scores['adp_score'], stack_scores['stack_score']], 
#                                 db_output)

# # write out tracking results to tracking DB
# db_output_pd = pd.DataFrame(db_output)
# db_output_pd = db_output_pd.assign(perc_low=prc[set_pos][0]).assign(perc_high=prc[set_pos][1]).assign(perc_range=prc[set_pos][2])
# db_output_pd = db_output_pd.assign(kb_low=kbs[set_pos][0]).assign(kb_high=kbs[set_pos][1]).assign(kb_range=kbs[set_pos][2])
# db_output_pd = db_output_pd.assign(n_iters=n_iters).assign(to_keep=to_keep).assign(val_week=val_week_min).assign(val_year=val_year_min)
# dm.delete_from_db('Results', 'Model_Tracking',f"pkey='{pkey}'")
# dm.write_to_db(db_output_pd, 'Results', 'Model_Tracking', 'append')
