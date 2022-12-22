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
from joblib import Parallel, delayed

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

run_weeks = [16]

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

    # set number of weeks back to begin validation
    'back_weeks': {
        'QB': 32,
        'RB': 28,
        'WR': 28,
        'TE': 32,
        'Defense': 32
    },

    'rush_pass': ''
}

n_splits = run_params['n_splits']

# set position and model type
set_pos = 'RB'
model_type = 'full_model'

# set weights for running model
r2_wt = 0
sera_wt = 1
matt_wt = 1
brier_wt = 1

# set version and iterations
vers = 'sera1_rsq0_brier1_matt1_lowsample_perc_ffa_fc'

#----------------
# Data Loading
#----------------

def create_pkey_output_path(set_pos, run_params, model_type, vers):

    pkey = f"{set_pos}_year{run_params['set_year']}_week{run_params['set_week']}_{model_type}{vers}"
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
    
    df = df.dropna(axis=1)
    drop_cols = list(df.dtypes[df.dtypes=='object'].index)
    run_params['drop_cols'] = drop_cols
    print(drop_cols)

    return df, run_params

def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))


def get_cv_time_input(df, back_weeks):
    df = df[(df.year < run_params['set_year']) | \
            (
              (df.year==run_params['set_year']) & \
              (df.week<=run_params['set_week'])
            )]
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
    cv_time_input = np.max([cv_time_input, 20200114])
    print(f'Begin Validation Using {cv_time_input}')
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
        elif m in ('rf_q', 'knn_q'): pipe.steps[-1][-1].q = alpha
        else: pipe.steps[-1][-1].alpha = alpha
    


    # get the params for the current pipe and adjust if needed
    params = skm.default_params(pipe, 'rand')
    if m=='adp': 
        params['feature_select__cols'] = [
                                            ['game_date', 'year', 'week', 'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
                                            ['year', 'week',  'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
                                            [ 'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank']
                                        ]
        params['k_best__k'] = range(1, 14)
    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)
    if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)
    if m=='knn_q': params['knn_q__n_neighbors'] = range(1, min_samples-1)
    if stack_model: params['k_best__k'] = range(2, 40)

    return pipe, params


def post_model_fit(bm, X, y):
    bm.fit(X, y)
    return bm

def get_model_output(model_name, cur_df, model_obj, out_dict, run_params, i, min_samples=10, alpha=''):

    print(f'\n{model_name}\n============\n')

    skm, X, y = get_skm(cur_df, model_obj, to_drop=run_params['drop_cols'])
    pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples)

    if model_obj == 'class': 
        proba = True
        alpha = f'_{cut}' 
    else: 
        proba = False

    # fit and append the ADP model
    import time
    start = time.time()
    best_models, oof_data, param_scores = skm.time_series_cv(pipe, X, y, params, n_iter=run_params['n_iters'], n_splits=run_params['n_splits'],
                                                             col_split='game_date', time_split=run_params['cv_time_input'],
                                                             bayes_rand=run_params['opt_type'], proba=proba,
                                                             random_seed=(i+7)*19+(i*12)+6, alpha=alpha)

    print('Time Elapsed:', np.round((time.time()-start)/60,1), 'Minutes')
    # best_models = Parallel(n_jobs=-1, verbose=0)(delayed(post_model_fit)(bm, X, y) for bm in best_models)
    
    col_label = str(alpha)+run_params['rush_pass']
    out_dict = update_output_dict(model_obj, model_name, col_label, out_dict, oof_data, best_models)
    # db_output = add_result_db_output('reg', m, oof_data['scores'], db_output, run_params)

    try: save_param_scores(param_scores, model_obj, model_name, run_params)
    except: print(f'Param save for {model_name} failed')

    return out_dict, best_models, oof_data


#------------------
# Million Predict
#------------------

def select_main_slate_teams(df):

    import datetime as dt

    good_teams = dm.read(f'''
                    SELECT away_team team, gametime, week, year 
                    FROM Gambling_Lines 
                    WHERE year >= 2020
                    UNION
                    SELECT home_team team, gametime, week, year 
                    FROM Gambling_Lines
                    WHERE year >= 2020
                ''', 'Pre_TeamData')


    good_teams.gametime = pd.to_datetime(good_teams.gametime)
    good_teams['day_of_week'] = good_teams.gametime.apply(lambda x: x.weekday())
    good_teams['hour_in_day'] = good_teams.gametime.apply(lambda x: x.hour)

    good_teams = good_teams[(good_teams.day_of_week==6) & (good_teams.hour_in_day <= 17) & (good_teams.hour_in_day > 10)]
    good_teams = good_teams[['team', 'week', 'year']]

    if set_pos == 'Defense': 
        good_teams = good_teams.rename(columns={'team': 'player'})
        df = pd.merge(df, good_teams, on=['player', 'week', 'year'])

    else:
        df = pd.merge(df, good_teams, on=['team', 'week', 'year'])

    return df


def add_sal_columns(df):
    
    for c in df.columns:
        if 'expert' in c: df[c+'_salary'] = df[c] * df.dk_salary
        if 'proj' in c: df[c+'_salary'] = df[c] / df.dk_salary
    
    return df

def predict_million_df(df, run_params):

    df = select_main_slate_teams(df)

    df = df.drop('y_act', axis=1)
    top_players = dm.read("SELECT player, week, year, y_act FROM Top_Players", "DK_Results")

    df = pd.merge(df, top_players, on=['player', 'week', 'year'], how='left')
    df = df.fillna({'y_act': 0})

    df_train_mil, df_predict_mil, _, min_samples_mil = train_predict_split(df, run_params)

    return df_train_mil, df_predict_mil, min_samples_mil, run_params


def predict_million_df_roi(df, run_params):

    df = select_main_slate_teams(df)

    df = df.drop('y_act', axis=1)
    top_players = dm.read('''SELECT player, week, year, prize_return_delta y_act
                             FROM Top_Players_ROI
                             WHERE total_lineups > 250
                          ''', "DK_Results")

    df = pd.merge(df, top_players, on=['player', 'week', 'year'], how='inner')

    df_train_mil, df_predict_mil, _, min_samples_mil = train_predict_split(df, run_params)

    return df_train_mil, df_predict_mil, min_samples_mil, run_params


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

def save_output_dict(out_dict, model_output_path, label, rush_pass):

    save_pickle(out_dict['pred'], model_output_path, f'{rush_pass}{label}_pred')
    save_pickle(out_dict['actual'], model_output_path, f'{rush_pass}{label}_actual')
    save_pickle(out_dict['models'], model_output_path, f'{rush_pass}{label}_models')
    save_pickle(out_dict['scores'], model_output_path, f'{rush_pass}{label}_scores')
    save_pickle(out_dict['full_hold'], model_output_path, f'{rush_pass}{label}_full_hold')



#%%
run_list = [
            ['QB', '', 'full_model'],
            ['RB', '', 'full_model'],
            ['WR', '', 'full_model'],
            ['TE', '', 'full_model'],
            ['Defense', '', 'full_model'],
            ['QB', '', 'backfill'],
            ['RB', '', 'backfill'],
            ['WR', '', 'backfill'],
            ['TE', '', 'backfill'],

            # ['QB', 'rush', 'full_model'],
            # ['QB', 'pass', 'full_model'],
            # ['QB', 'rush', 'backfill'],
            # ['QB', 'pass', 'backfill'],
]

for w in run_weeks:
    run_params['set_week'] = w

    for set_pos, rush_pass, model_type in run_list:

        run_params['rush_pass'] = rush_pass
        run_params['n_splits'] = n_splits
        print(f"\n==================\n{set_pos} {model_type} {rush_pass} {run_params['set_year']} {run_params['set_week']} {vers}\n====================")

        #==========
        # Pull and clean compiled data
        #==========

        # load data and filter down
        pkey, db_output, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
        df, run_params = load_data(model_type, set_pos, run_params)
        df, run_params = create_game_date(df, run_params)

        df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

        # set up blank dictionaries for all metrics
        out_reg, out_class, out_quant, out_million = output_dict(), output_dict(), output_dict(), output_dict()

        #=========
        # Run Models
        #=========

        # run all other models
        model_list = ['adp', 'bridge', 'huber', 'lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'gbmh', 'rf']
        for i, m in enumerate(model_list):
            out_reg, _, _ = get_model_output(m, df_train, 'reg', out_reg, run_params, i, min_samples)
        save_output_dict(out_reg, model_output_path, 'reg', rush_pass)

        # run all other models
        model_list = ['lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c', 'gbmh_c']
        for cut in run_params['cuts']:
            print(f"\n--------------\nPercentile {cut}\n--------------\n")
            df_train_class, df_predict_class = get_class_data(df, cut, run_params)    
            for i, m in enumerate(model_list):
                out_class, _, _= get_model_output(m, df_train_class, 'class', out_class, run_params, i, min_samples)
        save_output_dict(out_class, model_output_path, 'class', rush_pass)

        # run all other models
        model_list = ['gbm_q', 'lgbm_q', 'qr_q', 'knn_q', 'rf_q']
        for i, m in enumerate(model_list):
            for alph in [0.8, 0.95]:
                out_quant, _, _ = get_model_output(m, df_train, 'quantile', out_quant, run_params, i, alpha=alph)
        save_output_dict(out_quant, model_output_path, 'quant', rush_pass)

        # # run the million predict
        # print(f"\n--------------\nRunning Million Predict\n--------------\n")
        # n_splits = run_params['n_splits']; cut='million'
        # df_train_mil, df_predict_mil, min_samples_mil, run_params = predict_million_df(df, run_params)

        # model_list = ['lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c', 'gbmh_c']
        # for i, m in enumerate(model_list):
        #     out_million, _, _= get_model_output(m, df_train_mil, 'class', out_million, run_params, i, min_samples)
        # save_output_dict(out_million, model_output_path, 'million', rush_pass)


        # # run the million predict
        # print(f"\n--------------\nRunning Million Predict ROI\n--------------\n")
        # df_train_mil, df_predict_mil, min_samples_mil, run_params = predict_million_df_roi(df, run_params)

        # model_list = ['adp', 'huber', 'lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'gbmh', 'rf']
        # for i, m in enumerate(model_list):
        #     out_million, _, _= get_model_output(m, df_train_mil, 'reg', out_million, run_params, i, min_samples)
        # save_output_dict(out_million, model_output_path, 'million', rush_pass)

        



#%%
