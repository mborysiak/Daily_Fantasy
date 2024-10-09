
#%%
# core packages
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt
import matplotlib.pyplot as plt
import time
from hyperopt import Trials
from wakepy import keep
import copy
import optuna

from ff.db_operations import DataManage
from ff import general as ffgeneral
from skmodel import SciKitModel
import zModel_Functions as mf
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

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
verbosity = 50
run_params = {
    
    # set year and week to analyze
    'set_year': 2024,

    # set beginning of validation period
    'val_year_min': 2020,
    'val_week_min': 14,

    # opt params
    'n_iters': 20,
    'n_splits': 5,

    # other parameters
    'use_sample_weight': False,
    'cuts': [33, 80, 95],
    'met': 'y_act',

    # set number of weeks back to begin validation
    'back_weeks': {
        'QB': 32,
        'RB': 32,
        'WR': 32,
        'TE': 32,
        'Defense': 32
    },

    'rush_pass': '',

    'opt_type': 'optuna',
    'hp_algo': 'tpe',
    'num_past_trials': 100,
    'optuna_timeout': 60*8
}

n_splits = run_params['n_splits']

# set position and model type
set_pos = 'RB'
model_type = 'full_model'

# set weights for running model
r2_wt = 0
sera_wt = 0
mse_wt = 1
matt_wt = 0
brier_wt = 1

# set version and iterations
vers = f"sera{sera_wt}_rsq{r2_wt}_mse{mse_wt}_brier{brier_wt}_matt{matt_wt}_{run_params['opt_type']}_{run_params['hp_algo']}_numtrials{run_params['num_past_trials']}_higherkb"

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
                                                  FROM {set_pos}_Data{run_params['rush_pass']}_Week{run_params['set_week']}
                                                ''', f"Model_Features_{run_params['set_year']}")
        
    elif model_type=='backfill': df = dm.read(f'''SELECT * 
                                                  FROM Backfill_{set_pos}_Week{run_params['set_week']}
                                                  WHERE pos='{set_pos}' ''', 
                                                  f"Model_Features_{run_params['set_year']}")

    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * 
                          FROM {set_pos}_Data{run_params['rush_pass']}_Week{run_params['set_week']}_v2
                       ''', 
                       f"Model_Features_{run_params['set_year']}")
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
    min_samples = int(df_train[df_train.game_date < run_params['cv_time_input']].shape[0] / 4)  
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
    return {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}, 'param_scores': {}, 'trials': {}}

def get_last_run_week(w, run_params):
    if w == 1: 
        run_params['last_run_year'] = run_params['set_year'] - 1
        run_params['last_run_week'] = 16
    else: 
        run_params['last_run_year'] = run_params['set_year']
        run_params['last_run_week'] = w - 1
    
    return run_params


def rename_existing(old_study_db, new_study_db, study_name):

    import datetime as dt
    new_study_name = study_name + '_' + dt.datetime.now().strftime('%Y%m%d%H%M%S')
    optuna.copy_study(from_study_name=study_name, from_storage=old_study_db, to_storage=new_study_db, to_study_name=new_study_name)
    optuna.delete_study(study_name=study_name, storage=new_study_db)


def get_new_study(old_db, new_db, old_name, new_name, num_trials):

    old_storage = optuna.storages.RDBStorage(
                                url=old_db,
                                engine_kwargs={"pool_size": 64, 
                                            "connect_args": {"timeout": 60},
                                            },
                                )
    
    new_storage = optuna.storages.RDBStorage(
                                url=new_db,
                                engine_kwargs={"pool_size": 64, 
                                            "connect_args": {"timeout": 60},
                                            },
                                )
    
    if old_name is not None:
        old_study = optuna.create_study(
            study_name=old_name,
            storage=old_storage,
            load_if_exists=True
        )
    
    try:
        next_study = optuna.create_study(
            study_name=new_name, 
            storage=new_storage, 
            load_if_exists=False
        )

    except:
        rename_existing(old_storage, new_storage, new_name)
        next_study = optuna.create_study(
            study_name=new_name, 
            storage=new_storage, 
            load_if_exists=False
        )
    
    if old_name is not None and len(old_study.trials) > 0:
        print(f"Loaded {new_name} study with {old_name} {len(old_study.trials)} trials")
        next_study.add_trials(old_study.trials[-num_trials:])

    return next_study
    

def get_optuna_study(model_name, label, vers, run_params, set_pos, model_type):
    time.sleep(5*np.random.random())
    old_name = f"{label}_{model_name}_{vers}_{set_pos}_{run_params['model_type']}_{run_params['last_run_year']}_{run_params['last_run_week']}"
    new_name = f"{label}_{model_name}_{vers}_{set_pos}_{run_params['model_type']}_{run_params['set_year']}_{run_params['set_week']}"
    next_study = get_new_study(run_params['last_study_db'], run_params['study_db'], old_name, new_name, run_params['num_past_trials'])
    return next_study
    

def get_trial_times(root_path, run_params, set_pos, model_type, vers):

    yr = run_params['last_run_year']
    wk = run_params['last_run_week']
    recent_save = f"{root_path}/Model_Outputs/{yr}/{set_pos}_year{yr}_week{wk}_{model_type}{vers}"
    all_trials = load_pickle(recent_save, 'all_trials')

    times = []
    for k,v in all_trials.items():
        if k!='reg_adp':
            max_trial = len(v.trials) - 1
            trial_times = []
            for i in range(max_trial-100, max_trial):
                trial_times.append(v.trials[i]['refresh_time'] - v.trials[i]['book_time'])
            trial_time = np.mean(trial_times).seconds
            times.append([f'{set_pos}_{k}_{model_type}', np.round(trial_time / 60, 2)])

    time_per_trial = pd.DataFrame(times, columns=['model', 'time_per_trial']).sort_values(by='time_per_trial', ascending=False)
    time_per_trial['total_time'] = time_per_trial.time_per_trial * 100

    return time_per_trial

def calc_num_trials(time_per_trial, run_params):

    n_iters = run_params['n_iters']
    time_per_trial['percentile_90_time'] = time_per_trial.time_per_trial.quantile(0.5)
    time_per_trial['num_trials'] = n_iters * time_per_trial.percentile_90_time / time_per_trial.time_per_trial
    time_per_trial['num_trials'] = time_per_trial.num_trials.apply(lambda x: np.min([n_iters, np.max([x, 4])])).astype('int')
    
    return {k:v for k,v in zip(time_per_trial.model, time_per_trial.num_trials)}

def reg_params(df_train, min_samples, num_trials, run_params, set_pos, model_type):
    model_list = ['adp', 'knn','mlp', 'bridge', 'ridge', 'svr', 'lasso', 'enet','xgb', 'cb', 'gbm', 'gbmh', 'lgbm', 'rf']
    label = 'reg'
    func_params_reg = []
    for i, m  in enumerate(model_list):
        try: num_trial = num_trials[f'{label}_{m}']
        except: num_trial = run_params['n_iters']
        rp = copy.deepcopy(run_params)
        func_params_reg.append([set_pos, m, label,  df_train, 'reg', rp, i, min_samples, '', num_trial, model_type])

    return func_params_reg

def class_params(df, min_samples, num_trials, run_params, set_pos, model_type):
    model_list = ['gbm_c', 'cb_c', 'mlp_c', 'rf_c','gbmh_c', 'lgbm_c', 'lr_c', 'knn_c','xgb_c']
    func_params_c = []
    for cut in run_params['cuts']:
        label = f'class_{cut}'
        df_train_class, _ = get_class_data(df, cut, run_params) 
        for i, m  in enumerate(model_list):
            try: num_trial = num_trials[f'{label}_{m}']
            except: num_trial = run_params['n_iters']
            rp = copy.deepcopy(run_params)
            func_params_c.append([set_pos, m, label,  df_train_class, 'class', rp, i, min_samples, '', num_trial, model_type])

    return func_params_c

def quant_params(df_train, alphas, min_samples, num_trials, run_params, set_pos, model_type):
    model_list =  ['qr_q', 'lgbm_q', 'gbm_q', 'gbmh_q', 'cb_q']#'knn_q','rf_q',
    func_params_q = []
    for alph in alphas:
        label = f'quant_{alph}'
        for i, m  in enumerate(model_list):
            try: num_trial = num_trials[f'{label}_{m}']
            except: num_trial = run_params['n_iters']
            rp = copy.deepcopy(run_params)
            func_params_q.append([set_pos, m, label, df_train, 'quantile', rp, i, min_samples, alph, num_trial, model_type])

    return func_params_q

def million_params(df, num_trials, run_params, set_pos, model_type):
    model_list = ['gbm_c', 'cb_c', 'mlp_c', 'rf_c', 'gbmh_c', 'lgbm_c', 'lr_c', 'knn_c', 'xgb_c' ]
    label = 'million'
    df_train_mil, _, min_samples_mil = predict_million_df(df, run_params)

    func_params_mil = []
    for i, m  in enumerate(model_list):
        try: num_trial = num_trials[f'{label}_{m}']
        except: num_trial = run_params['n_iters']
        rp = copy.deepcopy(run_params)
        func_params_mil.append([set_pos, m, label, df_train_mil, 'class', rp, i, min_samples_mil, '', num_trial, model_type])

    return func_params_mil

def order_func_params(func_params, trial_times):
    
    if trial_times is not None:
        missing_trials = [f'{x[0]}_{x[2]}_{x[1]}_{x[-1]}' for x in func_params if f'{x[0]}_{x[2]}_{x[1]}_{x[-1]}' not in trial_times.values]
        trial_order = missing_trials + list(trial_times.model.values)
        func_params = sorted(func_params, key=lambda x: trial_order.index(f'{x[0]}_{x[2]}_{x[1]}_{x[-1]}'))

    return func_params

def get_skm(skm_df, model_obj, to_drop, hp_algo):
    
    skm_options = {
        'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt, mse_wt=mse_wt, hp_algo=hp_algo),
        'class': SciKitModel(skm_df, model_obj='class', brier_wt=brier_wt, matt_wt=matt_wt, hp_algo=hp_algo),
        'quantile': SciKitModel(skm_df, model_obj='quantile', hp_algo=hp_algo)
    }
    
    skm = skm_options[model_obj]
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10, bayes_rand='rand'):

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
                                                skm.piece('k_best_fu'),
                                                skm.piece('pca')
                                                ]),
                                skm.piece('k_best'),
                                skm.piece(m)])

    elif skm.model_obj == 'class':
        pipe = skm.model_pipe([skm.piece('random_sample'),
                               skm.piece('std_scale'), 
                               skm.piece('select_perc_c'),
                               skm.feature_union([
                                                skm.piece('pca'),
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best_c_fu'),
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

        if m in ('qr_q', 'gbmh_q'): pipe.set_params(**{f'{m}__quantile': alpha})
        elif m in ('rf_q', 'knn_q'): pipe.set_params(**{f'{m}__q': alpha})
        elif m == 'cb_q': pipe.set_params(**{f'{m}__loss_function': f'Quantile:alpha={alpha}'})
        else: pipe.set_params(**{f'{m}__alpha': alpha})
    

    # get the params for the current pipe and adjust if needed
    params = skm.default_params(pipe, bayes_rand, min_samples=min_samples)
    if m=='adp': 
        params['feature_select__cols'] = [
                                           ['game_date', 'year', 'week', 'ProjPts', 'dk_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
                                           ['year', 'week',  'ProjPts', 'dk_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
                                            [ 'ProjPts', 'dk_salary','projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank']
                                        ]
        params['k_best__k'] = range(1, 14)
    
    return pipe, params

def update_trials_params(trials, m, params, pipe):

    if m not in ('cb', 'cb_c', 'cb_q'):
        
        m_params = [p.split('__')[1] for p in params.keys() if m in p]

        hyper_params = {
            p: pipe.steps[-1][1].get_params()[p] for p in m_params
        }

        for trial in trials:
            # Update each trial with the new hyperparameters
            for k,v in hyper_params.items():
                if k not in trial['misc']['vals']:
                    trial['misc']['vals'][k] = [v]
                    trial['misc']['idxs'][k] = [trial['tid']]

    return trials

def get_proba(model_obj):
    if model_obj == 'class': proba = True
    else: proba = False
    return proba

def adjust_adp(model_name):
    if model_name == 'adp': bayes_rand = 'rand'
    else: bayes_rand = run_params['opt_type']
    return bayes_rand, None


def get_recent_trials(trials, n=200):
    # Sort trials based on 'tid'
    sorted_trials = sorted(trials.trials, key=lambda trial: trial['tid'], reverse=True)
    # Get the most recent 'n' trials
    recent_trials = sorted_trials[:n]

    # Create a new Trials object and add the recent trials to it
    new_trials = Trials()
    for trial in recent_trials:
        new_trials.insert_trial_docs([trial])
        new_trials.refresh()
    return new_trials



def get_trials(label, m, bayes_rand, run_params, set_pos):

    yr = run_params['last_run_year']
    wk = run_params['last_run_week']
    recent_save = f"{root_path}/Model_Outputs/{yr}/{set_pos}_year{yr}_week{wk}_{model_type}{vers}"

    if recent_save is not None and bayes_rand=='bayes': 
        try:
            trials = load_pickle(recent_save, 'all_trials')
            trials = trials[f'{label}_{m}']
            print('Loading previous trials')
            trials = get_recent_trials(trials, n=run_params['num_past_trials'])
        except:
            print('No Previous Trials Exist')
            trials = Trials()

    elif bayes_rand=='bayes':
        print('Creating new Trials object')
        trials = Trials()

    else:
        trials = None

    return trials
    

def get_model_output(set_pos, model_name, label, cur_df, model_obj, run_params, i, min_samples=10, alpha='', n_iter=20):

    print(f'\n{model_name}\n============\n')
    
    bayes_rand, trials = adjust_adp(model_name)
    proba = get_proba(model_obj)

    if bayes_rand == 'bayes': trials = get_trials(label, model_name, bayes_rand, run_params, set_pos)
    elif bayes_rand == 'optuna': trials = get_optuna_study(label, model_name, vers, run_params, set_pos, model_type)
    
    skm, X, y = get_skm(cur_df, model_obj, to_drop=run_params['drop_cols'], hp_algo=run_params['hp_algo'])
    pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples, bayes_rand=bayes_rand)
    if trials is not None and bayes_rand=='bayes': trials = update_trials_params(trials, model_name, params, pipe)

#    trials.trials_dataframe().duration.mean().seconds


    # fit and append the ADP model
    start = time.time()
    best_models, oof_data, param_scores, trials = skm.time_series_cv(pipe, X, y, params, n_iter=n_iter, 
                                                                     n_splits=run_params['n_splits'], col_split='game_date', 
                                                                     time_split=run_params['cv_time_input'],
                                                                     bayes_rand=bayes_rand, proba=proba, trials=trials,
                                                                     random_seed=(i+7)*19+(i*12)+6, alpha=alpha,
                                                                     optuna_timeout=run_params['optuna_timeout'])
    best_models = [bm.fit(X,y) for bm in best_models]
    print('Time Elapsed:', np.round((time.time()-start)/60,1), 'Minutes')
    
    return best_models, oof_data, param_scores, trials


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
        if 'rank' in c: df[c+'_salary'] = df[c] * df.dk_salary
        if 'proj' in c: df[c+'_salary'] = df[c] / df.dk_salary
        if 'ffa' in c: df[c+'_salary'] = df[c] / df.dk_salary
    
    return df

def predict_million_df(df, run_params):

    df = select_main_slate_teams(df)

    df = df.drop('y_act', axis=1)
    top_players = dm.read("SELECT player, week, year, y_act FROM Top_Players", "DK_Results")

    df = pd.merge(df, top_players, on=['player', 'week', 'year'], how='left')
    df = df.fillna({'y_act': 0})

    df_train_mil, df_predict_mil, _, min_samples_mil = train_predict_split(df, run_params)

    return df_train_mil, df_predict_mil, min_samples_mil


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

def update_output_dict(out_dict, label, m, result):

    best_models, oof_data, param_scores, trials = result

    # append all of the metric outputs
    lbl = f'{label}_{m}'
    out_dict['pred'][lbl] = oof_data['hold']
    out_dict['actual'][lbl] = oof_data['actual']
    out_dict['scores'][lbl] = oof_data['scores']
    out_dict['models'][lbl] = best_models
    out_dict['full_hold'][lbl] = oof_data['full_hold']
    out_dict['param_scores'][lbl] = param_scores
    out_dict['trials'][lbl] = trials

    return out_dict


def unpack_results(out_dict, func_params, results, set_pos, model_type):
    for fp, result in zip(func_params, results):
        set_pos_cur, model_name, label, _, _, _, _, _, _, _, model_type_cur = fp
        if set_pos_cur == set_pos and model_type_cur == model_type:
            out_dict = update_output_dict(out_dict, label, model_name, result)
    return out_dict


def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')


def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


def save_output_dict(out_dict, label, model_output_path):

    label = label.split('_')[0]
    save_pickle(out_dict['pred'], model_output_path, f'{label}_pred')
    save_pickle(out_dict['actual'], model_output_path, f'{label}_actual')
    save_pickle(out_dict['models'], model_output_path, f'{label}_models')
    save_pickle(out_dict['scores'], model_output_path, f'{label}_scores')
    save_pickle(out_dict['full_hold'], model_output_path, f'{label}_full_hold')
    save_pickle(out_dict['param_scores'], model_output_path, f'{label}_param_scores')
    save_pickle(out_dict['trials'], model_output_path, f'{label}_trials')


#%%

# w = run_weeks[0]
# run_params['set_week'] = w
# run_params = get_last_run_week(w, run_params)

# if not os.path.exists(f"{root_path}/Scripts/Modeling/optuna/{run_params['set_year']}/week{run_params['set_week']}/"):
#     os.makedirs(f"{root_path}/Scripts/Modeling/optuna/{run_params['set_year']}/week{run_params['set_week']}/")

# run_params['last_study_db'] = f"sqlite:///optuna/{run_params['last_run_year']}/week{run_params['last_run_week']}/weekly_train_week{run_params['last_run_week']}_year{run_params['last_run_year']}.sqlite3"
# run_params['study_db'] = f"sqlite:///optuna/{run_params['set_year']}/week{run_params['set_week']}/weekly_train_week{run_params['set_week']}_year{run_params['set_year']}.sqlite3"

# func_params = []
# adp_result_dict = {}
# trial_times = pd.DataFrame()
# all_model_output_path = {}

# set_pos = 'QB'
# run_params['rush_pass'] = ''
# run_params['n_splits'] = 5
# run_params['model_type'] = 'full_model'
# print(f"\n==================\n{set_pos} {model_type} {run_params['set_year']} {run_params['set_week']} {vers}\n====================")

# #==========
# # Pull and clean compiled data
# #==========

# # load data and filter down
# pkey, db_output, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
# all_model_output_path[f'{set_pos}_{model_type}'] = model_output_path
# df, run_params = load_data(model_type, set_pos, run_params)
# df, run_params = create_game_date(df, run_params)

# df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)
# print(df_train.loc[-10:, ['player', 'week', 'year', 'y_act']])

# func_params.extend(reg_params(df_train, min_samples, 5, run_params, set_pos, model_type))

# import time
# start = time.time()
# for set_pos, m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter, _  in func_params[3:4]:
#     results = get_model_output(set_pos, m, label, df, model_obj, run_params, i, min_samples, alpha, 3)
# time.time() - start


#%%

with keep.running() as m:
    if not m.success:
        print('Fell Asleep')

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
    ]

    for w in run_weeks:
        run_params['set_week'] = w
        run_params = get_last_run_week(w, run_params)
        
        if not os.path.exists(f"{root_path}/Scripts/Modeling/optuna/{run_params['set_year']}/week{run_params['set_week']}/"):
            os.makedirs(f"{root_path}/Scripts/Modeling/optuna/{run_params['set_year']}/week{run_params['set_week']}/")
        run_params['last_study_db'] = f"sqlite:///optuna/{run_params['last_run_year']}/week{run_params['last_run_week']}/weekly_train_week{run_params['last_run_week']}_year{run_params['last_run_year']}.sqlite3"
        run_params['study_db'] = f"sqlite:///optuna/{run_params['set_year']}/week{run_params['set_week']}/weekly_train_week{run_params['set_week']}_year{run_params['set_year']}.sqlite3"

        func_params = []
        adp_result_dict = {}
        trial_times = pd.DataFrame()
        all_model_output_path = {}
        for set_pos, rush_pass, model_type in run_list:

            run_params['rush_pass'] = rush_pass
            run_params['n_splits'] = n_splits
            run_params['model_type'] = model_type
            print(f"\n==================\n{set_pos} {model_type} {rush_pass} {run_params['set_year']} {run_params['set_week']} {vers}\n====================")

            #==========
            # Pull and clean compiled data
            #==========

            # load data and filter down
            pkey, db_output, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
            all_model_output_path[f'{set_pos}_{model_type}'] = model_output_path
            df, run_params = load_data(model_type, set_pos, run_params)
            df, run_params = create_game_date(df, run_params)

            df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)
            print(df_train.loc[-10:, ['player', 'week', 'year', 'y_act']])

            try:
                trial_times_pos = get_trial_times(root_path, run_params, set_pos, model_type, vers)
                num_trials = calc_num_trials(trial_times_pos, run_params)
                trial_times = pd.concat([trial_times, trial_times_pos], axis=0)
                trial_times = trial_times.sort_values(by='time_per_trial', ascending=False).reset_index(drop=True)
                print('Lower trials:', {k:v for k,v in num_trials.items() if v < run_params['n_iters']})
            except:
                num_trials = None
                trial_times = None
                print('No Trials Exist')

            func_params.extend(quant_params(df_train, [0.8, 0.95], min_samples,  num_trials, run_params, set_pos, model_type))
            func_params.extend(reg_params(df_train, min_samples, num_trials, run_params, set_pos, model_type))
            func_params.extend(class_params(df, min_samples, num_trials, run_params, set_pos, model_type))
            func_params.extend(million_params(df, num_trials, run_params, set_pos, model_type))
        
        # func_params = order_func_params(func_params, trial_times)
            
        # run all models in parallel
        results = Parallel(n_jobs=-1, verbose=verbosity)(
                        delayed(get_model_output)
                        (set_pos, m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter) \
                            for set_pos, m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter, _ in func_params
                        )

        # save output for all models
        for k, v in all_model_output_path.items():
            cur_set_pos = k.split('_')[0]
            cur_model_type = '_'.join(k.split('_')[1:])
            out_dict = output_dict()
            out_dict = unpack_results(out_dict, func_params, results, cur_set_pos, cur_model_type)
            save_output_dict(out_dict, 'all', v)


#%%

for i, f in enumerate(func_params):
    print(i, f[0], f[1])

#%%
import time
start = time.time()
for set_pos, m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter, _  in func_params[14:15]:
    results = get_model_output(set_pos, m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter)
time.time() - start
# results = Parallel(n_jobs=-1, verbose=verbosity)(
#                 delayed(get_model_output)
#                 (set_pos, m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter) \
#                     for set_pos, m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter, _ in func_params_test # skip ADP to append due to Scaler error
#                 )

# %%

from optuna.visualization import plot_parallel_coordinate

study = optuna.create_study(
            study_name='lr_c_million_sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_higherkb_WR_full_model_2023_2',
            storage=run_params['study_db'],
            load_if_exists=True
        )

plot_parallel_coordinate(study)
# %%
