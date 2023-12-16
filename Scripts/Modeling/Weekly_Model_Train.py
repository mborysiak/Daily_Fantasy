
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

run_weeks = [15]
verbosity = 50
run_params = {
    
    # set year and week to analyze
    'set_year': 2023,

    # set beginning of validation period
    'val_year_min': 2020,
    'val_week_min': 14,

    # opt params
    'n_iters': 20,
    'n_splits': 5,

    # other parameters
    'use_sample_weight': False,
    'opt_type': 'bayes',
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

    'rush_pass': ''
}

n_splits = run_params['n_splits']

# set position and model type
set_pos = 'RB'
model_type = 'full_model'

# set weights for running model
r2_wt = 0
sera_wt = 0
mse_wt = 1
matt_wt = 1
brier_wt = 1

# set version and iterations
vers = f'sera{sera_wt}_rsq{r2_wt}_mse{mse_wt}_brier{brier_wt}_matt{matt_wt}_bayes'

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


def get_trial_times(root_path, run_params, set_pos, model_type, vers):

    newest_folder = get_newest_folder(f"{root_path}/Model_Outputs/")
    recent_save = get_newest_folder_with_keywords(newest_folder, [set_pos, model_type, vers], [f"_week{run_params['set_week']}_"])

    all_trials = load_pickle(recent_save, 'all_trials')

    times = []
    for k,v in all_trials.items():
        if k!='reg_adp':
            max_trial = len(v.trials) - 1
            trial_times = []
            for i in range(max_trial-100, max_trial):
                trial_times.append(v.trials[i]['refresh_time'] - v.trials[i]['book_time'])
            trial_time = np.mean(trial_times).seconds
            times.append([k, np.round(trial_time / 60, 2)])

    time_per_trial = pd.DataFrame(times, columns=['model', 'time_per_trial']).sort_values(by='time_per_trial', ascending=False)
    time_per_trial['total_time'] = time_per_trial.time_per_trial * 100
    return time_per_trial

def calc_num_trials(time_per_trial, run_params):

    n_iters = run_params['n_iters']
    time_per_trial['percentile_90_time'] = time_per_trial.time_per_trial.quantile(0.8)
    time_per_trial['num_trials'] = n_iters * time_per_trial.percentile_90_time / time_per_trial.time_per_trial
    time_per_trial['num_trials'] = time_per_trial.num_trials.apply(lambda x: np.min([n_iters, np.max([x, n_iters/4])])).astype('int')
    
    return {k:v for k,v in zip(time_per_trial.model, time_per_trial.num_trials)}

def reg_params(df_train, min_samples, num_trials, run_params):
    model_list = ['adp', 'bridge', 'gbm', 'gbmh', 'rf', 'lgbm', 'ridge', 'svr', 'lasso', 'enet', 'knn','xgb']
    label = 'reg'
    func_params_reg = []
    for i, m  in enumerate(model_list):
        try: num_trial = num_trials[f'{label}_{m}']
        except: num_trial = run_params['n_iters']
        func_params_reg.append([m, label, df_train, 'reg', i, min_samples, '', num_trial])

    return func_params_reg

def class_params(df, min_samples, num_trials, run_params):
    model_list = ['gbm_c', 'rf_c','gbmh_c', 'lgbm_c', 'lr_c', 'knn_c','xgb_c']
    func_params_c = []
    for cut in run_params['cuts']:
        label = f'class_{cut}'
        df_train_class, _ = get_class_data(df, cut, run_params) 
        for i, m  in enumerate(model_list):
            try: num_trial = num_trials[f'{label}_{m}']
            except: num_trial = run_params['n_iters']
            func_params_c.append([m, label, df_train_class, 'class', i, min_samples, '', num_trial])

    return func_params_c

def quant_params(df_train, alphas, min_samples, num_trials, run_params):
    model_list =  ['qr_q', 'lgbm_q', 'gbm_q', 'gbmh_q']#'knn_q','rf_q',
    func_params_q = []
    for alph in alphas:
        label = f'quant_{alph}'
        for i, m  in enumerate(model_list):
            try: num_trial = num_trials[f'{label}_{m}']
            except: num_trial = run_params['n_iters']
            func_params_q.append([m, label, df_train, 'quantile', i, min_samples, alph, num_trial])

    return func_params_q

def million_params(df, num_trials, run_params):
    model_list = ['gbm_c', 'rf_c', 'gbmh_c', 'lgbm_c', 'lr_c', 'knn_c','xgb_c' ]
    label = 'million'
    df_train_mil, _, min_samples_mil = predict_million_df(df, run_params)

    func_params_mil = []
    for i, m  in enumerate(model_list):
        try: num_trial = num_trials[f'{label}_{m}']
        except: num_trial = run_params['n_iters']
        func_params_mil.append([m, label, df_train_mil, 'class', i, min_samples_mil, '', num_trial])

    return func_params_mil

def order_func_params(func_params, trial_times):
    
    if trial_times is not None:
        missing_trials = [f'{x[1]}_{x[0]}' for x in func_params if f'{x[1]}_{x[0]}' not in trial_times.values]
        trial_order = missing_trials + list(trial_times.model.values)
        func_params = sorted(func_params, key=lambda x: trial_order.index(f'{x[1]}_{x[0]}'))

    return func_params

def get_skm(skm_df, model_obj, to_drop):
    
    skm_options = {
        'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt, mse_wt=mse_wt),
        'class': SciKitModel(skm_df, model_obj='class', brier_wt=brier_wt, matt_wt=matt_wt),
        'quantile': SciKitModel(skm_df, model_obj='quantile')
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

        if m in ('qr_q', 'gbmh_q'): pipe.steps[-1][-1].quantile = alpha
        elif m in ('rf_q', 'knn_q'): pipe.steps[-1][-1].q = alpha
        else: pipe.steps[-1][-1].alpha = alpha
    


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
    return bayes_rand

def get_newest_folder(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    newest_folder = max(folders, key=lambda f: os.path.getctime(os.path.join(path, f)))
    return os.path.join(path, newest_folder)

def get_newest_folder_with_keywords(path, keywords, ignore_keywords=None):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    # Apply ignore_keywords if provided
    if ignore_keywords:
        folders = [f for f in folders if not any(ignore_keyword in f for ignore_keyword in ignore_keywords)]
    
    matching_folders = [f for f in folders if all(keyword in f for keyword in keywords)]
    
    if not matching_folders:
        return None
    
    newest_folder = max(matching_folders, key=lambda f: os.path.getctime(os.path.join(path, f)))
    return os.path.join(path, newest_folder)


def get_trials(label, m, bayes_rand):

    newest_folder = get_newest_folder(f"{root_path}/Model_Outputs/")
    recent_save = get_newest_folder_with_keywords(newest_folder, [set_pos, model_type, vers], [f"_week{run_params['set_week']}_"])

    if recent_save is not None and bayes_rand=='bayes': 
        try:
            trials = load_pickle(recent_save, 'all_trials')
            trials = trials[f'{label}_{m}']
            print('Loading previous trials')
        except:
            print('No Previous Trials Exist')
            trials = Trials()

    elif bayes_rand=='bayes':
        print('Creating new Trials object')
        trials = Trials()

    else:
        trials = None

    return trials
    

def get_model_output(model_name, label, cur_df, model_obj, run_params, i, min_samples=10, alpha='', n_iter=20):

    print(f'\n{model_name}\n============\n')
    
    bayes_rand = adjust_adp(model_name)
    proba = get_proba(model_obj)
    trials = get_trials(label, model_name, bayes_rand)

    skm, X, y = get_skm(cur_df, model_obj, to_drop=run_params['drop_cols'])
    pipe, params = get_full_pipe(skm, model_name, alpha, min_samples=min_samples, bayes_rand=bayes_rand)
    if trials is not None: trials = update_trials_params(trials, model_name, params, pipe)

    # fit and append the ADP model
    start = time.time()
    best_models, oof_data, param_scores, trials = skm.time_series_cv(pipe, X, y, params, n_iter=n_iter, 
                                                                     n_splits=run_params['n_splits'], col_split='game_date', 
                                                                     time_split=run_params['cv_time_input'],
                                                                     bayes_rand=bayes_rand, proba=proba, trials=trials,
                                                                     random_seed=(i+7)*19+(i*12)+6, alpha=alpha)
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


def unpack_results(out_dict, func_params, results):
    for fp, result in zip(func_params, results):
        model_name, label, _, _, _, _, _, _ = fp
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
            print(df_train.loc[-10:, ['player', 'week', 'year', 'y_act']])

            try:
                trial_times = get_trial_times(root_path, run_params, set_pos, model_type, vers)
                num_trials = calc_num_trials(trial_times, run_params)
                print('Lower trials:', {k:v for k,v in num_trials.items() if v < run_params['n_iters']})
            except:
                num_trials = None
                trial_times = None
                print('No Trials Exist')

            adp_result = get_model_output('adp', 'reg', df_train, 'reg', run_params, 0, min_samples, '', run_params['n_iters'])
            adp_result = list(adp_result)

            func_params = []
            func_params.extend(quant_params(df_train, [0.8, 0.95], min_samples,  num_trials, run_params))
            func_params.extend(reg_params(df_train, min_samples, num_trials, run_params))
            func_params.extend(class_params(df, min_samples, num_trials, run_params))
            func_params.extend(million_params(df, num_trials, run_params))
            func_params = order_func_params(func_params, trial_times)
            
            # run all models in parallel
            results = Parallel(n_jobs=-1, verbose=verbosity)(
                            delayed(get_model_output)
                            (m, label, df, model_obj, run_params, i, min_samples, alpha, n_iter) \
                                for m, label, df, model_obj, i, min_samples, alpha, n_iter in func_params[1:] # skip ADP to append due to Scaler error
                            )
            all_results = []
            all_results.append(adp_result)
            all_results.extend(results)

            # save output for all models
            out_dict = output_dict()
            out_dict = unpack_results(out_dict, func_params, all_results)
            save_output_dict(out_dict, 'all', model_output_path)


 #%%
import os
import shutil


def navigate_folders(root_dir, search_keywords, old_filename, new_filename):
    for dirpath, _, _ in os.walk(root_dir):
        if all(keyword in dirpath for keyword in search_keywords):
            try:
                source_path = os.path.join(dirpath, old_filename)
                destination_path = os.path.join(dirpath, new_filename)
                shutil.copy2(source_path, destination_path)
            except:
                print(dirpath, 'failed')

root_directory = '/Users/mborysia/Documents/Github/Daily_Fantasy//Model_Outputs/2022/'
search_keywords = [f'sera{sera_wt}_rsq{r2_wt}_mse{mse_wt}_brier{brier_wt}_matt{matt_wt}_bayes']

# old_filename = 'quantile80.0_random_kbest_sera1_rsq0_mse0_include2_kfold3.p'
# new_filename = 'quantile80.0_random_kbest_sera0_rsq0_mse1_include2_kfold3.p'

# old_filename = 'class_random_kbest_sera1_rsq0_mse0_include2_kfold3.p'
# new_filename=   'class_random_kbest_sera0_rsq0_mse1_include2_kfold3.p'

old_filename = 'all_trials.p'
new_filename = 'all_trials_old.p'


navigate_folders(root_directory, search_keywords, old_filename, new_filename)

#%%

class_pred = 'sera0_rsq0_mse1_brier1_matt1_bayes'
reg_pred = 'sera0_rsq0_mse1_brier1_matt1_bayes'
new_save = 'sera0_rsq0_mse1_brier1_matt0_bayes'

pos = 'TE'
model_type = 'backfill'
week = 9
filename = 'all_trials'


def extract_dict_data(filename):

    class_data = load_pickle(f'/Users/borys/OneDrive/Documents/Github/Daily_Fantasy//Model_Outputs/2022/{pos}_year2022_week{week}_{model_type}{class_pred}', filename)
    reg_data = load_pickle(f'/Users/borys/OneDrive/Documents/Github/Daily_Fantasy//Model_Outputs/2022/{pos}_year2022_week{week}_{model_type}{reg_pred}', filename)
    
    new_data = {}
    for k,v in reg_data.items():
        if 'class' not in k and 'million' not in k: new_data[k] = v

    for k,v in class_data.items():
        if 'class' in k or 'million' in k: new_data[k] = v

    return new_data

# for week in range(2, 9):
#     for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
#         for model_type in ['full_model', 'backfill']:
#             if pos=='Defense' and model_type=='backfill': continue
#             print(week, pos, model_type)
#             for f in ['all_trials', 'all_models', 'all_pred', 'all_actual', 'all_scores', 'all_param_scores', 'all_full_hold']:
#                 new_data = extract_dict_data(f)
#                 path = f'/Users/borys/OneDrive/Documents/Github/Daily_Fantasy//Model_Outputs/2022/{pos}_year2022_week{week}_{model_type}{new_save}'
#                 if not os.path.exists(path):
#                     os.makedirs(path)
#                 save_pickle(new_data, path, f)

#%%
extract_dict_data('all_trials')
extract_dict_data('all_models')
extract_dict_data('all_pred')
extract_dict_data('all_actual')
extract_dict_data('all_scores')
extract_dict_data('all_param_scores')
extract_dict_data('all_full_hold')

# %%
extract_dict_data('all_scores')
# %%
extract_dict_data('all_models')

# %%
out = get_model_output('lr_c', 'million', func_params[2][2], 'class', run_params, 10, min_samples, None, 10)
# %%
out
# %%
out[1]['full_hold'][out[1]['full_hold'].player=='Kenneth Gainwell']
# %%
