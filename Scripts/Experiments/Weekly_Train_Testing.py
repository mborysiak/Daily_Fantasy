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

def load_all_pickles(model_output_path, label):
    pred = load_pickle(model_output_path, f'{label}_pred')
    actual = load_pickle(model_output_path, f'{label}_actual')
    models = load_pickle(model_output_path, f'{label}_models')
    scores = load_pickle(model_output_path, f'{label}_scores')
    try: full_hold = load_pickle(model_output_path, f'{label}_full_hold')
    except: full_hold = None
    return pred, actual, models, scores, full_hold


def load_data(model_type, set_pos, run_params, model_feature_db):

    # load data and filter down
    if model_type=='full_model': df = dm.read(f'''SELECT * 
                                                  FROM {set_pos}_Data{run_params['rush_pass']}
                                                ''', model_feature_db)
    elif model_type=='backfill': df = dm.read(f'''SELECT * 
                                                  FROM Backfill 
                                                  WHERE pos='{set_pos}' ''', model_feature_db)

    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * 
                          FROM {set_pos}_Data{run_params['rush_pass']}2
                       ''', model_feature_db)
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

    df_predict = df[(df.game_date >= run_params['train_time_split']) & \
                    (df.week < 17)].reset_index(drop=True)
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

def add_sal_columns(df):

    for c in df.columns:
        if 'expert' in c: df[c+'_salary'] = df[c] * df.dk_salary
        if 'rank' in c: df[c+'_salary'] = df[c] * df.dk_salary
        if 'proj' in c: df[c+'_salary'] = df[c] / df.dk_salary
        if 'ffa' in c: df[c+'_salary'] = df[c] / df.dk_salary
    
    return df

def show_calibration_curve(y_true, y_pred, n_bins=10):

    from sklearn.calibration import calibration_curve

    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    
    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='quantile')
    plt.plot(y, x, marker = '.', label = 'Quantile')

    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
    plt.plot(y, x, marker = '+', label = 'Uniform')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.show()

    print('Brier Score:', brier_score_loss(y_true, y_pred))

def output_test_results(oof_data, df_predict, pred, run_params, model_name, model_obj, test_settings):

    from sklearn.metrics import mean_squared_error, r2_score, brier_score_loss

    if model_obj == 'reg':
        val_score = mean_squared_error(oof_data['full_val'].y_act, oof_data['full_val'].pred)
        test_score = mean_squared_error(oof_data['full_hold'].y_act, oof_data['full_hold'].pred)
        pred_score = mean_squared_error(df_predict.y_act, pred)

    elif model_obj == 'class':
        val_score = brier_score_loss(oof_data['full_val'].y_act, oof_data['full_val'].pred)
        test_score = brier_score_loss(oof_data['full_hold'].y_act, oof_data['full_hold'].pred)
        pred_score = brier_score_loss(df_predict.y_act, pred)

    val_r2 = r2_score(oof_data['full_val'].y_act, oof_data['full_val'].pred)
    test_r2 = r2_score(oof_data['full_hold'].y_act, oof_data['full_hold'].pred)
    pred_r2 = r2_score(df_predict.y_act, pred)


    output = pd.DataFrame({
        'Pos': [set_pos],
        'ModelObj': [model_obj],
        'ModelName': [model_name],
        'TestWeekStart': [run_params['set_week']],
        'TestYear': [run_params['set_year']],
        'Experiment': [test_settings['Experiment']],
        'TrialsObj': [test_settings['TrialsObj']],
        'HyperOptAlgo': [test_settings['HyperOptAlgo']],
        'NumTrials': [test_settings['NumTrials']],
        'LearningRate': [test_settings['LearningRate']],
        'ValScore': [val_score],
        'TestScore': [test_score],
        'PredScore': [pred_score],
        'ValR2': [val_r2],
        'TestR2': [test_r2],
        'PredR2': [pred_r2],
        'DateRun': [dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    })

    for c in ['ValScore', 'TestScore', 'PredScore', 'ValR2', 'TestR2', 'PredR2']:
        output[c] = output[c].round(3)

    del_str = f'''Pos='{set_pos}' 
                AND ModelObj='{model_obj}' 
                AND ModelName='{model_name}'
                AND TestWeekStart='{run_params['set_week']}'
                AND Experiment='{test_settings['Experiment']}'
                AND TrialsObj='{test_settings['TrialsObj']}'
                AND HyperOptAlgo='{test_settings['HyperOptAlgo']}'
                AND NumTrials='{test_settings['NumTrials']}'
                AND LearningRate='{test_settings['LearningRate']}'
                '''
    dm.delete_from_db('Results', 'Model_Evaluations', del_str, create_backup=False)
    dm.write_to_db(output, 'Results', 'Model_Evaluations', 'append')


#======================================================================================================================
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

    new_df = pd.DataFrame()
    for c in df.columns:
        try:
            if 'expert' in c: new_df[c+'_salary'] = df[c] * df.dk_salary
            if 'rank' in c: new_df[c+'_salary'] = df[c] * df.dk_salary
            if 'proj' in c: new_df[c+'_salary'] = df[c] / df.dk_salary
            if 'ffa' in c: new_df[c+'_salary'] = df[c] / df.dk_salary
        except:
            pass

    df = pd.concat([df, new_df], axis=1)
    
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
    top_players = dm.read('''SELECT player, week, year, CASE WHEN prize_return_pct > 0 THEN 1 ELSE 0 END y_act 
                             FROM Top_Players_ROI
                             WHERE total_lineups > 250
                          ''', "DK_Results")

    df = pd.merge(df, top_players, on=['player', 'week', 'year'], how='inner')

    df_train_mil, df_predict_mil, _, min_samples_mil = train_predict_split(df, run_params)

    return df_train_mil, df_predict_mil, min_samples_mil, run_params


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

#==================================================================
# SHAP Plots for Best Models
#==================================================================
#%%
# set year to analyze

from hyperopt import fmin, hp, atpe, tpe, Trials, space_eval
from catboost import CatBoostRegressor, CatBoostClassifier
from hyperopt.pyll import scope

# set the model version
model_type ='full_model'

run_params = {
    
    # set year and week to analyze
    'set_year': 2023,
    'set_week': 8,

    # set beginning of validation period
    'val_year_min': 2021,
    'val_week_min': 2,

    # opt params
    'n_iters': 10,
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
print('Running Model for Year:', run_params['set_year'], 'Week:', run_params['set_week'])


set_pos = 'WR'
is_million = False
is_over_proj = True
model_feature_db = 'Model_Features'
# model_feature_db = 'Model_Features - Copy'


#==========
# Pull and clean compiled data
#==========

df, run_params = load_data(model_type, set_pos, run_params, model_feature_db)
df, run_params = create_game_date(df, run_params)

for c in df.columns:
    if len(df[df[c]==np.inf]) >0:
        df = df.drop(c, axis=1)

if is_million: 
    df_train, df_predict, min_samples = predict_million_df(df, run_params)
    df_train = add_sal_columns(df_train)
    df_predict = add_sal_columns(df_predict)
elif is_over_proj:
    df['proj_diff' ] = df.y_act - df.avg_proj_points
    proj_diff_cut = df.groupby('game_date')['proj_diff'].apply(lambda x: np.percentile(x, 90)).reset_index()
    top_scores = df.groupby('game_date')['y_act'].apply(lambda x: np.percentile(x, 75)).reset_index()
    df = pd.merge(df, proj_diff_cut.rename(columns={'proj_diff': 'cut_perc'}), on=['game_date'])
    df = pd.merge(df, top_scores.rename(columns={'y_act': 'top_perc'}), on=['game_date'])
    df['y_act'] = np.where((df.y_act > (df.avg_proj_points + df.cut_perc)) & (df.y_act > df.top_perc), 1, 0)
    df = df.drop(['proj_diff', 'top_perc', 'cut_perc'], axis=1)

    df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)
    df_train = add_sal_columns(df_train)
    df_predict = add_sal_columns(df_predict)
else:
    df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)


def run_model(model_name, df_train, df_predict, run_params, model_obj, test_settings):
    
    param_scores_all = {}
    
    hp_algo = test_settings['HyperOptAlgo']
    skm = SciKitModel(df_train, model_obj=model_obj, hp_algo=hp_algo, 
                      matt_wt=0, brier_wt=1, mse_wt=1, sera_wt=0, mae_wt=0, rsq_wt=0, logloss_wt=0)
    X, y = skm.Xy_split('y_act', run_params['drop_cols'])
    opt_type = run_params['opt_type']
    param_scores_all[model_name] = {}

    #------------
    # Get Regression Data
    #------------

    if model_obj == 'class': 
        proba=True
        sperc = 'select_perc_c'
        kbest = 'k_best_c'
        sfm = 'select_from_model_c'
    else: 
        proba=False
        sperc = 'select_perc'
        kbest = 'k_best'
        sfm = 'select_from_model'


    if model_obj == 'class': 
        pipe = skm.model_pipe([ skm.piece('random_sample'),
                            skm.piece('std_scale'), 
                            skm.piece(sperc),
                            skm.feature_union([
                                            skm.piece('agglomeration'), 
                                            skm.piece(f'{kbest}'),
                                            ]),
                            skm.piece(kbest),
                            skm.piece(model_name)
                        ])
    else: 
        pipe = skm.model_pipe([ skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.piece(sperc),
                                skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece(f'{kbest}'),
                                                skm.piece('pca')
                                                ]),
                                skm.piece(kbest),
                                skm.piece(model_name)
                            ])

    params = skm.default_params(pipe, opt_type)

    if model_name in ('lgbm', 'xgb', 'gbmh', 'cb', 'gbm', 'lgbm_c', 'xgb_c', 'gbmh_c', 'cb_c', 'gbm_c') and \
        test_settings['LearningRate'] == 'learning_rate=loguniform(-5, -0.5)':
        params[f'{model_name}__learning_rate'] = hp.loguniform(f'learning_rate', -5, -0.5)
    

    if is_million: trial_label = 'million'
    else: trial_label = 'reg'

    models_path = f'C:/Users/borys/OneDrive/Documents/GitHub/Daily_Fantasy/Model_Outputs'
    trials_path = f"{models_path}/{run_params['set_year']}/{set_pos}_year{run_params['set_year']}_week{run_params['set_week']}_{model_type}sera0_rsq0_mse1_brier1_matt1_bayes/"
    trials = load_pickle(trials_path, 'all_trials')

    try:
        if test_settings['TrialsObj'] == 'Cumulative': 
            cur_trial = trials[f'{trial_label}_{model_name}']
        elif test_settings['TrialsObj'] == 'Recent': 
            cur_trial = trials[f'{trial_label}_{model_name}']
            cur_trial = get_recent_trials(cur_trial, test_settings['NumTrials'])
        else: 
            cur_trial = Trials()
    except:
        cur_trial = Trials()

    # trials_path = f'C:/Users/borys/OneDrive/Documents/GitHub/Daily_Fantasy/Model_Outputs/Model_Evaluations/{hp_algo}/{set_pos}/'
    # trials = load_pickle(trials_path, 'reg_trials_week7')
    # if test_settings['TrialsObj'] == 'Recent':
    #     cur_trial = trials[f"{model_name}_{test_settings['LearningRate']}"]
    #     cur_trial = get_recent_trials(cur_trial, test_settings['NumTrials'])
    # else:
    #     cur_trial = Trials()

    print('Running Model:', model_name)
    # run the model with parameter search
    best_models, oof_data, param_scores, trials = skm.time_series_cv(pipe, X, y, params, n_iter=20,
                                                                col_split='game_date',n_splits=5,
                                                                time_split=run_params['cv_time_input'],
                                                                bayes_rand=opt_type, proba=proba,
                                                                sample_weight=False, trials=cur_trial,
                                                                random_seed=12345)
    param_scores_all[model_name] = param_scores
    mf.show_scatter_plot(oof_data['full_hold']['pred'], oof_data['full_hold']['y_act'])

    if model_obj == 'class':
        show_calibration_curve(oof_data['full_hold']['y_act'], oof_data['full_hold']['pred'])

    i=0
    best_models[i].fit(X,y)
    if model_obj == 'class':
        pred = pd.Series(best_models[i].predict_proba(df_predict[X.columns])[:,1], name='pred')
        print(r2_score(df_predict.y_act, pred))
        print(pd.concat([
            df_predict[['player', 'week', 'year', 'y_act']],
            pred], axis=1).sort_values(by='pred', ascending=False).head(50))
    else:
        pred = pd.Series(best_models[i].predict(df_predict[X.columns]), name='pred')
        print(r2_score(df_predict.y_act, pred))
        print(pd.concat([
            df_predict[['player', 'week', 'year', 'y_act']],
            pred], axis=1).sort_values(by='pred', ascending=False).head(50))
        
    output_test_results(oof_data, df_predict, pred, run_params, model_name, model_obj, test_settings)

    return trials


if is_million: 
    # models_test = ['lgbm_c', 'xgb_c', 'gbmh_c', 'cb_c', 'gbm_c']
    models_test = ['lr_c', 'cb_c', 'mlp_c',  'lgbm_c', 'xgb_c', 'gbmh_c', 'rf_c', 'gbm_c', 'knn_c']
    model_obj = 'class'
elif is_over_proj:
    models_test = ['lr_c', 'cb_c', 'mlp_c', 'lgbm_c', 'xgb_c', 'gbmh_c', 'rf_c', 'gbm_c', 'knn_c']
    model_obj = 'class'
else: 
    models_test = ['ridge', 'enet', 'bridge', 'lasso', 'lgbm', 'xgb', 'bridge', 'knn', 'gbm', 'gbmh', 'cb', 'rf', 'mlp']
    # models_test = ['lgbm', 'xgb', 'gbmh', 'cb', 'gbm']
    model_obj = 'reg'


test_settings = {
    'Experiment': 'OverProj', 
    'TrialsObj': 'New', 
    'HyperOptAlgo': 'atpe', 
    'NumTrials': 100, 
    'LearningRate': 'loguniform(-5, -0.5)'
    }


param_scores_all = {}
for model_name in models_test[:1]:
    run_model(model_name, df_train, df_predict, run_params, model_obj, test_settings)



# results = []
# for trials_obj in [ 'Cumulative']:
#     for hp_algo in ['atpe', 'tpe']:
#         for learning_rate in ['loguniform(-3, -0.5)', 'loguniform(-5, -0.5)']:
#             if trials_obj == 'New': num_trials = 0
#             elif trials_obj == 'Cumulative': num_trials = 2500
#             else: num_trials = 100

#             test_settings = {
#                 'Experiment': 'Reg, Week7 Create Trials',
#                 'TrialsObj': trials_obj,
#                 'HyperOptAlgo': hp_algo,
#                 'NumTrials': num_trials,
#                 'LearningRate': learning_rate
#             }
#             print(test_settings)

#             output_trials = Parallel(n_jobs=-1, verbose=1)(
#                 delayed(run_model)
#                 (model_name, df_train, df_predict, run_params, model_obj, test_settings) \
#                 for model_name in models_test 
#                 )
#             test_settings['TrialsOutput'] = output_trials
#             results.append(test_settings)

#%%

# trials_out_tpe = {}
# trials_out_atpe = {}

# for r in results:
#     trials = r['TrialsOutput']
#     learning_rate = r['LearningRate']
#     hp_algo = r['HyperOptAlgo']

#     for m, trials in zip(models_test, trials):
#         if hp_algo == 'tpe': trials_out_tpe[f'{m}_{learning_rate}'] = trials
#         else: trials_out_atpe[f'{m}_{learning_rate}'] = trials

# models_path = f'C:/Users/borys/OneDrive/Documents/GitHub/Daily_Fantasy/Model_Outputs/Model_Evaluations/tpe/{set_pos}/'
# if not os.path.exists(models_path): os.makedirs(models_path)
# save_pickle(trials_out_tpe, models_path, 'reg_trials_week7')

# models_path = f'C:/Users/borys/OneDrive/Documents/GitHub/Daily_Fantasy/Model_Outputs/Model_Evaluations/atpe/{set_pos}/'
# if not os.path.exists(models_path): os.makedirs(models_path)
# save_pickle(trials_out_atpe, models_path, 'reg_trials_week7')

#%%

# df = dm.read('''SELECT * FROM Model_Evaluations ''', 'Results')
# df['Experiment'] = 'Hyperopt Setting'
# df = df[['Pos', 'ModelObj', 'ModelName', 'TestWeekStart', 'TestYear', 'Experiment', 'TrialsObj', 'HyperOptAlgo', 'NumTrials', 'LearningRate', 
#          'ValScore', 'TestScore', 'PredScore', 'ValR2', 'TestR2', 'PredR2', 'DateRun']]
# dm.write_to_db(df, 'Results', 'Model_Evaluations', 'replace')

#%%
df = dm.read('''SELECT * FROM Model_Evaluations ''', 'Results')
xx = (
    df
    .groupby(['Experiment', 'Pos', 'ModelObj', 'TrialsObj', 'HyperOptAlgo', 'NumTrials', 'LearningRate'])
    .agg({'PredR2': 'mean', 'TestR2': 'mean'})
    .reset_index()
)
xx['MeanScore'] = (xx.PredR2 + xx.TestR2) / 2
xx.sort_values(by=['Experiment', 'ModelObj', 'Pos', 'MeanScore'], ascending=[True, True, True, False])

#%%

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

class FoldPredict:

    def __init__(self, save_path, retrain=True):
        self.save_path = save_path
        self.retrain = retrain

    def cross_fold_train(self, model_type, model, params, X, y, n_iter=10):

        for i, (train_idx, test_idx) in enumerate(KFold(n_splits=4, shuffle=True).split(X)):
            print(f'Fold {i+1}')
            X_train, _ = X.iloc[train_idx], X.iloc[test_idx]
            y_train, _ = y.iloc[train_idx], y.iloc[test_idx]

            grid = RandomizedSearchCV(model, params, n_iter=n_iter, scoring='neg_mean_squared_error', n_jobs=2, cv=4)
            grid.fit(X_train,y_train)
            
            scores = pd.concat([pd.DataFrame(grid.cv_results_['params']), 
                                pd.DataFrame(np.sqrt(-grid.cv_results_['mean_test_score']))], axis=1).sort_values(by=0, ascending=False)
            print(scores)

            best_model = grid.best_estimator_
            print(best_model)
    
            if not os.path.exists(self.save_path): os.makedirs(self.save_path)
            save_pickle(best_model, self.save_path, f'{model_type}_fold{i}')

    def cross_fold_predict(self, model_type, X, y):

        predictions = pd.DataFrame()
        for _, (train_idx, test_idx) in enumerate(KFold(n_splits=4, shuffle=True).split(X)):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            cur_predict = pd.DataFrame()
            for i in range(4):
                model = load_pickle(self.save_path, f'{model_type}_fold{i}')
                if self.retrain: model.fit(X_train, y_train)
                model_i_predict = pd.DataFrame(model.predict(X_test), index=test_idx, columns=[f'score_{i}'])
                cur_predict = pd.concat([cur_predict, model_i_predict], axis=1)
            
            cur_predict = pd.DataFrame(cur_predict.mean(axis=1), columns=[f'{model_type}_pred'])
            print('MAE:', np.round(np.mean(np.abs(cur_predict[f'{model_type}_pred'] - y_test)), 5))
            cur_predict = pd.concat([cur_predict, y_test], axis=1)
            predictions = pd.concat([predictions, cur_predict], axis=0)
                
        predictions = pd.merge(predictions, X,left_index=True, right_index=True)

        return predictions
    
    

df = dm.read('''SELECT * FROM Model_Evaluations WHERE ModelObj='reg' ''', 'Results')
df['Score'] = df.PredR2 + df.TestR2
y = df.Score
X = df.drop(['Score', 'ModelName', 'DateRun', 'ValScore', 'TestScore','PredScore','ValR2', 'PredR2', 'TestR2'], axis=1)

for c in X.columns:
    X = pd.concat([X, pd.get_dummies(X[c], prefix=c)], axis=1).drop(c, axis=1)

X.columns = [c.replace('(', '').replace(')','').replace('-','_').replace(',','') for c in X.columns]

params = {
    'n_estimators': range(100, 400, 10),
    'num_leaves': range(20, 300, 10),
    'min_child_samples': range(10, 70, 2),
    'learning_rate': np.arange(0.01, 0.35, 0.02),
    'subsample': np.arange(0.8, 1, 0.02)
}

retrain = False
model = LGBMRegressor(n_jobs=16)
fp = FoldPredict(f'{root_path}/Model_Outputs/Model_Evaluations/LGBM/', retrain=retrain)
fp.cross_fold_train('winnings', model, params, X, y, n_iter=20)

model = load_pickle(fp.save_path, 'winnings_fold1')
import shap
shap_values = shap.TreeExplainer(model).shap_values(X)
shap.summary_plot(shap_values, X, feature_names=X.columns, plot_size=(8,10), max_display=30, show=False)
