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


set_pos = 'QB'
is_million = False
model_feature_db = 'Model_Features'
# model_feature_db = 'Model_Features - Copy'


#==========
# Pull and clean compiled data
#==========
#%%
if model_feature_db=='Model_Features':
    
    df2, _ = load_data(model_type, set_pos, {'rush_pass': ''}, "Model_Features - Copy")
    
    df, run_params = load_data(model_type, set_pos, run_params, "Model_Features")
    df, run_params = create_game_date(df, run_params)

    df = pd.merge(df, df2[['player', 'week', 'year']], on=['player', 'week', 'year'])
else:
    df, run_params = load_data(model_type, set_pos, run_params, model_feature_db)
    df, run_params = create_game_date(df, run_params)

for c in df.columns:
    if len(df[df[c]==np.inf]) >0:
        df = df.drop(c, axis=1)

if is_million: 
    df_train, df_predict, min_samples = predict_million_df(df, run_params)
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
    models_test = ['cb_c', 'mlp_c', 'lr_c', 'lgbm_c', 'xgb_c', 'gbmh_c', 'rf_c', 'gbm_c', 'knn_c']
    model_obj = 'class'
else: 
    models_test = ['ridge', 'enet', 'bridge', 'lasso', 'lgbm', 'xgb', 'bridge', 'knn', 'gbm', 'gbmh', 'cb', 'rf', 'mlp']
    # models_test = ['lgbm', 'xgb', 'gbmh', 'cb', 'gbm']
    model_obj = 'reg'


# test_settings = {
#     'Experiment': 'Week7 Create Trials', 
#     'TrialsObj': 'Recent', 
#     'HyperOptAlgo': 'atpe', 
#     'NumTrials': 100, 
#     'LearningRate': 'loguniform(-5, -0.5)'
#     }


# param_scores_all = {}
# for model_name in models_test:
#     run_model(model_name, df_train, df_predict, run_params, model_obj, test_settings)



results = []
for trials_obj in [ 'Cumulative']:
    for hp_algo in ['atpe', 'tpe']:
        for learning_rate in ['loguniform(-3, -0.5)', 'loguniform(-5, -0.5)']:
            if trials_obj == 'New': num_trials = 0
            elif trials_obj == 'Cumulative': num_trials = 2500
            else: num_trials = 100

            test_settings = {
                'Experiment': 'Reg, Week7 Create Trials',
                'TrialsObj': trials_obj,
                'HyperOptAlgo': hp_algo,
                'NumTrials': num_trials,
                'LearningRate': learning_rate
            }
            print(test_settings)

            output_trials = Parallel(n_jobs=-1, verbose=1)(
                delayed(run_model)
                (model_name, df_train, df_predict, run_params, model_obj, test_settings) \
                for model_name in models_test 
                )
            test_settings['TrialsOutput'] = output_trials
            results.append(test_settings)

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




#%%


df = param_scores_all['cb'].copy()

for c in df.columns:
    if 'perc__score_func' in c: df[c] = df[c].apply(lambda x: 'perc_r_regression' if 'r_regression' in str(x) else 'perc_f_regression')
    if 'k_best__score_func' in c and 'feature_union__k_best__score_func' not in c: 
        df[c] = df[c].apply(lambda x: 'kb_r_regression' if 'r_regression' in str(x) else 'kb_f_regression')

    if 'feature_union__k_best__score_func' in c: df[c] = df[c].apply(lambda x: 'fu_kb_r_regression' if 'r_regression' in str(x) else 'fu_kb_f_regression')
    if 'select_from_model' in c: df[c] = df[c].apply(lambda x: str(x))

df['input_features'] = df.feature_union__k_best__k + df.feature_union__agglomeration__n_clusters + df.feature_union__pca__n_components
df.loc[df.k_best__k < df.input_features, 'k_best__k'] = df.loc[df.k_best__k < df.input_features, 'input_features']
# df.select_perc__percentile = df.select_perc__percentile.fillna(100)

X = df.drop('score', axis=1)
for c in X.columns:
    try: X[c] = X[c].astype('float')
    except: pass

for c in X.dtypes[X.dtypes=='object'].index:
    print(c)
    X = pd.concat([X, pd.get_dummies(X[c])], axis=1).drop(c, axis=1)
y = -df.score.astype('float')

# m = ElasticNet(alpha=0.01, l1_ratio=0.05)
# m = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2)
m = LGBMRegressor()

# if type(m) == sklearn.linear_model._ridge.Ridge or type(m)==sklearn.linear_model._coordinate_descent.ElasticNet:
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
        all_coef = all_coef[~all_coef.index.str.contains('entry_type')]
        all_coef[abs(all_coef) > 0.005].sort_values().plot.barh(figsize=(10,18))
    except:
        all_coef = all_coef[[c for c in all_coef.columns if 'week' not in c]]
        X_all = X_all[[c for c in X_all.columns if 'week' not in c]]
        shap.summary_plot(all_coef.values, X_all, feature_names=X_all.columns, plot_size=(18,10), max_display=40, show=False)

def entry_optimize_params(df, max_adjust, model_name):

    adjust_winnings = df.groupby(['trial_num', 'entry_type']).agg(max_lineup_num=('lineup_num', 'max')).reset_index()

    adjust_winnings.loc[adjust_winnings.entry_type=='millions_only', 'max_lineup_num'] = \
        13 / (adjust_winnings.loc[adjust_winnings.entry_type=='millions_only', 'max_lineup_num'] + 1)
    
    adjust_winnings.loc[adjust_winnings.entry_type=='millions_playaction', 'max_lineup_num'] = \
        30 / (adjust_winnings.loc[adjust_winnings.entry_type=='millions_playaction', 'max_lineup_num'] + 1)
    
    df = pd.merge(df, adjust_winnings.drop('entry_type', axis=1), on='trial_num')
    df.winnings = df.winnings / df.max_lineup_num
    
    df.loc[df.winnings >= max_adjust, 'winnings'] = max_adjust
    df.loc[(df.winnings >= 500) & (df.week==8) & (df.year==2022), 'winnings'] = 500

    df.loc[df.trial_num < 520, 'player_drop_multiple'] = 0

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
                     SELECT week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, entry_type, trial_num, repeat_num
                      FROM Entry_Optimize_Results
                      ) USING (week, year, trial_num, repeat_num)
                WHERE trial_num >= 460
                      AND pred_vers = 'sera0_rsq0_mse1_brier1_matt1_bayes'
                      AND week < 17
                    --  AND NOT (week=8 AND year=2022)
                    --  AND (reg_ens_vers LIKE '%team_stats%' OR million_ens_vers LIKE '%team_stats%')
             
                ''', 'Results')

df['week'] = df.week.astype(str) + '_' + df.year.astype(str)
df.loc[df.week!='8_2022', 'winnings'] = df.loc[df.week!='8_2022', 'winnings']*2

model_type = {
 'enet': ElasticNet(alpha=1, l1_ratio=0.1),
 'lasso': Lasso(alpha=0.1),
 'ridge': Ridge(alpha=100),
 'rf': RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=10, n_jobs=-1),
 'lgbm': LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.05, min_samples_leaf=5, n_jobs=-1)

}
w=1
model_name='lgbm'
m = model_type[model_name] 
X, y = entry_optimize_params(df, max_adjust=10000, model_name=model_name)
coef_vals, X = get_model_coef(X, y, m)
show_coef(coef_vals, X)

#%%

weeks = [
         1, 2, 3, 4, 5, 6, 7, 8, 
         9, 10, 11, 12, 13, 14, 15, 16,
         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
         11, 12, 13, 14, 15, 16]
years = [
          2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 
          2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 
          2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023,
          2023, 2023, 2023, 2023, 2023, 2023]

i=0
all_coef = None; X_all = None
for w, yr in zip(weeks, years):
    df = dm.read(f'''SELECT *  
                     FROM Entry_Optimize_Params_Detail 
                     JOIN (
                            SELECT week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, entry_type, trial_num, repeat_num
                            FROM Entry_Optimize_Results          
                          ) USING (week, year, trial_num, repeat_num)
                     WHERE trial_num >= 460
                           AND pred_vers = 'sera0_rsq0_mse1_brier1_matt1_bayes'
                        --   AND (reg_ens_vers LIKE '%team_stats%' OR million_ens_vers LIKE '%team_stats%')
                           AND week = {w}
                           AND year = {yr}
                     ''', 'Results')
    df['week'] = df.week.astype(str) + '_' + df.year.astype(str)
    df.loc[df.week!='8_2022', 'winnings'] = df.loc[df.week!='8_2022', 'winnings']*2

    model_name = 'enet'
    m = model_type[model_name]
    if w == 8 and yr==2022: max_adjust = 1000
    else: max_adjust = 10000
    X, y = entry_optimize_params(df, max_adjust=max_adjust, model_name=model_name)
    coef_vals, X = get_model_coef(X, y, m)
    all_coef, X_all = join_coef(i, all_coef, coef_vals, X_all, X, model_name); i+=1

show_coef(all_coef, X_all)

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
    

#%%

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



#%%

best_trials = dm.read('''SELECT *
                         FROM Entry_Optimize_Results
                         WHERE trial_num >= 460
                         ''', 'Results')

best_trials.loc[best_trials.avg_winnings > 10000, 'avg_winnings'] = 10000

best_trials['non8_winnings'] = best_trials.avg_winnings
best_trials.loc[(best_trials.week == 8) & (best_trials.year==2022), 'non8_winnings'] = 0 

best_trials['avg_winnings_sqrt_pre'] = best_trials.avg_winnings ** 0.5

best_trials = (
    best_trials
    .groupby(['pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers',
              'model_notes', 'manual_adjust', 'trial_num', 'week', 'year'])
    .agg({'avg_winnings': 'mean',
          'avg_winnings_sqrt_pre': 'mean',
          'non8_winnings': 'mean',
          #'perc80': lambda x: np.percentile(x, 80)
          })
    .reset_index()
)

best_trials['avg_winnings_sqrt_post'] = best_trials.avg_winnings ** 0.5
best_trials = (
    best_trials
    .groupby(['pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers', 'model_notes', 'manual_adjust', 'trial_num'])
    .agg({'avg_winnings': 'sum',
          'avg_winnings_sqrt_pre': 'sum',
          'avg_winnings_sqrt_post': 'sum',
          'non8_winnings': 'sum'})
)

winnings_cols = ['avg_winnings', 'avg_winnings_sqrt_pre', 'avg_winnings_sqrt_post', 'non8_winnings']
for c in winnings_cols:
    best_trials[c+'_rank'] = best_trials[c].rank(ascending=False)

winnings_cols = [c+'_rank' for c in winnings_cols]
best_trials['avg_rank'] = best_trials[winnings_cols].mean(axis=1)
best_trials = best_trials.sort_values(by='avg_rank').reset_index()
# best_trials['avg_winnings'] = best_trials.avg_winnings ** 2

best_trials.iloc[:25]

#%%

params = dm.read('''SELECT *
                    FROM Entry_Optimize_Params
                    WHERE trial_num >= 460
                    ''', 'Results')
param_vars = list(params.param.unique())
params['param'] = params.param.astype('str') + '_' + params.param_option.astype('str')

params = params.pivot_table(index=['trial_num'], columns='param', values='option_value').reset_index().fillna(0)

results = dm.read('''SELECT *
                    FROM Entry_Optimize_Results
                    WHERE trial_num >= 460
                         -- AND NOT (week=8 AND year=2022)
                          AND reg_ens_vers = 'random_full_stack_sera0_rsq0_mse1_include2_kfold3'
                          AND million_ens_vers IN (
                                                   'random_full_stack_team_stats_matt0_brier1_include2_kfold3',
                                                   'random_kbest_team_stats_matt0_brier1_include2_kfold3',
                                                   'random_kbest_matt0_brier1_include2_kfold3'
                          )
                          AND std_dev_type iN (
                                                    'spline_class80_q80_matt0_brier1_kfold3',
                                                    'spline_pred_class80_q80_matt0_brier1_kfold3',
                                                    'spline_pred_class80_matt0_brier1_kfold3'
                          )
                    ''', 'Results')

results.loc[results.avg_winnings > 10000, 'avg_winnings'] = 10000
results.loc[~((results.week==8)&(results.year==2022)), 'avg_winnings'] = results.loc[~((results.week==8)&(results.year==2022)), 'avg_winnings'] * 5

results = results.groupby(['pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers', 
                           'ownership_vers', 'entry_type', 'trial_num', 'repeat_num']).agg({'avg_winnings': 'sum'}).reset_index()

results = pd.merge(results, params, on='trial_num')
results = results.drop(['trial_num', 'repeat_num'], axis=1)
for c in results.columns:
    if results.dtypes[c] == 'object': 
        results[c] = results[c].astype('category')

# results.week = results.week.astype('category')
# results.year = results.year.astype('category')

X = results.drop('avg_winnings', axis=1)
y = results.avg_winnings

#%%


params = {
    'n_estimators': range(100, 400, 10),
    'num_leaves': range(20, 300, 10),
    'min_child_samples': range(10, 70, 2),
    'learning_rate': np.arange(0.01, 0.35, 0.02),
    'subsample': np.arange(0.8, 1, 0.02)
}

retrain = False
model = LGBMRegressor(n_jobs=16)
fp = FoldPredict(f'{root_path}/Model_Outputs/Final_LGBM/', retrain=retrain)
fp.cross_fold_train('winnings', model, params, X, y, n_iter=20)

#%%

model = load_pickle(fp.save_path, 'winnings_fold1')
import shap
shap_values = shap.TreeExplainer(model).shap_values(X)
shap.summary_plot(shap_values, X, feature_names=X.columns, plot_size=(8,10), max_display=30, show=False)

#%%
more_opt_columns = [ 'qb_min_iter', 'matchup_seed', 'max_team_type', 'matchup_drop', 'qb_set_max_team', 'ownership_vers', 'use_ownership']
X[[c for c in X.columns if 'max_team_type' in c]].value_counts()

#%%

shap.dependence_plot("ownership_vers_standard_ln", shap_values, X)




#%%

extra_cutoff_val = -20

counts_base = 0
while counts_base < 7 * 1000000:
    for cnt_cutoff in range(-200, -30, 10):
        cnt_cutoff = abs(cnt_cutoff)

        drop_cols = []
        for pv in param_vars:
            for c in X.columns:
                if pv in c:
                    drop_cols.append(c)
        base_cols = [c for c in X.columns if c not in drop_cols]
        counts_base = len(X[base_cols].drop_duplicates())

        for pv in param_vars:
            cur_cols = [c for c in X.columns if pv in c]

            for moc in more_opt_columns:
                if moc in cur_cols[0]: 
                    extra_cutoff = extra_cutoff_val
                    break
                else: 
                    extra_cutoff = 0

            if X[cur_cols].value_counts().max() < cnt_cutoff:
                cur_cnt_cutoff = X[cur_cols].value_counts().max()
            else:
                cur_cnt_cutoff = cnt_cutoff

            cur_cnts = X[cur_cols].value_counts()
            # num_above_cutoff = len(cur_cnts[cur_cnts>=(cur_cnt_cutoff+extra_cutoff)])
            # if num_above_cutoff > 3: c_multiplier = 3
            # else: c_multiplier = len(cur_cnts[cur_cnts>=(cur_cnt_cutoff+extra_cutoff)])
            c_multiplier = len(cur_cnts[cur_cnts>=(cur_cnt_cutoff+extra_cutoff)])
            counts_base = counts_base * c_multiplier
        print(cnt_cutoff, counts_base/1000000)

        if counts_base > 7 * 1000000: break


cnt_cutoff = cnt_cutoff + 10

#%%

drop_cols = []
for pv in param_vars:
    for c in X.columns:
        if pv in c:
            drop_cols.append(c)
base_cols = [c for c in X.columns if c not in drop_cols]
X_predict = X[base_cols].drop_duplicates()
X_predict['cross_idx'] = 1

for pv in param_vars:
    cur_cols = [c for c in X.columns if pv in c]
    cur_cnts = X[cur_cols].value_counts()

    for moc in more_opt_columns:
        if moc in cur_cols[0]: 
            extra_cutoff = extra_cutoff_val
            break
        else: 
            extra_cutoff = 0

    if X[cur_cols].value_counts().max() < cnt_cutoff:
        cur_cnt_cutoff = X[cur_cols].value_counts().max()
    else:
        cur_cnt_cutoff = cnt_cutoff

    cur_col = X[cur_cols].value_counts()[X[cur_cols].value_counts()>=(cur_cnt_cutoff+extra_cutoff)].reset_index().drop(0, axis=1)
    print(cur_col)
    cur_col['cross_idx'] = 1
    X_predict = pd.merge(X_predict, cur_col, on='cross_idx')
    print(pv, X_predict.shape)

X_predict = X_predict.drop('cross_idx', axis=1)
X_predict = X_predict[X.columns]

#%%
winnings_pr = fp.cross_fold_predict('winnings', X_predict, y=pd.Series(np.array([260*32]*len(X_predict)), name='entry'))
grp_cols = [c for c in winnings_pr.columns if c not in ('winnings_pred', 'avg_winnings', 'week', 'year')]
for c in grp_cols:
    winnings_pr[c] = winnings_pr[c].astype('str')

winnings_pr = winnings_pr.sort_values(by='winnings_pred', ascending=False).drop_duplicates().reset_index(drop=True)
winnings_pr = winnings_pr.reset_index().rename(columns={'index': 'param_rank'})

model_notes = 'only_reg_full_stack_non8_times_5_more_options'
date_run = dt.datetime.now().strftime('%Y-%m-%d')
winnings_pr = winnings_pr.assign(model_notes=model_notes, date_run=date_run)

#%%

winnings_pr.loc[:500, [c for c in winnings_pr.columns if 'covar_type' in c]].value_counts()

#%%

try:
    dm.delete_from_db('SimParams', 'Entry_Optimize_Hyperparams', f"model_notes='{model_notes}' AND date_run='{date_run}'", create_backup=False)
    dm.write_to_db(winnings_pr, 'SimParams','Entry_Optimize_Hyperparams', 'append')

except:
    old_df = dm.read('''SELECT * FROM Entry_Optimize_Hyperparams''', 'SimParams')
    winnings_pr = pd.concat([ winnings_pr, old_df], axis=0).reset_index(drop=True)
    winnings_pr = winnings_pr.fillna(0)
    dm.write_to_db(winnings_pr, 'SimParams','Entry_Optimize_Hyperparams', 'replace')

# %%

from sklearn.neighbors import KernelDensity

params = dm.read('''SELECT *
                    FROM Entry_Optimize_Params
                    WHERE trial_num >= 460
                    ''', 'Results')
param_vars = list(params.param.unique())
params['param'] = params.param.astype('str') + '_' + params.param_option.astype('str')

params = params.pivot_table(index=['trial_num'], columns='param', values='option_value').reset_index().fillna(0)

results = dm.read('''SELECT *
                    FROM Entry_Optimize_Results
                    WHERE trial_num >= 460
                         -- AND NOT (week=8 AND year=2022)
                          AND reg_ens_vers = 'random_full_stack_sera0_rsq0_mse1_include2_kfold3'
                          AND million_ens_vers IN ('random_full_stack_team_stats_matt0_brier1_include2_kfold3',
                                                   'random_kbest_team_stats_matt0_brier1_include2_kfold3',
                                                   'random_kbest_matt0_brier1_include2_kfold3')
                          AND std_dev_type NOT IN (
                                                   'spline_pred_class80_matt0_brier1_kfold3', 
                                                   'spline_pred_q80_matt0_brier1_kfold3'
                                                  )
                    ''', 'Results')

results.loc[results.avg_winnings > 10000, 'avg_winnings'] = 10000
results.loc[~((results.week==8)&(results.year==2022)), 'avg_winnings'] = results.loc[~((results.week==8)&(results.year==2022)), 'avg_winnings'] * 5

results = results.groupby(['pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers', 
                           'ownership_vers', 'entry_type', 'trial_num', 'repeat_num']).agg({'avg_winnings': 'sum'}).reset_index()

results = pd.merge(results, params, on='trial_num')
results = results.drop(['trial_num', 'repeat_num'], axis=1)
for c in results.columns:
    if results.dtypes[c] == 'object': 
        results = pd.concat([results, pd.get_dummies(results[c], prefix=c)], axis=1).drop(c, axis=1)


results

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
