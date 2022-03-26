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

def load_pickle(path, fname):
        with gzip.open(f"{path}/{fname}.p", 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object

def get_class_data(df, cut):

    # set up the training and prediction datasets for the classification 
    df_train_class = df[df.game_date < train_time_split].reset_index(drop=True)
    df_predict_class = df[df.game_date == train_time_split].reset_index(drop=True)

    # set up the target variable to be categorical based on Xth percentile
    cut_perc = df_train_class.groupby('game_date')['y_act'].apply(lambda x: np.percentile(x, cut))
    df_train_class = pd.merge(df_train_class, cut_perc.reset_index().rename(columns={'y_act': 'cut_perc'}), on=['game_date'])
    
    if cut > 33:
        df_train_class['y_act'] = np.where(df_train_class.y_act >= df_train_class.cut_perc, 1, 0)
    else: 
        df_train_class['y_act'] = np.where(df_train_class.y_act <= df_train_class.cut_perc, 1, 0)

    df_train_class = df_train_class.drop('cut_perc', axis=1)

    return df_train_class, df_predict_class

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


def load_all_pickles(model_output_path, label):
    pred = load_pickle(model_output_path, f'{label}_pred')
    actual = load_pickle(model_output_path, f'{label}_actual')
    models = load_pickle(model_output_path, f'{label}_models')
    scores = load_pickle(model_output_path, f'{label}_scores')
    try: full_hold = load_pickle(model_output_path, f'{label}_full_hold')
    except: full_hold = None
    return pred, actual, models, scores, full_hold


def X_y_stack_old(met, pred, actual):

    X = pd.DataFrame([v for k,v in pred.items() if met in k]).T
    X.columns = [k for k,_ in pred.items() if met in k]
    y = pd.Series(actual[X.columns[0]], name='y_act')

    return X, y


def X_y_stack_full(met, full_hold):
    i = 0
    for k, v in full_hold.items():
        if i == 0:
            df = v.copy()
            df = df.rename(columns={'pred': k})
        else:
            df_cur = v.rename(columns={'pred': k}).drop('y_act', axis=1)
            df = pd.merge(df, df_cur, on=['player', 'team', 'week','year'])
        i+=1

    X = df[[c for c in df.columns if met in c or 'y_act_' in c]].reset_index(drop=True)
    y = df['y_act'].reset_index(drop=True)
    return X, y, df


def X_y_stack(met, full_hold, pred, actual):
    if full_hold is not None:
        X_stack, y_stack, _ = X_y_stack_full(met, full_hold)
    else:
        X_stack, y_stack = X_y_stack(met, pred, actual)

    return X_stack, y_stack


def get_class_predictions(df, models_class):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict_class = pd.DataFrame()
    for cut in cuts:

        print(f"\n--------------\nPercentile {cut}\n--------------\n")

        df_train_class, df_predict_class = get_class_data(df, cut)

        skm_class_final = SciKitModel(df_train_class, model_obj='class')
        X_class_final, y_class_final = skm_class_final.Xy_split(y_metric='y_act', to_drop=drop_cols)
        
        for k, v in models_class.items():
            if str(cut) in k:
                m = skm_class_final.ensemble_pipe(v)
                m.fit(X_class_final, y_class_final)
                cur_pred = pd.Series(m.predict_proba(df_predict_class[X_class_final.columns])[:,1], name=k)
                X_predict_class = pd.concat([X_predict_class, cur_pred], axis=1)

    return X_predict_class


def get_quant_predictions(df_train, df_predict, models_quant):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict_quant = pd.DataFrame()
    for alpha in [0.8, 0.95]:

        print(f"\n--------------\nAlpha {alpha}\n--------------\n")

        skm_quant = SciKitModel(df_train, model_obj='quantile')
        X_quant_final, y_quant_final = skm_quant.Xy_split(y_metric='y_act', to_drop=drop_cols)
        
        for k, v in models_quant.items():
            if str(alpha) in k:
                m = skm_quant.ensemble_pipe(v)
                m.fit(X_quant_final, y_quant_final)
                cur_pred = m.predict(df_predict[X_quant_final.columns])
                cur_pred = pd.Series(cur_pred, name=k)
                X_predict_quant = pd.concat([X_predict_quant, cur_pred], axis=1)
    
    return X_predict_quant


def optimize_reg_model(final_m, skm_stack, X_stack, y_stack, rs=1234):

    if 'yes_kbest' in ensemble_vers:
        feature_piece = skm_stack.piece('k_best')
        print_coef = True
    elif 'randsample' in ensemble_vers:
        feature_piece = skm_stack.piece('random_sample')
        print_coef = False

    # get the model pipe for stacking setup and train it on meta features
    stack_pipe = skm_stack.model_pipe([
                            feature_piece,
                            # skm_stack.feature_union([
                            #                 skm_stack.piece('agglomeration'), 
                            #                 skm_stack.piece('k_best'),
                            #                 skm_stack.piece('pca')
                            #                 ]),
                            # skm_stack.piece('k_best'), 
                            skm_stack.piece(final_m)
                        ])

    stack_params = skm_stack.default_params(stack_pipe)
    if 'yes_kbest' in ensemble_vers:
        stack_params['k_best__k'] = range(1, int(X_stack.shape[1]))

    if set_pos != 'Defense': use_sample_weight = sample_weight_models[final_m]
    else: use_sample_weight = False

    best_model, stack_score, adp_score = skm_stack.best_stack(stack_pipe, stack_params,
                                                                X_stack, y_stack, n_iter=50, 
                                                                run_adp=True, print_coef=print_coef,
                                                                sample_weight=use_sample_weight,
                                                                random_state=rs)

    return best_model, stack_score, adp_score


def optimize_quant_model(df_train, X_stack, y_stack, alpha, rs=1234):

    skm_stack = SciKitModel(df_train, model_obj='quantile')

    # get the model pipe for stacking setup and train it on meta features
    stack_pipe = skm_stack.model_pipe([
                           skm_stack.piece('random_sample'),
                            skm_stack.piece('gbm_q')
                        ])

    stack_params = skm_stack.default_params(stack_pipe)
    stack_params['random_sample__frac'] = np.arange(0.2, 1, 0.05)
    stack_pipe.steps[-1][-1].alpha = alpha
    best_model, stack_score, adp_score = skm_stack.best_stack(stack_pipe, stack_params,
                                                              X_stack, y_stack, n_iter=50, 
                                                              run_adp=False, print_coef=False,
                                                              alpha=alpha, random_state=rs)

    return best_model, stack_score, adp_score



def get_reg_predict_features(models, X, y):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
        
        predictions = pd.DataFrame()

        for m in ind_models:

            if 'sweight' in vers:
                sweight = f'{m.steps[-1][0]}__sample_weight'
                wts = np.where(y > 0, y, 0)
                fit_params = {sweight: wts}
            else:
                fit_params = {}

            m.fit(X, y, **fit_params)
            reg_predict = m.predict(df_predict[X.columns])
            predictions = pd.concat([predictions, pd.Series(reg_predict)], axis=1)
            
        predictions = predictions.mean(axis=1)
        predictions = pd.Series(predictions, name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict


def stack_predictions(X_predict, best_models, final_models):
    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):
        cur_prediction = np.round(bm.predict(X_predict), 2)
        cur_prediction = pd.Series(cur_prediction, name=f'pred_{met}_{fm}')
        predictions = pd.concat([predictions, cur_prediction], axis=1)

    return predictions


def show_scatter_plot(y_pred, y, label='Total', r2=True):
    plt.scatter(y_pred, y)
    plt.xlabel('predictions');plt.ylabel('actual')
    plt.show()

    from sklearn.metrics import r2_score
    if r2: print(f'{label} R2:', r2_score(y, y_pred))
    else: print(f'{label} Corr:', np.corrcoef(y, y_pred)[0][1])


def show_top_predictions(y_pred, y, r2=False):

    val_high_score = pd.concat([pd.Series(y_pred), pd.Series(y)], axis=1)
    val_high_score.columns = ['predictions','y_act']
    val_high_score = val_high_score[val_high_score.predictions >= \
                                    np.percentile(val_high_score.predictions, 75)]
    show_scatter_plot(val_high_score.predictions, val_high_score.y_act, label='Top', r2=r2)


def best_average_models(scores, final_models, stack_val_pred, predictions, use_sample_weight=False):

    n_scores = []
    for i in range(len(scores)):
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:i+1]
        model_idx = np.array(final_models)[top_n]

        if use_sample_weight and set_pos != 'Defense':
            wts = y_stack
        else:
            wts = None
        
        n_score = skm_stack.custom_score(y_stack, stack_val_pred[model_idx].mean(axis=1))
        n_scores.append(n_score)

    print('All Average Scores:', np.round(n_scores, 3))
    best_n = np.argmin(n_scores[2:])
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:best_n+3]
    
    model_idx = np.array(final_models)[top_n]
    best_val = stack_val_pred[model_idx]

    model_idx = [f'pred_y_act_{m}' for m in model_idx]
    best_predictions = predictions[model_idx]

    return best_val, best_predictions


def get_quantile_sd(output):
    
    for alpha in [5, 16, 84, 95]:

        print(f'\nRunning Percentile {alpha}\n=============')
        perc_model, _, perc_pred = optimize_quant_model(df_train, X_stack, y_stack, alpha=alpha/100)
        # show_scatter_plot(perc_pred['stack_pred'], perc_pred['y'], r2=False)
        output[f'perc{alpha}'] = perc_model.predict(X_predict)

    output['std_dev'] = (output.perc84 - output.perc16) / 2

    ratio = (3.4*output.std_dev + output.pred_fp_per_game) / (1.96*output.std_dev + output.pred_fp_per_game)
    output['max_score'] = output.perc95 * ratio
    output['min_score'] = output.perc5 / ratio

    return output


def add_actual(df):
    if set_pos=='Defense': pl = 'defTeam'
    else: pl = 'player'
    actual_pts = dm.read(f'''SELECT {pl} player, fantasy_pts actual_pts
                            FROM {set_pos}_Stats 
                            WHERE week={set_week} 
                                and season={set_year}''', 'FastR')
    df = pd.merge(df, actual_pts, on='player')
    return df


def save_output_to_db(output):

    output['pos'] = set_pos
    output['version'] = vers
    output['ensemble_vers'] = ensemble_vers
    output['std_dev_type'] = std_dev_type
    output['model_type'] = model_type
    output['week'] = set_week
    output['year'] = set_year

    output = output[['player', 'dk_salary', 'sd_metric', 'max_metric', 'pred_fp_per_game', 'std_dev',
                        'dk_rank', 'pos', 'version', 'model_type', 'max_score', 'min_score',
                        'week', 'year', 'ensemble_vers', 'std_dev_type']]

    del_str = f'''pos='{set_pos}' 
                AND version='{vers}'
                AND ensemble_vers='{ensemble_vers}' 
                AND std_dev_type='{std_dev_type}'
                AND week={set_week} 
                AND year={set_year}
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
np.random.seed(1234)

# set year to analyze
set_year = 2021


# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 10

met = 'y_act'
show_plots = True

# set the model version
set_weeks = [18]
pred_versions = [
                 'standard_proba_sera_brier_lowsample',
                 ]
ensemble_versions = [
                     'no_weight_yes_kbest_sera',
                     ]

std_dev_types = ['spline_pred_actuals']


sample_weight_models = {'adp': False,
                        'ridge': False,
                        'lasso': False,
                        'bridge': False,
                        'lgbm': False,
                        'xgb': False,
                        'rf': False,
                        'gbm': False}


######################################################################

sd_cols_choice = {
    'spline': ['fantasyPoints', 'ProjPts', 'projected_points', 'roll_std'],
    'spline_pred': ['pred_fp_per_game'],
    'spline_pred_actuals': ['pred_fp_per_game', 'roll_std'],
    'spline_quantile': ['']
}

max_cols_choice = {
    'spline': ['fantasyPoints', 'ProjPts', 'projected_points', 'roll_max'],
    'spline_pred': ['pred_fp_per_game'],
    'spline_pred_actuals': ['pred_fp_per_game', 'roll_max']
}


for set_week, vers, ensemble_vers, std_dev_type in zip(set_weeks, pred_versions, ensemble_versions, std_dev_types):

    sd_cols = sd_cols_choice[std_dev_type]
    max_cols = max_cols_choice[std_dev_type]

    splines = {}    
    for k, p in zip([1, 2, 2, 2, 2], ['QB', 'RB', 'WR', 'TE', 'Defense']):
        print(f'Checking Splines for {p}')
        spl_sd, spl_perc = get_std_splines(p, set_week, set_year, sd_cols, max_cols, show_plot=show_plots, k=k)
        splines[p] = [spl_sd, spl_perc]

    for model_type in ['full_model', 'backfill']:

        if model_type == 'full_model': positions = ['QB', 'RB', 'WR', 'TE',  'Defense']
        elif model_type == 'backfill': positions =  ['QB', 'RB', 'WR', 'TE']

        for set_pos in positions:

            def_cuts = [33, 75, 90]
            off_cuts = [33, 80, 95]
            if set_pos == 'Defense': cuts = def_cuts
            else: cuts = off_cuts

            #-------------
            # Set up output dataset
            #-------------

            all_vars = [set_pos, set_year, set_week]

            pkey = f'{set_pos}_year{set_year}_week{set_week}_{model_type}{vers}'
            db_output = {'set_pos': set_pos, 'set_year': set_year, 'set_week': set_week}
            db_output['pkey'] = pkey

            model_output_path = f'{root_path}/Model_Outputs/{set_year}/{pkey}/'
            if not os.path.exists(model_output_path): os.makedirs(model_output_path)


            #================================================================================================================================

            #==========
            # Pull and clean compiled data
            #==========
            
            df, drop_cols = load_data(model_type, set_pos)
            df, cv_time_input, train_time_split = create_game_date(df)
            df_train, df_predict, output_start, min_samples = train_predict_split(df, train_time_split, cv_time_input)

            if 'quantile' in std_dev_type:

                #------------
                # Make the Class Predictions
                #------------

                pred_class, actual_class, models_class, scores_class, full_hold_class = load_all_pickles(model_output_path, 'class')
                X_predict_class = get_class_predictions(df, models_class)
                X_stack_class, _ = X_y_stack('class', full_hold_class, pred_class, actual_class)

                #------------
                # Get Regression Data
                #------------

                # get the X and y values for stack training for the current metric
                pred, actual, models, scores, full_hold_reg = load_all_pickles(model_output_path, 'reg')
                X_stack, y_stack = X_y_stack('reg', full_hold_reg, pred, actual)
                X_stack = pd.concat([X_stack, X_stack_class], axis=1)

                #------------
                # Make the Quantile Predictions
                #------------
                
                try:
                    pred_quant, actual_quant, models_quant, scores_quant, full_hold_quant = load_all_pickles(model_output_path, 'quant')
                    X_predict_quant = get_quant_predictions(df_train, df_predict, models_quant)
                    X_stack_quant, _ = X_y_stack('quant', full_hold_quant, pred_quant, actual_quant)
                    X_stack = pd.concat([X_stack, X_stack_quant], axis=1)
                except:
                    print('No Quantile Data Available')
                    X_stack_quant = None
                    X_predict_quant = None
#%%
            # pull in the predictions from old runs
            old_predictions = dm.read(f'''SELECT * 
                                           FROM Model_Predictions 
                                           WHERE version='{vers}'
                                                 AND week={set_week}
                                                 AND year={set_year}
                                                 AND ensemble_vers='{ensemble_vers}'
                                                 AND pos='{set_pos}'
                                                 AND model_type='{model_type}'
                                                 AND std_dev_type='spline'
                                            ''', 'Simulation')

            best_predictions = old_predictions[['player', 'pred_fp_per_game']]
            output_start = pd.merge(output_start, best_predictions, on='player')
            best_predictions = output_start[['pred_fp_per_game']]
            output_start = output_start.drop('pred_fp_per_game', axis=1)

#%%

def get_rolling_stats(output, set_pos, week, year):

    # pull actual stats, rolling stats, and projections
    actual_stats = pull_actual_stats(set_pos, week, year)
    stats = rolling_max_std(actual_stats)
    recent = recent_roll_stats(stats)

    output = pd.merge(output, recent, on='player', how='left')
    output.roll_std = output.roll_std.fillna(recent.roll_std.median())
    output.roll_max = output.roll_max.fillna(recent.roll_std.median())

    return output


def spline_std_max(df, splines, set_pos, sd_cols, max_cols):

    df = create_sd_max_metrics(df, sd_cols, max_cols)

    df['std_dev'] = splines[set_pos][0](df.sd_metric)
    df['max_score'] = splines[set_pos][1](df.max_metric)

    return df


def get_std_dev_features(output_start, predictions):

    output = output_start[['player', 'dk_salary', 'fantasyPoints', 'ProjPts', 'projected_points']].copy()
    output['pred_fp_per_game'] = predictions.mean(axis=1)
    output = get_rolling_stats(output, set_pos, set_week, set_year)

    return output

std_dev_features = get_std_dev_features(output_start, best_predictions)
spline_std_max(std_dev_features, splines, set_pos, sd_cols, max_cols)


# if 'spline' in std_dev_type and 'quantile' in std_dev_type:
    #     output = spline_std_max(output, splines, set_pos, set_week, set_year, sd_cols, max_cols)
    #     output['min_score'] = df_train.y_act.min()
    #     output = output.rename(columns={'std_dev': 'std_dev_sp', 
    #                                     'max_score': 'max_score_sp', 
    #                                     'min_score': 'min_score_sp'})
    #     output = get_quantile_sd(output)
    #     output['std_dev'] = (output.std_dev + output.std_dev_sp) / 2
    #     output['min_score'] = (output.min_score + output.min_score_sp) / 2
    #     output['max_score'] = (output.max_score + output.max_score_sp) / 2

    # elif 'spline' in std_dev_type:
    #     output = spline_std_max(output, splines, set_pos, set_week, set_year, sd_cols, max_cols)
    #     output['min_score'] = df_train.y_act.min()

    # elif 'quantile' in std_dev_type:
    #     output = get_quantile_sd(output)

    # if 'pred' in std_dev_type and 'spline' in std_dev_type:
    #     output['pred_fp_per_game'] = predictions.mean(axis=1)
    
    # output = output.sort_values(by='dk_salary', ascending=False)
    # output['dk_rank'] = range(len(output))
    # output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

#%%
#===================
# Create Outputs
#===================

output = create_output(output_start, best_predictions, splines, sd_cols, max_cols)
try:  
    output = add_actual(output)
    display(output.loc[:50, ['player', 'dk_salary','dk_rank', 'pred_fp_per_game', 'actual_pts', 'std_dev', 'max_score']])
    show_scatter_plot(output.pred_fp_per_game, output.actual_pts)
    output = output.drop('actual_pts', axis=1)
except:
    display(output.loc[:50, ['player', 'dk_salary','dk_rank', 'pred_fp_per_game', 'std_dev', 'max_score']])

# save_output_to_db(output)


#%%
def trunc_normal(player_data, num_samples=1000):

    import scipy.stats as stats

    # create truncated distribution
    lower, upper = player_data.min_score,  player_data.max_score
    lower_bound = (lower - player_data.pred_fp_per_game) / player_data.std_dev, 
    upper_bound = (upper - player_data.pred_fp_per_game) / player_data.std_dev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc= player_data.pred_fp_per_game, scale= player_data.std_dev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates

import seaborn as sns

sns.distplot(trunc_normal(output.iloc[0]))
plt.xlim(0, 50);

# %%
