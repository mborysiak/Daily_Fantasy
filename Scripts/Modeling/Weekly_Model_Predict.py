#%%

# core packages
from random import sample
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
from Fix_Standard_Dev import *
from sklearn.metrics import brier_score_loss
import zModel_Functions as mf
from joblib import Parallel, delayed
from hyperopt import Trials, hp
from hyperopt.pyll import scope

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

from sklearn import set_config
set_config(display='text')

#====================
# Data Loading Functions
#====================

def create_pkey_output_path(set_pos, run_params, model_type, vers):


    pkey = f"{set_pos}_year{run_params['set_year']}_week{run_params['set_week']}_{model_type}{vers}"
    model_output_path = f"{root_path}/Model_Outputs/{run_params['set_year']}/{pkey}/"
    if not os.path.exists(model_output_path): os.makedirs(model_output_path)
    run_params['model_output_path'] = model_output_path
    
    return pkey, run_params, model_output_path

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
    print('Drop cols:', drop_cols)

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
    output_start = df_predict[['player', 'team', 'week', 'year', 'dk_salary']].copy().drop_duplicates()

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
    
    skm_options = {
        'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt, mse_wt=mse_wt),
        'class': SciKitModel(skm_df, model_obj='class', brier_wt=brier_wt, matt_wt=matt_wt),
        'quantile': SciKitModel(skm_df, model_obj='quantile')
    }
    
    skm = skm_options[model_obj]
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10, bayes_rand='rand'):

    if skm.model_obj=='class': kb = 'k_best_c'
    else: kb = 'k_best'

    stack_models = {

        'full_stack': skm.model_pipe([
                                      skm.piece('std_scale'), 
                                      skm.feature_union([
                                                    skm.piece('agglomeration'), 
                                                    skm.piece(kb),
                                                    skm.piece('pca')
                                                    ]),
                                      skm.piece(kb),
                                      skm.piece(m)
                                      ]),

        'random_full_stack': skm.model_pipe([
                                      skm.piece('random_sample'),
                                      skm.piece('std_scale'), 
                                      skm.feature_union([
                                                    skm.piece('agglomeration'), 
                                                    skm.piece(kb),
                                                    skm.piece('pca')
                                                    ]),
                                      skm.piece(kb),
                                      skm.piece(m)
                                      ]),

        'kbest': skm.model_pipe([
                                 skm.piece('std_scale'),
                                 skm.piece(kb),
                                 skm.piece(m)
                                 ]),

        'random' : skm.model_pipe([
                                    skm.piece('random_sample'),
                                    skm.piece('std_scale'),
                                    skm.piece(m)
                                    ]),

        'random_kbest': skm.model_pipe([
                                        skm.piece('random_sample'),
                                        skm.piece('std_scale'),
                                        skm.piece(kb),
                                        skm.piece(m)
                                        ])
    }

    pipe = stack_models[stack_model]
    params = skm.default_params(pipe, bayes_rand=bayes_rand, min_samples=min_samples)
    
    if skm.model_obj == 'quantile':
        if m == 'qr_q': pipe.steps[-1][-1].quantile = alpha
        elif m in ('rf_q', 'knn_q'): pipe.steps[-1][-1].q = alpha
        else: pipe.steps[-1][-1].alpha = alpha

    if stack_model=='random_full_stack': 
        params['random_sample__frac'] = hp.uniform('random_sample__frac', 0.5, 1)
        # params['select_perc__percentile'] = hp.uniform('percentile', 0.5, 1)
        # params['feature_union__agglomeration__n_clusters'] = scope.int(hp.quniform('n_clusters', 2, 10, 1))
        # params['feature_union__pca__n_components'] = scope.int(hp.quniform('n_components', 2, 10, 1))

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


def load_all_pickles(model_output_path, label):
    models = load_pickle(model_output_path, f'{label}_models')
    full_hold = load_pickle(model_output_path, f'{label}_full_hold')
    return models, full_hold

def X_y_stack(full_hold):
    i = 0
    y = None
    y_class = None
    for k, v in full_hold.items():
        if 'million' not in k:
            if i == 0:
                df = v.copy().drop('y_act', axis=1)
                df = df.rename(columns={'pred': k})
            else:
                if 'reg' in k and y is None:
                    y = v[['player', 'team', 'week', 'year', 'y_act']]
                if 'class_80' in k and y_class is None:
                    y_class = v[['player', 'team', 'week', 'year', 'y_act']]
                df_cur = v.rename(columns={'pred': k}).drop('y_act', axis=1)
                df = pd.merge(df, df_cur, on=['player', 'team', 'week','year'])
            i+=1

    X = df.reset_index(drop=True)
    y = pd.merge(X, y, on=['player', 'team', 'week', 'year'])
    y = y[['player', 'team', 'week', 'year', 'y_act']]
    y_class = pd.merge(X, y_class, on=['player', 'team', 'week', 'year'])
    y_class = y_class[['player', 'team', 'week', 'year', 'y_act']]

    return X, y, y_class

def col_ordering(X):
    col_order = [c for c in X.columns if 'reg' in c]
    col_order.extend([c for c in X.columns if 'class' in c])
    col_order.extend([c for c in X.columns if 'quant' in c])
    return X[col_order]

def load_all_stack_pred(model_output_path):

    # load the regregression predictions
    models, full_hold = load_all_pickles(model_output_path, 'all')
    X_stack, y_stack, y_stack_class = X_y_stack(full_hold)

    models_reg = {k: v for k, v in models.items() if 'reg' in k}
    models_class = {k: v for k, v in models.items() if 'class' in k}
    models_quant = {k: v for k, v in models.items() if 'quant' in k}

    return X_stack, y_stack, y_stack_class, models_reg, models_class, models_quant

def load_all_stack_pred_million(model_output_path):
    models_mil, full_hold = load_all_pickles(model_output_path, 'all')
    models_mil = {k: v for k, v in models_mil.items() if 'million' in k}
    i = 0
    y_mil = None
    for k, v in full_hold.items():
        if 'million' in k:
            if i == 0:
                df = v.copy().drop('y_act', axis=1)
                df = df.rename(columns={'pred': k})
            else:
                if y_mil is None:
                    y_mil = v[['player', 'team', 'week', 'year', 'y_act']]
                df_cur = v.rename(columns={'pred': k}).drop('y_act', axis=1)
                df = pd.merge(df, df_cur, on=['player', 'team', 'week','year'])
            i+=1

    X_mil = df.reset_index(drop=True)
    y_mil = pd.merge(X_mil, y_mil, on=['player', 'team', 'week', 'year'])
    y_mil = y_mil[['player', 'team', 'week', 'year', 'y_act']]
    return X_mil, X_mil, y_mil, models_mil



def fit_and_predict(m, df_predict, X, y, proba):
    try:
        m.fit(X,y)

        if proba: cur_predict = m.predict_proba(df_predict[X.columns])[:,1]
        else: cur_predict = m.predict(df_predict[X.columns])
    except:
        cur_predict = []

    return cur_predict

def create_stack_predict(df_predict, models, X, y, proba=False):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
        predictions = Parallel(n_jobs=-1, verbose=0)(delayed(fit_and_predict)(m, df_predict, X, y, proba) for m in ind_models)
        predictions = [p for p in predictions if len(p) > 0]
        predictions = pd.Series(pd.DataFrame(predictions).T.mean(axis=1), name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict

def get_stack_predict_data(df_train, df_predict, df, run_params, 
                           models_reg, models_class, models_quant):

    _, X, y = get_skm(df_train, 'reg', to_drop=run_params['drop_cols'])
    print('Predicting Regression Models')
    X_predict = create_stack_predict(df_predict, models_reg, X, y)

    print('Predicting Class Models')
    for cut in run_params['cuts']:
        df_train_class, df_predict_class = get_class_data(df, cut, run_params)
        _, X, y = get_skm(df_train_class, 'class', to_drop=run_params['drop_cols'])
        X_predict_class = create_stack_predict(df_predict_class, models_class, X, y, proba=True)
        X_predict_class = X_predict_class[[c for c in X_predict_class.columns if str(cut) in c]]
        X_predict = pd.concat([X_predict, X_predict_class], axis=1)

    print('Predicting Quant Models')
    _, X, y = get_skm(df_train, 'quantile', to_drop=run_params['drop_cols'])
    X_predict_quant = create_stack_predict(df_predict, models_quant, X, y)
    X_predict = pd.concat([X_predict, X_predict_quant], axis=1)

    X_predict_player = pd.concat([df_predict[['player', 'team', 'week', 'year']], X_predict], axis=1)

    return X_predict_player, X_predict


def get_stack_predict_data_mil(df_train, df_predict, run_params, models_mil):

    _, X, y = get_skm(df_train, 'class', to_drop=run_params['drop_cols'])
    print('Predicting Million Models')
    X_predict = create_stack_predict(df_predict, models_mil, X, y)
    X_predict = pd.concat([df_predict[['player', 'team', 'week', 'year']], X_predict], axis=1)
    return X_predict


def stack_predictions(X_predict, best_models, final_models, model_obj='reg'):
    
    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):
        
        if model_obj in ('reg', 'quantile'): cur_prediction = np.round(bm.predict(X_predict), 2)
        elif model_obj=='class': cur_prediction = np.round(bm.predict_proba(X_predict)[:,1], 3)
        
        cur_prediction = pd.Series(cur_prediction, name=fm)
        predictions = pd.concat([predictions, cur_prediction], axis=1)

    return predictions


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


def best_average_models(scores, final_models, y_stack, stack_val_pred, predictions, model_obj, min_include = 3):
    
    skm, _, _ = get_skm(df_train, model_obj=model_obj, to_drop=[])
    
    n_scores = []
    models_included = []
    for i in range(len(scores)-min_include+1):
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)[:i+min_include]
        models_included.append(top_n)
        model_idx = np.array(final_models)[top_n]
        
        n_score = skm.custom_score(y_stack, stack_val_pred[model_idx].mean(axis=1))
        n_scores.append(n_score)
        
    print('All Average Scores:', np.round(n_scores, 3))
    best_n = np.argmin(n_scores)
    best_score = n_scores[best_n]
    top_models = models_included[best_n]

    model_idx = np.array(final_models)[top_models]

    print('Top Models:', model_idx)
    best_val = stack_val_pred[model_idx]
    best_predictions = predictions[model_idx]

    return best_val, best_predictions, best_score


def average_stack_models(scores, final_models, y_stack, stack_val_pred, predictions, model_obj, show_plot=True, min_include=3):
    
    best_val, best_predictions, best_score = best_average_models(scores, final_models, y_stack, stack_val_pred, predictions, 
                                                                 model_obj=model_obj, min_include=min_include)
    
    if show_plot:
        mf.show_scatter_plot(best_val.mean(axis=1), y_stack, r2=True)
        if model_obj == 'class':
            show_calibration_curve(y_stack, best_val.mean(axis=1), n_bins=8)

    
    return best_val, best_predictions, best_score


def create_output(output_start, predictions, predictions_class=None, predictions_quantile=None):

    output = output_start.copy()
    output['pred_fp_per_game'] = predictions.mean(axis=1)

    if predictions_class is not None: 
        output['pred_fp_per_game_class'] = predictions_class.mean(axis=1)

    if predictions_quantile is not None:
        output['pred_fp_per_game_quantile'] = predictions_quantile.mean(axis=1)

    output = output.sort_values(by='dk_salary', ascending=False)
    output['dk_rank'] = range(len(output))
    output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

    return output



def std_dev_features(cur_df, model_name, run_params, show_plot=True):

    skm, X, y = get_skm(cur_df, 'reg', to_drop=run_params['drop_cols'])
    pipe, params = get_full_pipe(skm, model_name, stack_model='kbest')

    # fit and append the ADP model
    best_models, _, _ = skm.time_series_cv(pipe, X, y, params, n_iter=run_params['n_iters'], n_splits=run_params['n_splits'],
                                           col_split='game_date', time_split=run_params['cv_time_input'],
                                           bayes_rand='custom_rand', random_seed=1234)

    if model_name=='adp': X = X[list(best_models[0][0].cols)]
    
    for bm in best_models: bm.fit(X, y)
    if show_plot:
        mf.shap_plot(best_models, X, model_num=0)
        plt.show()

    return best_models, X


def add_std_dev_max(df_train, df_predict, output, model_name, run_params, num_cols=10, iso_spline='iso'):

    std_dev_models, X = std_dev_features(df_train, model_name, run_params, show_plot=True)
    sd_cols, df_train, df_predict = mf.get_sd_cols(df_train, df_predict, X, std_dev_models, num_cols=num_cols)
    
    if iso_spline=='iso':
        sd_m, max_m, min_m = get_std_splines(df_train, sd_cols, show_plot=True, k=2, 
                                            min_grps_den=int(df_train.shape[0]*0.08), 
                                            max_grps_den=int(df_train.shape[0]*0.04),
                                            iso_spline=iso_spline)

    elif iso_spline=='spline':
        sd_m, max_m, min_m = get_std_splines(df_train, sd_cols, show_plot=True, k=2, 
                                            min_grps_den=int(df_train.shape[0]*0.1), 
                                            max_grps_den=int(df_train.shape[0]*0.05),
                                            iso_spline=iso_spline)

    output = assign_sd_max(output, df_predict, df_train, sd_cols, sd_m, max_m, min_m, iso_spline)

    return output


def assign_sd_max(output, df_predict, sd_df, sd_cols, sd_m, max_m, min_m, iso_spline):
    
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(sd_df[list(sd_cols.keys())])

    df_predict = df_predict.set_index('player')
    df_predict = df_predict.reindex(index=output['player'])
    df_predict = df_predict.reset_index()

    pred_sd_max = pd.DataFrame(sc.transform(df_predict[list(sd_cols.keys())])) * list(sd_cols.values())
    pred_sd_max = pred_sd_max.mean(axis=1)

    if iso_spline=='spline':
        output['std_dev'] = sd_m(pred_sd_max)
        output['max_score'] = max_m(pred_sd_max)
        output['min_score'] = min_m(pred_sd_max)
    elif iso_spline=='iso':
        output['std_dev'] = sd_m.predict(pred_sd_max)
        output['max_score'] = max_m.predict(pred_sd_max)
        output['min_score'] = min_m.predict(pred_sd_max)

    output.loc[(output.max_score < output.pred_fp_per_game), 'max_score'] = \
        output.loc[(output.max_score < output.pred_fp_per_game), 'pred_fp_per_game'] * 2
    
    return output

def val_std_dev(val_data, metrics={'pred_fp_per_game': 1}, iso_spline='iso', show_plot=True):
        
    sd_max_met = StandardScaler().fit(val_data[list(metrics.keys())]).transform(output[list(metrics.keys())])
    sd_max_met = np.mean(sd_max_met, axis=1)

    if iso_spline=='iso':
        sd_m, max_m, min_m = get_std_splines(val_data, metrics, show_plot=show_plot, k=2, 
                                            min_grps_den=int(val_data.shape[0]*0.1), 
                                            max_grps_den=int(val_data.shape[0]*0.05),
                                            iso_spline=iso_spline)
        output['std_dev'] = sd_m.predict(sd_max_met)
        output['max_score'] = max_m.predict(sd_max_met)
        output['min_score'] = min_m.predict(sd_max_met)

    elif iso_spline=='spline':
        sd_m, max_m, min_m = get_std_splines(val_data, metrics, show_plot=show_plot, k=2, 
                                            min_grps_den=int(val_data.shape[0]*0.1), 
                                            max_grps_den=int(val_data.shape[0]*0.05),
                                            iso_spline=iso_spline)
        output['std_dev'] = sd_m(sd_max_met)
        output['max_score'] = max_m(sd_max_met)
        output['min_score'] = min_m(sd_max_met)
 
    return output

def vegas_points(run_params, metrics={'implied_points_for': 1}, show_plot=False):

    scores = dm.read("SELECT * FROM Scores_Lines", 'Model_Features')
    scores = scores.rename(columns={'team': 'player', 'final_score': 'y_act'})

    output_cols = ['player', 'week', 'year']
    output_cols.extend(list(metrics.keys()))
    output = scores.loc[(scores.week==run_params['set_week']) & \
                        (scores.year==run_params['set_year']),
                        output_cols]

    scores = scores[(scores.year < run_params['set_year']) | \
                    ((scores.year==run_params['set_year']) & \
                    (scores.week < run_params['set_week']))].reset_index(drop=True)

    sd_max_met = StandardScaler().fit(scores[list(metrics.keys())]).transform(output[list(metrics.keys())])
    sd_max_met = np.mean(sd_max_met, axis=1)

    sd_m, max_m, min_m = get_std_splines(scores, metrics, show_plot=show_plot, k=2, 
                                        min_grps_den=int(scores.shape[0]*0.08), 
                                        max_grps_den=int(scores.shape[0]*0.04),
                                        iso_spline='spline')

    output['std_dev'] = sd_m(sd_max_met)
    output['max_score'] = max_m(sd_max_met)
    output['min_score'] = min_m(sd_max_met)
    output = output.rename(columns={'player': 'team'})
    return output


def add_actual(df):

    if run_params['rush_pass'] != '': rush_pass = f"_{run_params['rush_pass']}"
    else: rush_pass = ''

    actual_pts = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
        if pos=='Defense': pl = 'defTeam'
        else: pl = 'player'

        actual_pts_cur = dm.read(f'''SELECT {pl} player, week, season year, fantasy_pts{rush_pass} actual_pts
                                     FROM {pos}_Stats 
                                     WHERE week>={run_params['set_week']} 
                                           and week < 18
                                           and season={run_params['set_year']
                                           }''', 'FastR')
        actual_pts = pd.concat([actual_pts, actual_pts_cur], axis=0)
    
    if len(actual_pts) > 0:
        df = pd.merge(df, actual_pts, on=['player', 'week', 'year'], how='left')
    return df


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


def add_sal_columns(df, df_sal):
    
    df = pd.merge(df, df_sal[['player', 'week', 'year', 'dk_salary']], on=['player', 'week', 'year'])

    for c in df.columns:
        if 'expert' in c: df[c+'_salary'] = df[c] * df.dk_salary
        if 'rank' in c: df[c+'_salary'] = df[c] * df.dk_salary
        if 'proj' in c: df[c+'_salary'] = df[c] / df.dk_salary
        if 'ffa' in c: df[c+'_salary'] = df[c] / df.dk_salary
        if 'reg' in c: df[c+'_salary'] = df[c] / df.dk_salary
        if 'quant' in c: df[c+'_salary'] = df[c] / df.dk_salary
    
    return df


def predict_million_df(df, run_params):

    df = select_main_slate_teams(df)

    df = df.drop('y_act', axis=1)
    top_players = dm.read("SELECT player, week, year, y_act FROM Top_Players", "DK_Results")

    df = pd.merge(df, top_players, on=['player', 'week', 'year'], how='left')
    df = df.fillna({'y_act': 0})

    df_train_mil, df_predict_mil, _, min_samples_mil = train_predict_split(df, run_params)

    return df_train_mil, df_predict_mil, min_samples_mil, run_params





def save_mil_data(X_mil, y_mil, best_val_mil, df_predict_mil, best_predictions_mil,  run_params):
    test_output = pd.concat([df_predict_mil[['player', 'team', 'week', 'year']], 
                            pd.Series(best_predictions_mil.mean(axis=1), name='pred_fp_per_game_class')], 
                            axis=1).sort_values(by='pred_fp_per_game_class', ascending=False)

    save_val_to_db(X_mil, y_mil, best_val_mil, run_params, table_name='Model_Validations_Million')
    save_prob_to_db(test_output, run_params, 'Predicted_Million')

#-----------------------
# Saving validations
#-----------------------

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def validation_compare_df(X, y, best_val):

    best_val = pd.Series(best_val.mean(axis=1), name='pred_fp_per_game')
    val_compare = pd.concat([X[['player', 'team', 'week', 'year']], y, best_val], axis=1)
    
    return val_compare


def save_val_to_db(X, y, best_val, run_params, table_name):

    df = validation_compare_df(X, y, best_val)

    df['pos'] = set_pos
    df['pred_version'] = vers
    df['ensemble_vers'] = ensemble_vers
    df['std_dev_type'] = std_dev_type
    df['model_type'] = model_type
    df['set_week'] = run_params['set_week']
    df['set_year'] = run_params['set_year']

    df = df[['player', 'team', 'year', 'week',  'y_act', 'pred_fp_per_game', 'pos',
             'pred_version', 'ensemble_vers', 'std_dev_type', 'model_type', 'set_week', 'set_year']]

    del_str = f'''pos='{set_pos}' 
                  AND pred_version='{vers}'
                  AND ensemble_vers='{ensemble_vers}' 
                  AND std_dev_type='{std_dev_type}'
                  AND set_week={run_params['set_week']} 
                  AND set_year={run_params['set_year']}
                  AND model_type='{model_type}'
                '''
    
 
    dm.delete_from_db('Validations', table_name, del_str, create_backup=False)
    dm.write_to_db(df, 'Validations', table_name, 'append')

  


def save_prob_to_db(output, run_params, table_name):

    
    df = output[['player', 'team', 'week', 'year', 'pred_fp_per_game_class']].copy()
    df = df.rename(columns={'week': 'set_week', 'year': 'set_year', 'pred_fp_per_game_class': 'pred_fp_per_game'})
    df['std_dev'] = df.pred_fp_per_game * 0.25
    df['min_score'] = 0
    df['max_score'] = df.pred_fp_per_game * 1.5
    df['max_score'] = df.max_score.apply(lambda x: np.min([x, 1]))

    df['pos'] = set_pos
    df['pred_version'] = vers
    df['ensemble_vers'] = ensemble_vers
    df['std_dev_type'] = std_dev_type
    df['model_type'] = model_type
    df['week'] = df.set_week
    df['year'] = df.set_year

    df = df[['player', 'set_week', 'set_year', 'pos', 'pred_fp_per_game', 'std_dev', 'min_score', 'max_score',
             'pred_version', 'ensemble_vers', 'model_type', 'std_dev_type', 'week', 'year']]

    del_str = f'''pos='{set_pos}' 
                  AND pred_version='{vers}'
                  AND ensemble_vers='{ensemble_vers}' 
                  AND std_dev_type='{std_dev_type}' 
                  AND set_week={run_params['set_week']} 
                  AND set_year={run_params['set_year']}
                  AND model_type='{model_type}'
                '''
    dm.delete_from_db('Simulation', table_name, del_str, create_backup=False)
    dm.write_to_db(df, 'Simulation', table_name, 'append')


def save_output_to_db(output, run_params):

    output['pos'] = set_pos
    output['version'] = vers
    output['ensemble_vers'] = ensemble_vers
    output['std_dev_type'] = std_dev_type
    output['model_type'] = model_type
    output['week'] = run_params['set_week']
    output['year'] = run_params['set_year']
    output['rush_pass'] = run_params['rush_pass']

    output = output[['player', 'dk_salary', 'pred_fp_per_game', 'std_dev',
                     'dk_rank', 'pos', 'version', 'model_type', 'max_score', 'min_score',
                      'week', 'year', 'ensemble_vers', 'std_dev_type', 'rush_pass']]

    del_str = f'''pos='{set_pos}' 
                AND version='{vers}'
                AND ensemble_vers='{ensemble_vers}' 
                AND std_dev_type='{std_dev_type}'
                AND week={run_params['set_week']} 
                AND year={run_params['set_year']}
                AND model_type='{model_type}'
                AND rush_pass="{run_params['rush_pass']}"
                '''
    dm.delete_from_db('Simulation', 'Model_Predictions', del_str, create_backup=False)
    dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')


#----------------
# New Functions
#----------------

def get_newest_folder(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    newest_folder = max(folders, key=lambda f: os.path.getctime(os.path.join(path, f)))
    return os.path.join(path, newest_folder)

def get_newest_folder_with_keywords(path, keywords, ignore_keywords=None, req_fname=None):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    # Apply ignore_keywords if provided
    if ignore_keywords:
        folders = [f for f in folders if not any(ignore_keyword in f for ignore_keyword in ignore_keywords)]
    
    matching_folders = [f for f in folders if all(keyword in f for keyword in keywords)]
    
    if req_fname is not None:
        matching_folders = [f for f in matching_folders if os.path.isfile(os.path.join(path, f, req_fname))]
    
    if not matching_folders:
        return None
    
    newest_folder = max(matching_folders, key=lambda f: os.path.getctime(os.path.join(path, f)))
    return os.path.join(path, newest_folder)


def get_proba_adp_coef(model_obj, final_m, run_params):
    if model_obj == 'class': proba = True
    else: proba = False

    if model_obj in ('class', 'quantile'): run_adp = False
    else: run_adp = True

    if 'gbmh' in final_m or 'knn' in final_m or 'full_stack' in run_params['stack_model']: print_coef = False
    else: print_coef = run_params['print_coef']

    return proba, run_adp, print_coef


def get_trials(fname, final_m, bayes_rand):

    newest_folder = get_newest_folder(f"{root_path}/Model_Outputs/")
    keep_words = [set_pos, model_type, vers]
    drop_words = [f"_week{run_params['set_week']}_"]
    recent_save = get_newest_folder_with_keywords(newest_folder, keep_words, drop_words, f'{fname}.p')

    if recent_save is not None and bayes_rand=='bayes': 
        try:
            trials = load_pickle(recent_save, fname)
            trials = trials['trials'][final_m]
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

def run_stack_models(fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params):

    print(f'\n{final_m}')

    min_samples = int(len(y_stack)/10)
    proba, run_adp, print_coef = get_proba_adp_coef(model_obj, final_m, run_params)

    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, to_drop=[])
    pipe, params = get_full_pipe(skm, final_m, stack_model=run_params['stack_model'], alpha=alpha, 
                                 min_samples=min_samples, bayes_rand=run_params['opt_type'])
    
    trials = get_trials(fname, final_m, run_params['opt_type'])

    best_model, stack_scores, stack_pred, trial = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                n_iter=run_params['n_iters'], alpha=alpha,
                                                                trials=trials, bayes_rand=run_params['opt_type'],
                                                                run_adp=run_adp, print_coef=print_coef,
                                                                proba=proba, num_k_folds=run_params['num_k_folds'],
                                                                random_state=(i*2)+(i*7))
    
    return best_model, stack_scores, stack_pred, trial

def get_func_params(model_obj):

    model_list = {
        'reg': ['rf', 'gbm', 'gbmh', 'huber', 'xgb', 'lgbm', 'knn', 'ridge', 'lasso', 'bridge'],
        'class': ['rf_c', 'gbm_c', 'gbmh_c', 'xgb_c','lgbm_c', 'knn_c', 'lr_c'],
        'quantile': ['qr_q', 'gbm_q', 'rf_q', 'lgbm_q', 'knn_q']
    }
    if model_obj=='quantile': alpha=0.8
    else: alpha=''

    func_params = [[m, i, model_obj, alpha] for i, m  in enumerate(model_list[model_obj])]

    return model_list[model_obj], func_params

def unpack_results(model_list, results):
    best_models = [r[0] for r in results]
    scores = [r[1]['stack_score'] for r in results]
    stack_val_pred = pd.concat([pd.Series(r[2]['stack_pred'], name=m) for r,m in zip(results, model_list)], axis=1)
    trials = {m: r[3] for m, r in zip(model_list, results)}
    return best_models, scores, stack_val_pred, trials
    
def cleanup_X_y(X, y):
    X_player = X[['player', 'team', 'week', 'year']].copy()
    X = X.drop(['player', 'team', 'week', 'year'], axis=1).dropna(axis=0)
    y = y[y.index.isin(X.index)].y_act
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    X = col_ordering(X)
    X_player = pd.concat([X_player, X], axis=1)
    return X_player, X, y


def save_stack_runs(model_output_path, fname, best_models, scores, stack_val_pred, trials):
    stack_out = {}
    stack_out['best_models'] = best_models
    stack_out['scores'] = scores
    stack_out['stack_val_pred'] = stack_val_pred
    stack_out['trials'] = trials
    save_pickle(stack_out, model_output_path, fname, protocol=-1)

def load_stack_runs(model_output_path, fname):

    stack_in = load_pickle(model_output_path, fname)
    return stack_in['best_models'], stack_in['scores'], stack_in['stack_val_pred']


def load_run_models(run_params, X_stack, y_stack, X_predict, model_obj, alpha=None, is_million=False):
    
    if alpha is not None: alpha_label = alpha*100
    else: alpha_label = ''

    if is_million: model_obj_label = 'million'
    else: model_obj_label = model_obj

    path = run_params['model_output_path']
    fname = f"{model_obj_label}{alpha_label}_{run_params['ensemble_vers']}"    
    model_list, func_params = get_func_params(model_obj)
    
    if os.path.exists(f"{path}/{fname}.p"):
        best_models, scores, stack_val_pred = load_stack_runs(path, fname)
    
    else:
        
        results = Parallel(n_jobs=-1, verbose=50)(
                        delayed(run_stack_models)
                        (fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params) 
                        for final_m, i, model_obj, alpha in func_params
                        )

        best_models, scores, stack_val_pred, trials = unpack_results(model_list, results)
        save_stack_runs(path, fname, best_models, scores, stack_val_pred, trials)
        
    predictions = stack_predictions(X_predict, best_models, model_list, model_obj=model_obj)
    best_val, best_predictions, _ = average_stack_models(scores, model_list, y_stack, stack_val_pred, 
                                                         predictions, model_obj=model_obj, 
                                                         show_plot=run_params['show_plot'], 
                                                         min_include=run_params['min_include'])

    return best_val, best_predictions


def join_stats_mil(X_stack_mil, X_stack_player, X_predict_mil, X_predict_player):
    X_stack_mil = pd.merge(X_stack_mil, X_stack_player, on=['player', 'team', 'week', 'year'], how='left')
    X_predict_mil = pd.merge(X_predict_mil, X_predict_player,  on=['player', 'team', 'week', 'year'], how='left')
    return X_stack_mil, X_predict_mil


def create_final_val_df(X_stack_player, y_stack, best_val_reg, best_val_class, best_val_quant):
    df_val_final = pd.concat([X_stack_player[['player', 'team', 'week', 'year']], 
                              pd.Series(best_val_reg.mean(axis=1), name='pred_fp_per_game'),
                              pd.Series(best_val_class.mean(axis=1), name='pred_fp_per_game_class'),
                              pd.Series(best_val_quant.mean(axis=1), name='pred_fp_per_game_quantile'),
                              y_stack], axis=1)
    # df_val_final = pd.merge(df_val_final, y_stack, on=['player', 'team', 'week', 'year'])
    return df_val_final


def create_mil_output(df_predict_mil, best_predictions_mil):
    output_mil = pd.concat([df_predict_mil[['player', 'team', 'week', 'year']], 
                            pd.Series(best_predictions_mil.mean(axis=1), name='pred_mil')], axis=1)
    try:
        actuals = dm.read("SELECT * FROM Top_Players", 'DK_Results')
        mil_results = pd.merge(output_mil, actuals[['player', 'week', 'year', 'y_act']], 
                            on=['player', 'week', 'year'], how='left').fillna(0)
        print('Showing Actual Results')
        display(mil_results.sort_values(by='pred_mil', ascending=False).iloc[:50])
        show_calibration_curve(mil_results['y_act'], mil_results['pred_mil'], n_bins=8)
    except:
        display(output_mil.sort_values(by='pred_mil', ascending=False).iloc[:50])

    return output_mil
            

def display_output(output, show_plot=True):
     
    try:  
        output = add_actual(output)
        output = output[output.week!=18].reset_index(drop=True)
        print('Showing Actual Results')
        print(output.loc[:50, ['player', 'week', 'year', 'dk_salary', 'dk_rank', 'pred_fp_per_game', 'pred_fp_per_game_class',
                                'pred_fp_per_game_quantile', 'actual_pts', 'std_dev', 'min_score', 'max_score']])
                        
        if show_plot: 
            mf.show_scatter_plot(output.pred_fp_per_game, output.actual_pts, r2=True)
            skm_score, _, _ = get_skm(df_train, model_obj='reg', to_drop=[])
            print('Score:', np.round(skm_score.custom_score(output.pred_fp_per_game, output.actual_pts),2))
        
        output = output.drop('actual_pts', axis=1)
    except:
        print(output.loc[:50, ['player', 'dk_salary','dk_rank', 'pred_fp_per_game', 'pred_fp_per_game_class', 
                            'pred_fp_per_game_quantile', 'std_dev', 'min_score', 'max_score']])

#%%
#==========
# General Setting
#==========

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

import yaml

def process_config(config):
    config['pred_vers'] = config['pred_vers'].format(**config['pred_params'])
    config['ensemble_vers'] = config['ensemble_vers'].format(**config['ensemble_params'])
    config['std_dev_type'] = config['std_dev_type'].format(**config['std_dev_params'])

    return config

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Assuming the config file is in the same directory as the Python script
config_file = f'{root_path}/Scripts/config.yaml'
config = read_config(config_file)
config = process_config(config)

globals().update(config)

#%%
#---------------
# Settings
#---------------

run_params = {
    
    # set year and week to analyze
    'set_year': 2022,

    # set beginning of validation period
    'val_year_min': 2020,
    'val_week_min': 1,

    'cuts': [33, 80, 95],

    'stack_model': 'random_kbest',

    # opt params
    'opt_type': 'bayes',
    'n_iters': 50,
    'n_splits': 5,
    'num_k_folds': 3,
    'show_plot': True,
    'print_coef': True,
    'min_include': 2,

    # other parameters
    'use_sample_weight': False,
    
    'met': 'y_act',

   
}

r2_wt = 0
sera_wt = 1
mse_wt = 0
brier_wt = 1
matt_wt = 0

# set the model version
set_weeks = [6]

pred_versions = len(set_weeks)*['sera1_rsq0_brier1_matt0_bayes']

ensemble_versions = len(set_weeks)*['random_kbest_sera1_rsq0_mse0_include2_kfold3']

std_dev_types = ['spline_pred_class80_q80_matt0_brier1_kfold3',
                 'spline_pred_class80_matt0_brier1_kfold3',
                 'spline_pred_q80_matt0_brier1_kfold3',
                 'spline_class80_q80_matt0_brier1_kfold3']
std_dev_type = std_dev_types[0]

for w, vers, ensemble_vers in zip(set_weeks, pred_versions, ensemble_versions):

    run_params['set_week'] = w
    run_params['ensemble_vers'] = ensemble_vers
    runs = [
        # ['QB', 'full_model', ''],
        # ['RB', 'full_model', ''],
        # ['WR', 'full_model', ''],
        # ['TE', 'full_model', ''],
        ['Defense', 'full_model', ''],
        # ['QB', 'backfill', ''],
        # ['RB', 'backfill', ''],
        # ['WR', 'backfill', ''],
        # ['TE', 'backfill', '']
    ]

    for set_pos, model_type, rush_pass in runs:

        run_params['rush_pass'] = rush_pass

        # load data and filter down
        pkey, run_params, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
        df, run_params = load_data(model_type, set_pos, run_params)
        df, run_params = create_game_date(df, run_params)
        df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

        # set up blank dictionaries for all metrics
        out_reg, out_class, out_quant, out_million = {}, {}, {}, {}

        #------------
        # Run the Stacking Models and Generate Output
        #------------

        # get the training data for stacking and prediction data after stacking
        X_stack, y_stack, y_stack_class, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path)
        X_predict_player, X_predict = get_stack_predict_data(df_train, df_predict, df, run_params, models_reg, models_class, models_quant)

        # cleanup the X and y datasets
        X_stack_player, X_stack, y_stack = cleanup_X_y(X_stack, y_stack)
        _, _, y_stack_class = cleanup_X_y(X_stack_player, y_stack_class)

        # run the class, quant, and reg models
        best_val_class, best_predictions_class = load_run_models(run_params, X_stack, y_stack_class, X_predict, 'class')
        best_val_quant, best_predictions_quant = load_run_models(run_params, X_stack, y_stack, X_predict, 'quantile', alpha=0.8)
        best_val_reg, best_predictions_reg = load_run_models(run_params, X_stack, y_stack, X_predict, 'reg')

        save_val_to_db(X_stack_player, y_stack_class, best_val_class, run_params, table_name='Model_Validations_Class')
        save_val_to_db(X_stack_player, y_stack_class, best_val_reg, run_params, table_name='Model_Validations')

        # create the output and add standard deviations / max score datasets
        df_val_final = create_final_val_df(X_stack_player, y_stack, best_val_reg, best_val_class, best_val_quant)
        output = create_output(output_start, best_predictions_reg, best_predictions_class, best_predictions_quant)

        # loop through std dev types and display / save output
        for i, std_dev_type in enumerate(std_dev_types):     
            print('Standard Deviation Type:', std_dev_type)       
            metrics = metrics_dict[std_dev_type]
            if i==0: sd_plot = True
            else: sd_plot = False
            output = val_std_dev(df_val_final, metrics=metrics, iso_spline='spline', show_plot=sd_plot)
            if i==0: display_output(output, run_params['show_plot'])
            save_output_to_db(output, run_params)

        #-------------
        # Running the million dataset
        #-------------

        df_train_mil, df_predict_mil, _, run_params = predict_million_df(df, run_params)
        X_mil_player, X_stack_mil, y_stack_mil, models_mil = load_all_stack_pred_million(model_output_path)
        X_predict_mil = get_stack_predict_data_mil(df_train_mil, df_predict_mil, run_params, models_mil)
        X_stack_mil, X_predict_mil = join_stats_mil(X_stack_mil, X_stack_player, X_predict_mil, X_predict_player)

        X_stack_mil = add_sal_columns(X_stack_mil, df_train_mil)
        X_predict_mil = add_sal_columns(X_predict_mil, df_predict_mil)

        X_predict_mil = X_predict_mil.drop(['player', 'team', 'week', 'year'], axis=1)
        X_stack_mil = X_stack_mil.drop(['player', 'team', 'week', 'year'], axis=1)
        y_stack_mil = y_stack_mil.y_act
        # class metrics
        best_val_mil, best_predictions_mil = load_run_models(run_params, X_stack_mil, y_stack_mil, X_predict_mil, 'class', is_million=True)

        output_mil = create_mil_output(df_predict_mil, best_predictions_mil)
        save_mil_data(X_mil_player, y_stack_mil, best_val_mil, df_predict_mil, best_predictions_mil, run_params)

    #---------------
    # Save vegas points and std dev
    #---------------

    vp = vegas_points(run_params, metrics={'implied_points_for': 1}, show_plot=run_params['show_plot'])
    dm.delete_from_db('Simulation', 'Vegas_Points', f"week={run_params['set_week']} AND year={run_params['set_year']}", create_backup=False)
    dm.write_to_db(vp, 'Simulation', 'Vegas_Points', 'append')
    print('All Runs Finished')

#%%
x=2

# %%
trials = load_pickle('/Users/mborysia/Documents/Github/Daily_Fantasy//Model_Outputs/2022/QB_year2022_week3_full_modelsera1_rsq0_brier1_matt0_bayes/', 
                     'reg_sera1_rsq0_include2_kfold3')['trials']
trials['lgbm'].tids

#%%
run_params['stack_model'] = 'full_stack'
fname = 'reg_sera1_rsq0_include2_kfold3_test'
final_m = 'lgbm'
i = 0
model_obj = 'reg'
alpha = None
# run_stack_models(fname, final_m, i, model_obj, alpha, X_stack_mil, y_stack_mil, run_params) 

min_samples = int(len(y_stack)/10)
proba, run_adp, print_coef = get_proba_adp_coef(model_obj, final_m, run_params)

skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, to_drop=[])
pipe, params = get_full_pipe(skm, final_m, stack_model=run_params['stack_model'], alpha=alpha, 
                                min_samples=min_samples, bayes_rand=run_params['opt_type'])

trials = get_trials(fname, final_m, run_params['opt_type'])
trials = update_trials_params(trials, final_m, params, pipe)

best_model, stack_scores, stack_pred, trial = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                            n_iter=50, alpha=alpha,
                                                            trials=trials, bayes_rand=run_params['opt_type'],
                                                            run_adp=run_adp, print_coef=print_coef,
                                                            proba=proba, num_k_folds=run_params['num_k_folds'],
                                                            random_state=(i*2)+(i*7))
best_model
# %%
models, full_hold = load_all_pickles(model_output_path, 'all')
full_hold['reg_adp'].plot.scatter(x='pred', y='y_act')
# %%
pd.concat([X_stack, y_stack.reset_index(drop=True)], axis=1).plot.scatter(x='reg_adp', y='y_act')
# %%
best_models = load_pickle('/Users/mborysia/Documents/Github/Daily_Fantasy//Model_Outputs/2022/QB_year2022_week1_full_modelsera1_rsq0_brier1_matt0_bayes/', 
                     'reg_sera1_rsq0_include2_kfold3')['best_models']
best_models[0].predict(X_predict)
# %%
_, X, y = get_skm(df_train, 'quantile', to_drop=run_params['drop_cols'])
X_predict_quant = create_stack_predict(df_predict, models_quant, X, y)

# %%

fname = 'million_sera1_rsq0_include2_kfold3'
final_m = 'lgbm'
i = 0
model_obj = 'class'
alpha = None
is_million=True

if alpha is not None: alpha_label = alpha*100
else: alpha_label = ''

if is_million: model_obj_label = 'million'
else: model_obj_label = model_obj

path = run_params['model_output_path']
fname = f"{model_obj_label}{alpha_label}_{run_params['ensemble_vers']}"    
model_list, func_params = get_func_params(model_obj)

if os.path.exists(f"{path}/{fname}.p"):
    print('Loading Existing Model Runs')
    best_models, scores, stack_val_pred = load_stack_runs(path, fname)

else:
    
    results = Parallel(n_jobs=-1, verbose=50)(
                    delayed(run_stack_models)
                    (fname, final_m, i, model_obj, alpha, X_stack_mil, y_stack_mil, run_params) 
                    for final_m, i, model_obj, alpha in func_params
                    )

    best_models, scores, stack_val_pred, trials = unpack_results(model_list, results)
    save_stack_runs(path, fname, best_models, scores, stack_val_pred, trials)

predictions = stack_predictions(X_predict, best_models, model_list, model_obj=model_obj)
best_val, best_predictions, _ = average_stack_models(scores, model_list, y_stack_class, stack_val_pred, 
                                                        predictions, model_obj=model_obj, show_plot=show_plot, 
                                                        min_include=min_include)
# %%
