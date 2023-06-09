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

    df_predict = df[df.game_date >= run_params['train_time_split']].reset_index(drop=True)
    output_start = df_predict[['player', 'team', 'week', 'year', 'dk_salary']].copy().drop_duplicates()

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.game_date < run_params['cv_time_input']].shape[0])  
    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, output_start, min_samples


def get_class_data(df, cut, run_params):

    # set up the training and prediction datasets for the classification 
    df_train_class = df[df.game_date < run_params['train_time_split']].reset_index(drop=True)
    df_predict_class = df[df.game_date >= run_params['train_time_split']].reset_index(drop=True)

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


def get_full_pipe(skm, m, alpha=None, stack_model=False, std_model=False, min_samples=10):

    if m == 'adp':
        
        # set up the ADP model pipe
        pipe = skm.model_pipe([skm.piece('feature_select'), 
                               skm.piece('std_scale'), 
                               skm.piece('k_best'),
                               skm.piece('lr')])

    elif stack_model and full_stack_features:
        if skm.model_obj=='class': kb = 'k_best_c'
        else: kb = 'k_best'

        pipe = skm.model_pipe([skm.piece('random_sample'),
                                skm.piece('std_scale'), 
                                skm.piece('select_perc'),
                                skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece(kb),
                                                skm.piece('pca')
                                                ]),
                                skm.piece(kb),
                                skm.piece(m)])

    elif stack_model:
        if skm.model_obj=='class': kb = 'k_best_c'
        else: kb = 'k_best'

        pipe = skm.model_pipe([
                            skm.piece('random_sample'),
                            skm.piece('std_scale'), 
                            skm.piece(kb),
                            skm.piece(m)
                        ])

    elif std_model:
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

    # get the params for the current pipe and adjust if needed
    params = skm.default_params(pipe, 'rand')
    if m=='adp': 
        params['feature_select__cols'] = [
                                            ['game_date', 'year', 'week', 'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
                                            ['year', 'week',  'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank'],
                                            [ 'ProjPts', 'dk_salary', 'fd_salary', 'projected_points', 'fantasyPoints', 'ffa_points', 'avg_proj_points', 'fc_proj_fantasy_pts_fc', 'log_fp_rank', 'log_avg_proj_rank']
                                        ]
        params['k_best__k'] = range(1, 14)
    
    if skm.model_obj == 'quantile':
        if m == 'qr_q': pipe.steps[-1][-1].quantile = alpha
        elif m in ('rf_q', 'knn_q'): pipe.steps[-1][-1].q = alpha
        else: pipe.steps[-1][-1].alpha = alpha

    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)
    if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)
    if m=='knn_q': params['knn_q__n_neighbors'] = range(1, min_samples-1)
    
    if stack_model and full_stack_features: 
        params['random_sample__frac'] = np.arange(0.6, 1.05, 0.05)
        params['select_perc__percentile'] = range(60, 105, 5)
        params['feature_union__agglomeration__n_clusters'] = range(2, 10, 1)
        params[f'feature_union__{kb}__k'] = range(5, 20, 2)
        params['feature_union__pca__n_components'] = range(2, 10, 1)
        params[f'{kb}__k'] = range(1, 30)
    
    elif stack_model:
        params['random_sample__frac'] = np.arange(0.3, 1, 0.05)
        params[f'{kb}__k'] = range(1, 30)

    return pipe, params

def load_all_stack_pred(model_output_path, stack_cut=80, prefix=''):

    # load the regregression predictions
    pred, actual, models_reg, _, full_hold_reg = mf.load_all_pickles(model_output_path, prefix+'reg')
    X_stack, y_stack, _ = mf.X_y_stack('reg', full_hold_reg, pred, actual)

    # load the class predictions
    pred_class, actual_class, models_class, _, full_hold_class = mf.load_all_pickles(model_output_path, prefix+'class')
    X_stack_class, _, _ = mf.X_y_stack('class', full_hold_class, pred_class, actual_class)
    _, _, _, _, y_stack_class = mf.load_all_pickles(model_output_path, 'class')

    if prefix=='': y_stack_class = y_stack_class[f'class_lr_c_{stack_cut}'].y_act.reset_index(drop=True)

    # load the quantile predictions
    pred_quant, actual_quant, models_quant, _, full_hold_quant = mf.load_all_pickles(model_output_path, prefix+'quant')
    X_stack_quant, _, df_labels = mf.X_y_stack('quant', full_hold_quant, pred_quant, actual_quant)

    # concat all the predictions together
    X_stack = pd.concat([df_labels[['player', 'team', 'week', 'year']], X_stack, X_stack_quant, X_stack_class], axis=1)
    
    if prefix=='':
        y_stack = pd.concat([df_labels[['player', 'team', 'week', 'year']], y_stack], axis=1)
        y_stack_class = pd.concat([df_labels[['player', 'team', 'week', 'year']], y_stack_class], axis=1)

    return X_stack, y_stack, y_stack_class, models_reg, models_class, models_quant


def run_stack_models(final_m, i, X_stack, y_stack, best_models, scores, 
                     stack_val_pred, model_obj='reg', alpha=None, run_adp=True, show_plots=True,
                     num_k_folds=3, print_coef=True, grp=None):

    print(f'\n{final_m}')

    if model_obj == 'class': proba = True
    else: proba = False

    if model_obj in ('class', 'quantile'): run_adp = False
    else: run_adp = True

    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, to_drop=[])
    pipe, params = get_full_pipe(skm, final_m, stack_model=True, alpha=alpha)
    
    if 'gbmh' in final_m or 'knn' in final_m: print_coef = False
    else: print_coef = print_coef

    best_model, stack_scores, stack_pred = skm.best_stack(pipe, params,
                                                          X_stack, y_stack, n_iter=run_params['n_iters'], 
                                                          run_adp=run_adp, print_coef=print_coef,
                                                          sample_weight=False, proba=proba,
                                                          num_k_folds=num_k_folds, alpha=alpha,
                                                          random_state=(i*12)+(i*17), grp=None)

    best_models.append(best_model)
    scores.append(stack_scores['stack_score'])
    stack_val_pred = pd.concat([stack_val_pred, pd.Series(stack_pred['stack_pred'], name=final_m)], axis=1)

    if show_plots:
        mf.show_scatter_plot(stack_pred['stack_pred'], stack_pred['y'], r2=True)


    return best_models, scores, stack_val_pred


def fit_and_predict(m, df_predict, X, y, proba):
    
    m.fit(X,y)

    if proba: cur_predict = m.predict_proba(df_predict[X.columns])[:,1]
    else: cur_predict = m.predict(df_predict[X.columns])

    return cur_predict

def create_stack_predict(df_predict, models, X, y, proba=False):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
        predictions = Parallel(n_jobs=-1, verbose=0)(delayed(fit_and_predict)(m, df_predict, X, y, proba) for m in ind_models)
        predictions = pd.Series(pd.DataFrame(predictions).T.mean(axis=1), name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict

def get_stack_predict_data(df_train, df_predict, df, run_params, 
                           models_reg, models_class, models_quant):

    _, X, y = get_skm(df_train, 'reg', to_drop=run_params['drop_cols'])
    print('Predicting Regression Models')
    X_predict = create_stack_predict(df_predict, models_reg, X, y)

    print('Predicting Quant Models')
    X_predict_quant = create_stack_predict(df_predict, models_quant, X, y)
    X_predict = pd.concat([X_predict, X_predict_quant], axis=1)

    print('Predicting Class Models')
    for cut in run_params['cuts']:
        df_train_class, df_predict_class = get_class_data(df, cut, run_params)
        _, X, y = get_skm(df_train_class, 'class', to_drop=run_params['drop_cols'])
        X_predict_class = create_stack_predict(df_predict_class, models_class, X, y, proba=True)
        X_predict_class = X_predict_class[[c for c in X_predict_class.columns if str(cut) in c]]
        X_predict = pd.concat([X_predict, X_predict_class], axis=1)

    X_predict_player = pd.concat([df_predict[['player', 'team', 'week', 'year']], X_predict], axis=1)

    return X_predict_player, X_predict


def stack_predictions(X_predict, best_models, final_models, model_obj='reg'):
    
    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):
        
        if model_obj in ('reg', 'quantile'): cur_prediction = np.round(bm.predict(X_predict), 2)
        elif model_obj=='class': cur_prediction = np.round(bm.predict_proba(X_predict)[:,1], 3)
        
        cur_prediction = pd.Series(cur_prediction, name=fm)
        predictions = pd.concat([predictions, cur_prediction], axis=1)

    return predictions


def add_rush_pass(X_stack, X_predict, y_stack, y_stack_class, run_params):

    X_stack_rush, X_predict_rush, run_params = pull_X_rush_pass('rush', run_params)
    X_stack_pass, X_predict_pass, run_params = pull_X_rush_pass('pass', run_params)

    X_stack = pd.merge(X_stack, X_stack_rush, on=['player', 'week', 'year'])
    X_stack = pd.merge(X_stack, X_stack_pass, on=['player', 'week', 'year'])

    y_stack = pd.merge(X_stack[['player', 'week', 'year']], y_stack, on=['player', 'week', 'year'])
    y_stack_class = pd.merge(X_stack[['player', 'week', 'year']], y_stack_class, on=['player', 'week', 'year'])

    X_predict = pd.concat([X_predict, X_predict_rush, X_predict_pass], axis=1)

    return X_stack, X_predict, y_stack, y_stack_class, run_params


def pull_X_rush_pass(rp, run_params):

    run_params['rush_pass'] = rp

    # load data and filter down
    _, _, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
    df, run_params = load_data(model_type, set_pos, run_params)
    df, run_params = create_game_date(df, run_params)
    df_train, df_predict, _, _ = train_predict_split(df, run_params)

    # get the training data for stacking and prediction data after stacking
    X_stack_rush, _, _, models_reg_rush, models_class_rush, models_quant_rush = \
        load_all_stack_pred(model_output_path, 80, prefix=rp)
    X_predict_rush = get_stack_predict_data(df_train, df_predict, df, run_params, 
                                            models_reg_rush, models_class_rush, models_quant_rush) 

    run_params['rush_pass'] = ''
    
    return X_stack_rush, X_predict_rush, run_params


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
    pipe, params = get_full_pipe(skm, model_name, std_model=True)

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


def X_y_stack_mil(df, run_params):
    df_train_mil, df_predict_mil, _, run_params = predict_million_df(df, run_params)

    # load the regregression predictions
    pred, actual, models_mil, _, full_hold_mil = mf.load_all_pickles(model_output_path, 'million')
    X_stack_mil, y_stack_mil, df_labels = mf.X_y_stack('mil', full_hold_mil, pred, actual)
    X_stack_mil = pd.concat([df_labels[['player', 'team', 'week', 'year']], X_stack_mil], axis=1)

    _, X, y = get_skm(df_train_mil, 'class', to_drop=run_params['drop_cols'])
    X_predict_mil = create_stack_predict(df_predict_mil, models_mil, X, y, proba=True)
    X_predict_mil = pd.concat([df_predict_mil[['player', 'team', 'week', 'year']], X_predict_mil], axis=1)

    return df_predict_mil, X_stack_mil, y_stack_mil, X_predict_mil


def save_mil_data(df_predict_mil, best_predictions_mil, best_val_mil, run_params):
    test_output = pd.concat([df_predict_mil[['player', 'team', 'week', 'year']], 
                            pd.Series(best_predictions_mil.mean(axis=1), name='pred_fp_per_game_class')], 
                            axis=1).sort_values(by='pred_fp_per_game_class', ascending=False)
    print(test_output)

    save_val_to_db(model_output_path, best_val_mil, run_params, 'million', table_name='Model_Validations_Million')
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

def save_stack_runs(model_output_path, fname, best_models, scores, stack_val_pred):
    stack_out = {}
    stack_out['best_models'] = best_models
    stack_out['scores'] = scores
    stack_out['stack_val_pred'] = stack_val_pred
    save_pickle(stack_out, model_output_path, fname, protocol=-1)

def load_stack_runs(model_output_path, fname):
    stack_in = load_pickle(model_output_path, fname)
    return stack_in['best_models'], stack_in['scores'], stack_in['stack_val_pred']

def validation_compare_df(model_output_path, best_val, model_obj='reg'):
    
    _, _, _, _, oof_data = mf.load_all_pickles(model_output_path, model_obj)
    
    if model_obj == 'reg': label='reg_adp'
    elif model_obj == 'class': label = 'class_lr_c_80'
    elif model_obj == 'million': label = 'class_lr_c_million'
    oof_data = oof_data[label][['player', 'team', 'year', 'week', 'y_act']].reset_index(drop=True)
    best_val = pd.Series(best_val.mean(axis=1), name='pred_fp_per_game')
    val_compare = pd.concat([oof_data, best_val], axis=1)
    
    return val_compare


def save_val_to_db(model_output_path, best_val, run_params, model_obj, table_name):

    df = validation_compare_df(model_output_path, best_val, model_obj)

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
    
def cleanup_X_y(X, y):
    X_player = X.copy()
    X = X.drop(['player', 'team', 'week', 'year'], axis=1).dropna(axis=0)
    y = y[y.index.isin(X.index)].y_act
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    return X_player, X, y

def add_to_all(df, df_all):
    df_all = pd.concat([df_all, df.assign(set_pos=set_pos)], axis=0).reset_index(drop=True)
    return df_all

def join_stats_mil(X_stack_mil, X_stack_player, X_predict_mil, X_predict_player):
    X_stack_mil = pd.merge(X_stack_mil, X_stack_player, on=['player', 'team', 'week', 'year'], how='left')
    X_predict_mil = pd.merge(X_predict_mil, X_predict_player,  on=['player', 'team', 'week', 'year'], how='left')
    return X_stack_mil, X_predict_mil


def create_final_val_df(X_stack_player, y_stack_all, best_val_reg, best_val_class, best_val_quant):
    df_val_final = pd.concat([X_stack_player[['player', 'team', 'week', 'year']], 
                              pd.Series(best_val_reg.mean(axis=1), name='pred_fp_per_game'),
                              pd.Series(best_val_class.mean(axis=1), name='pred_fp_per_game_class'),
                              pd.Series(best_val_quant.mean(axis=1), name='pred_fp_per_game_quantile')], axis=1)
    df_val_final = pd.merge(df_val_final, y_stack_all, on=['player', 'team', 'week', 'year'])
    return df_val_final


def load_run_models(run_params, final_models, X_stack, y_stack, X_predict, model_obj, alpha=None, grp=None, suffix=''):
    
    model_output_path = run_params['model_output_path']
    ens_vers = run_params['ensemble_vers']
    if alpha is not None: alpha_label = alpha*100
    else: alpha_label = ''

    if os.path.exists(f"{model_output_path}{model_obj}{alpha_label}_{ens_vers}{suffix}.p"):
            best_models, scores, stack_val_pred = load_stack_runs(model_output_path, f"{model_obj}{alpha_label}_{ens_vers}{suffix}")
    
    else:
        stack_val_pred = pd.DataFrame(); scores = []; best_models = []
        for i, fm in enumerate(final_models):
            best_models, scores, stack_val_pred = run_stack_models(fm, i, X_stack, y_stack, best_models, 
                                                                    scores, stack_val_pred, model_obj=model_obj,
                                                                    grp=grp, alpha=alpha, show_plots=show_plot, 
                                                                    num_k_folds=num_k_folds, print_coef=print_coef)

        save_stack_runs(model_output_path, f'{model_obj}{alpha_label}_{ens_vers}{suffix}', best_models, scores, stack_val_pred)
        
    predictions = stack_predictions(X_predict, best_models, final_models, model_obj=model_obj)
    best_val, best_predictions, _ = average_stack_models(scores, final_models, y_stack, stack_val_pred, 
                                                            predictions, model_obj=model_obj, show_plot=show_plot, 
                                                            min_include=min_include)

    return best_val, best_predictions


def create_mil_output(df_predict_mil, best_predictions_mil):
    output_mil = pd.concat([df_predict_mil[['player', 'team', 'week', 'year']], 
                            pd.Series(best_predictions_mil.mean(axis=1), name='pred_mil')], axis=1)
    try:
        actuals = dm.read("SELECT * FROM Top_Players", 'DK_Results')
        mil_results = pd.merge(output_mil, actuals[['player', 'week', 'year', 'y_act']], 
                            on=['player', 'week', 'year'], how='left').fillna(0)
        display(mil_results.sort_values(by='pred_mil', ascending=False))
        show_calibration_curve(mil_results['y_act'], mil_results['pred_mil'], n_bins=8)
    except:
        display(output_mil.sort_values(by='pred_mil', ascending=False))

    return output_mil
            

def display_output(output):
     
    try:  
        output = add_actual(output)
        print(output.loc[:50, ['player', 'week', 'year', 'dk_salary', 'dk_rank', 'pred_fp_per_game', 'pred_fp_per_game_class',
                                'pred_fp_per_game_quantile', 'actual_pts', 'std_dev', 'min_score', 'max_score']])
                        
        if show_plot: 
            mf.show_scatter_plot(output.pred_fp_per_game, output.actual_pts)
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

#---------------
# Settings
#---------------

run_params = {
    
    # set year and week to analyze
    'set_year': 2022,

    # set beginning of validation period
    'val_year_min': 2020,
    'val_week_min': 1,

    # opt params
    'n_iters': 50,
    'n_splits': 5,

    # other parameters
    'use_sample_weight': False,
    'opt_type': 'custom_rand',
    'met': 'y_act',
}

metrics_dict = {
    'pred_spline_class80_q80_matt1_brier1_kfold3':{'pred_fp_per_game': 1, 
                                                   'pred_fp_per_game_class': 1, 
                                                   'pred_fp_per_game_quantile': 1},
   
    'pred_spline_class80_matt1_brier1_kfold3': {'pred_fp_per_game': 1, 
                                                'pred_fp_per_game_class': 1},
    'pred_spline_q80_matt1_brier1_kfold3': {'pred_fp_per_game': 1, 
                                            'pred_fp_per_game_quantile': 1},
    'spline_class80_q80_matt1_brier1_kfold3': {'pred_fp_per_game_class': 1, 
                                               'pred_fp_per_game_quantile': 1},

    'pred_spline_class80_q80_matt0_brier1_kfold3':{'pred_fp_per_game': 1, 
                                                   'pred_fp_per_game_class': 1, 
                                                   'pred_fp_per_game_quantile': 1},
    'pred_spline_class80_matt0_brier1_kfold3': {'pred_fp_per_game': 1, 
                                                'pred_fp_per_game_class': 1},
    'pred_spline_q80_matt0_brier1_kfold3': {'pred_fp_per_game': 1, 
                                            'pred_fp_per_game_quantile': 1},
    'spline_class80_q80_matt0_brier1_kfold3': {'pred_fp_per_game_class': 1, 
                                               'pred_fp_per_game_quantile': 1}
}

full_stack_features = True

min_include = 2
show_plot= True
print_coef = False
num_k_folds = 3

r2_wt = 0
sera_wt = 1
mse_wt = 0
brier_wt = 1
matt_wt = 0

calibrate = False

# set the model version
set_weeks = [12]

pred_versions = len(set_weeks)*['sera1_rsq0_brier1_matt1_lowsample_perc_ffa_fc']

ensemble_versions = len(set_weeks) * ['sera1_rsq0_include2_fullstack_combined']
# ensemble_versions = len(set_weeks) * ['no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3']
# ensemble_versions = len(set_weeks) * ['no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3_fullstack']
# ensemble_versions = len(set_weeks) * ['no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3']

std_dev_types = [
                #  'boot_reg_quant_frac_random_replace_random'
                'pred_spline_class80_q80_matt0_brier1_kfold3',
                # 'pred_spline_class80_matt0_brier1_kfold3',
                # 'pred_spline_q80_matt0_brier1_kfold3',
                # 'spline_class80_q80_matt0_brier1_kfold3'
            ]
std_dev_type = std_dev_types[0]

for w, vers, ensemble_vers in zip(set_weeks, pred_versions, ensemble_versions):

    run_params['set_week'] = w
    run_params['ensemble_vers'] = ensemble_vers
    runs = [
        # ['QB', 'full_model', ''],
        # ['RB', 'full_model', ''],
        # ['WR', 'full_model', ''],
        # ['TE', 'full_model', ''],
        # ['Defense', 'full_model', ''],
        ['QB', 'backfill', ''],
        ['RB', 'backfill', ''],
        ['WR', 'backfill', ''],
        ['TE', 'backfill', '']
    ]

    X_stack_all = pd.DataFrame()
    y_stack_all = pd.DataFrame()
    X_predict_all = pd.DataFrame()
    y_stack_class_all = pd.DataFrame()
    output_start_all = pd.DataFrame()
    df_train_all = pd.DataFrame()

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

        # get the stack cuts
        _, _, best_models_class, _, full_hold_class = mf.load_all_pickles(model_output_path, 'class')

        run_params['cuts'] = sorted(list(set([int(c[-2:]) for c in full_hold_class.keys()])))
        class_cut = run_params['cuts'][-2]

        # get the training data for stacking and prediction data after stacking
        X_stack, y_stack, y_stack_class, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path, class_cut)
        X_predict_player, X_predict = get_stack_predict_data(df_train, df_predict, df, run_params, models_reg, models_class, models_quant)
        
        # add the X and y datasets to the all datasets
        X_stack_all = add_to_all(X_stack, X_stack_all)
        X_predict_all = add_to_all(X_predict, X_predict_all)
        y_stack_all = add_to_all(y_stack, y_stack_all)
        y_stack_class_all = add_to_all(y_stack_class, y_stack_class_all)
        output_start_all = add_to_all(output_start, output_start_all)

#%%
    # cleanup the X and y datasets
    X_stack_player, X_stack, y_stack = cleanup_X_y(X_stack_all, y_stack_all)
    _, _, y_stack_class = cleanup_X_y(X_stack_player, y_stack_class_all)

    X_predict = X_predict_all.copy()
    X_stack = pd.concat([X_stack, pd.get_dummies(X_stack.set_pos)], axis=1).drop('set_pos', axis=1)
    X_predict = pd.concat([X_predict, pd.get_dummies(X_predict.set_pos)], axis=1).drop('set_pos', axis=1)

    # class metrics
    final_models = ['lr_c', 'lgbm_c', 'rf_c', 'gbm_c', 'gbmh_c', 'xgb_c', 'knn_c']
    best_val_class, best_predictions_class = load_run_models(run_params, final_models, X_stack, y_stack_class, X_predict, 'class')
    if show_plot: show_calibration_curve(y_stack_class, best_val_class.mean(axis=1), n_bins=8)
    # save_val_to_db(model_output_path, best_val_class, run_params, 'class', table_name='Model_Validations_Class')

    # quantile regression metrics
    final_models = ['qr_q', 'gbm_q', 'lgbm_q', 'rf_q', 'knn_q']
    best_val_quant, best_predictions_quant = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'quantile', alpha=0.8)

    # create the stacking models
    final_models = ['ridge', 'lasso', 'huber', 'lgbm', 'xgb', 'rf', 'bridge', 'gbm', 'gbmh', 'knn']
    best_val_reg, best_predictions = load_run_models(run_params, final_models, X_stack, y_stack, X_predict, 'reg')
    # save_val_to_db(model_output_path, best_val_reg, run_params, 'reg', table_name='Model_Validations')

    df_val_final = create_final_val_df(X_stack_player, y_stack_all, best_val_reg, best_val_class, best_val_quant)

    # create the output and add standard deviations / max scores
    output = create_output(output_start_all, best_predictions, best_predictions_class, best_predictions_quant)

    # loop through std dev types and display / save output
    for std_dev_type in std_dev_types:            
        metrics = metrics_dict[std_dev_type]
        output = val_std_dev(df_val_final, metrics=metrics, iso_spline='spline', show_plot=show_plot)
        display_output(output)
        # save_output_to_db(output, run_params)

#%%
        #-------------
        # Running the million dataset
        #-------------

        df_predict_mil, X_stack_mil, y_stack_mil, X_predict_mil = X_y_stack_mil(df, run_params)    
        X_stack_mil, X_predict_mil = join_stats_mil(X_stack_mil, X_stack_player, X_predict_mil, X_predict_player)

        X_stack_mil = add_sal_columns(X_stack_mil, df_train)
        X_predict_mil = add_sal_columns(X_predict_mil, df_predict_mil)

        X_predict_mil = X_predict_mil.drop(['player', 'team', 'week', 'year'], axis=1)
        X_stack_mil = X_stack_mil.drop(['player', 'team', 'week', 'year'], axis=1)

        # class metrics
        final_models = ['lr_c', 'lgbm_c', 'rf_c', 'gbm_c', 'gbmh_c', 'xgb_c', 'knn_c']
        best_val_mil, best_predictions_mil = load_run_models(run_params, final_models, X_stack_mil, y_stack_mil, X_predict_mil, 'class', suffix='_mil')
        if show_plot: show_calibration_curve(y_stack_mil, best_val_mil.mean(axis=1), n_bins=8)

        output_mil = create_mil_output(df_predict_mil, best_predictions_mil)
        
        # save_mil_data(df_predict_mil, best_predictions_mil, best_val_mil, run_params)

    # #---------------
    # # Save vegas points and std dev
    # #---------------

    # vp = vegas_points(run_params, metrics={'implied_points_for': 1}, show_plot=show_plot)
    # dm.delete_from_db('Simulation', 'Vegas_Points', f"week={run_params['set_week']} AND year={run_params['set_year']}", create_backup=False)
    # dm.write_to_db(vp, 'Simulation', 'Vegas_Points', 'append')
    # print('All Runs Finished')

#%%

# def trunc_normal(mean_val, sdev, min_sc, max_sc, num_samples=500):

#     import scipy.stats as stats

#     # create truncated distribution
#     lower_bound = (min_sc - mean_val) / sdev, 
#     upper_bound = (max_sc - mean_val) / sdev
#     trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc=mean_val, scale=sdev)
    
#     estimates = trunc_dist.rvs(num_samples)

#     return estimates


# def trunc_normal_dist(df, num_options=500):
#     pred_list = []
#     for mean_val, sdev, min_sc, max_sc in df[['pred_fp_per_game', 'std_dev', 'min_score', 'max_score']].values:
#         pred_list.append(trunc_normal(mean_val, sdev, min_sc, max_sc, num_options))

#     return pd.DataFrame(pred_list)

# def get_predictions(df, num_options):

#     predictions = trunc_normal_dist(df, num_options)
#     labels = df[['player', 'team', 'week','year', 'dk_salary', 'y_act']]
#     predictions = pd.concat([labels, predictions], axis=1)

#     return predictions

# normal_trunc = add_actual(output).rename(columns={'actual_pts': 'y_act'})
# normal_trunc = get_predictions(normal_trunc, 1000)

#%%
from crepes import ConformalRegressor, ConformalPredictiveSystem
from crepes.fillings import sigma_knn, binning
from sklearn.model_selection import train_test_split


X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_stack, y_stack, test_size=0.5)

learner = best_models_reg[0] 
learner.fit(X_prop_train, y_prop_train)
y_hat_test = learner.predict(X_predict)

y_hat_cal = learner.predict(X_cal)
residuals_cal = (y_cal - y_hat_cal).values

sigmas_cal = sigma_knn(X=X_cal, residuals=residuals_cal)
bins_cal, bin_thresholds = binning(values=y_hat_cal, bins=5)
cps_mond_norm = ConformalPredictiveSystem()
cps_mond_norm.fit(residuals=residuals_cal, sigmas=sigmas_cal, bins=bins_cal)

sigmas_test = sigma_knn(X=X_cal, residuals=residuals_cal, X_test=X_predict)
bins_test = binning(values=y_hat_test, bins=bin_thresholds)

intervals = cps_mond_norm.predict(y_hat=y_hat_test,
                                  sigmas=sigmas_test,
                                  bins=bins_test,
                                  lower_percentiles=5,
                                  higher_percentiles=95,
                                  y_min=np.min(y_stack))

# dist = cps_mond_norm.predict(y_hat=y_hat_test,
#                              sigmas=sigmas_test,
#                              bins=bins_test,
#                              return_cpds=True)


output = add_actual(output_start).rename(columns={'actual_pts': 'y_act'})
output = pd.concat([output, pd.Series(sigmas_test, name='std_dev'), pd.DataFrame(intervals)], axis=1)
output = output.sort_values(by=1)
print('Outside:', output[(output.y_act < output[0]) | (output.y_act > output[1])].shape[0] / output.shape[0])
# output[(output.y_act < output[0]) | (output.y_act > output[1])]
output
#%%

cps_mond_norm.evaluate(y_hat=y_hat_test, y=output.y_act.values, sigmas=sigmas_test, bins=bins_test, confidence=0.9, y_min=np.min(y_stack))

#%%
sample = dist[0]
x = np.linspace(np.min(sample), np.max(sample))
dx = x[1]-x[0]
deriv = np.diff(sample)/dx
plt.plot(x, sample[:-1], label="cdf")
plt.plot(x[1:]-dx/2, deriv[:-1], label="derivative")
plt.legend(loc=1)
plt.show()

#%%        
# output = add_actual(output_start).rename(columns={'actual_pts': 'y_act'})

# final_models = [bm for bm in best_models_reg if bm.steps[-1][0] in best_val_reg.columns]
# bs_pred = BootstrapPredictions(final_models, X_stack, y_stack, X_predict, output, random_state=12546)
# df_out = bs_pred.run_bootstrap(N=500, sample_frac=np.random.choice(np.arange(0.5, 1, 0.05)), 
#                                replace=np.random.choice([True, False]), 
#                                label_append=True, n_jobs=-1)

# final_models = [bm for bm in best_models_quant if bm.steps[-1][0] in best_val_quant.columns]
# bs_pred = BootstrapPredictions(final_models, X_stack, y_stack, X_predict, output, random_state=36)
# df_out_quant = bs_pred.run_bootstrap(N=500, sample_frac=np.random.choice(np.arange(0.5, 1, 0.05)), 
#                                      replace=np.random.choice([True, False]), 
#                                      label_append=False, n_jobs=-1)

# df_out_quant.columns = [int(c)+500 for c in df_out_quant.columns]
# df_out = pd.concat([df_out, df_out_quant], axis=1)

# #%%

# def measure_dist_accuracy(df, N):
#     cols = ['perc1', 'perc5', 'perc25', 'perc75', 'perc95', 'perc99']
#     percs = [1, 5, 25, 75, 95, 99]
    
#     for c,p in zip(cols, percs):
#         df[c] = np.percentile(df[range(N)], p, axis=1)
    
#     perc1 = df[df.y_act < df['perc1']].shape[0] / df.shape[0]
#     perc1 -= 0.01
    
#     perc99 = df[df.y_act > df['perc99']].shape[0] / df.shape[0]
#     perc99 -= 0.01

#     perc95 = df[df.y_act > df['perc95']].shape[0] / df.shape[0]
#     perc95 -= 0.05

#     perc90 = df_out[(df.y_act < df['perc95']) & (df.y_act > df['perc5'])].shape[0] / df.shape[0]
#     perc90 -= .90

#     perc50 = df_out[(df.y_act > df['perc25']) & (df.y_act <  df['perc75'])].shape[0] / df.shape[0]
#     perc50 -= .50

#     pct1 = np.round(perc1*100, 1)
#     pct99 = np.round(perc99*100, 1)
#     pct95 = np.round(perc95*100, 1)
#     pct90 = np.round(perc90*100, 1)
#     pct50 = np.round(perc50*100, 1)

#     print(f'Accuracy = Lower 1%: {pct1}%, Middle 50%: {pct50}%, Middle 90%: {pct90}%, Upper 5%: {pct95}, Upper 1%: {pct99}%,')


# measure_dist_accuracy(normal_trunc, 1000)
# measure_dist_accuracy(df_out, 1000)

# #%%

# df_out['mean_pred'] = df_out[range(1000)].mean(axis=1)
# df_out.drop(range(1000), axis=1).sort_values(by='mean_pred', ascending=False).iloc[:50]

# #%%

# df_out[(df_out.player=='Josh Jacobs') & (df_out.week==12)].iloc[0,6:].plot.hist()
# normal_trunc[(normal_trunc.player=='Patrick Mahomes') & (normal_trunc.week==17)].iloc[0,6:].plot.hist()

#%%

# results = []
# for reg_wt in [0,0.5,1,1.5]:
#     for class_wt in [0,0.5, 1, 1.5]:
#         for quant_wt in [0,0.5,1,1.5]: 
#             if reg_wt + class_wt + quant_wt > 0:
#                 # create the output and add standard devations / max scores
#                 output = create_output(output_start, best_predictions, best_predictions_class, best_predictions_quant)

#                 metrics = {'pred_fp_per_game': reg_wt, 'pred_fp_per_game_class': class_wt, 'pred_fp_per_game_quantile': quant_wt}
#                 output = val_std_dev(model_output_path, output, best_val, best_val_class, best_val_quant, metrics=metrics, 
#                                         iso_spline='spline', show_plot=False)

#                 output = add_actual(output)
#                 output = output.rename(columns={'actual_pts': 'y_act'})
#                 output, _ = create_game_date(output, run_params)

#                 # set up the target variable to be categorical based on Xth percentile
#                 cut_perc = output.groupby('game_date')['y_act'].apply(lambda x: np.percentile(x, 80))
#                 output = pd.merge(output, cut_perc.reset_index().rename(columns={'y_act': 'cut_perc'}), on=['game_date'])
#                 output['y_act_class'] = np.where(output.y_act >= output.cut_perc, 1, 0)

#                 # output = output.sort_values(by='pred_fp_per_game_class', ascending=False).reset_index(drop=True)
#                 # show_calibration_curve(output.y_act_class, output.pred_fp_per_game_class, n_bins=8)

#                 # display(output[['player', 'week', 'year', 'y_act', 'y_act_class', 'cut_perc', 
#                 #         'pred_fp_per_game', 'pred_fp_per_game_class', 'pred_fp_per_game_quantile',
#                 #          'std_dev', 'min_score', 'max_score']].iloc[:50])

#                 # display(output.loc[output.week==8, ['player', 'week', 'year', 'y_act', 'y_act_class', 'cut_perc', 
#                 #         'pred_fp_per_game', 'pred_fp_per_game_class', 'pred_fp_per_game_quantile',
#                 #         'std_dev', 'min_score', 'max_score']])

#                 pct_min = output[output.y_act < output.min_score].shape[0] / output.shape[0]
#                 pct_max = output[output.y_act > output.max_score].shape[0] / output.shape[0]

#                 one_std = output[(output.y_act < (output.pred_fp_per_game + 1*output.std_dev)) & \
#                     (output.y_act > (output.pred_fp_per_game - 1*output.std_dev))].shape[0] / output.shape[0]
#                 one_std -= .68
                
#                 two_std = output[(output.y_act < (output.pred_fp_per_game + 2*output.std_dev)) & \
#                     (output.y_act > (output.pred_fp_per_game - 2*output.std_dev))].shape[0] / output.shape[0]
#                 two_std -= .95

#                 three_std = output[(output.y_act < (output.pred_fp_per_game + 3*output.std_dev)) & \
#                     (output.y_act > (output.pred_fp_per_game - 3*output.std_dev))].shape[0] / output.shape[0]
#                 three_std -= .997

#                 results.append([reg_wt, class_wt, quant_wt, pct_min, pct_max, one_std, two_std, three_std])
#                 results_df = pd.DataFrame(results, columns=['reg_wt', 'class_wt', 'quant_wt', 'pct_min', 'pct_max', 'one_std', 'two_std', 'three_std'])
#                 results_df['total_error'] = results_df[['pct_min', 'pct_max', 'one_std', 'two_std', 'three_std']].abs().sum(axis=1)
# results_df.sort_values(by='total_error')
