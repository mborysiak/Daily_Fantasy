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
from sklearn.metrics import r2_score, brier_score_loss
import zModel_Functions as mf
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.calibration import CalibratedClassifierCV

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

import pandas_bokeh
pandas_bokeh.output_notebook()

import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
# warnings.filterwarnings("ignore", category=UserWarning) 

from sklearn import set_config
set_config(display='text')

#====================
# Data Loading Functions
#====================

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


def get_full_pipe(skm, m, alpha=None, stack_model=False, std_model=False, min_samples=10):

    if m == 'adp':
        
        # set up the ADP model pipe
        pipe = skm.model_pipe([skm.piece('feature_select'), 
                               skm.piece('std_scale'), 
                               skm.piece('k_best'),
                               skm.piece('lr')])

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
    
    if stack_model: 
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
    X_stack = pd.concat([df_labels[['player', 'week', 'year']], X_stack, X_stack_quant, X_stack_class], axis=1)
    
    if prefix=='':
        y_stack = pd.concat([df_labels[['player', 'week', 'year']], y_stack], axis=1)
        y_stack_class = pd.concat([df_labels[['player', 'week', 'year']], y_stack_class], axis=1)

    return X_stack, y_stack, y_stack_class, models_reg, models_class, models_quant


def add_calibrator_to_pipe(pipe, params):

    est_name = pipe.steps[-1][0]
    est = CalibratedClassifierCV(pipe.steps[-1][1], cv=3)
    pipe.steps[-1] = (est_name, est)

    new_params = {}
    for p, v in params.items():
        if est_name in p:
            p = p.split('__')[0] + '__base_estimator__' + p.split('__')[1]
        
        new_params[p] = v
    
    return pipe, new_params


def run_stack_models(final_m, i, X_stack, y_stack, best_models, scores, 
                     stack_val_pred, model_obj='reg', run_adp=True, show_plots=True,
                     calibrate=False, num_k_folds=3, print_coef=True):

    print(f'\n{final_m}')

    if model_obj == 'class': proba = True
    else: proba = False

    if model_obj == 'quantile': alpha = 0.8
    else: alpha = None

    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, to_drop=[])
    pipe, params = get_full_pipe(skm, final_m, stack_model=True, alpha=alpha)
    
    if calibrate: pipe, params = add_calibrator_to_pipe(pipe, params)

    if 'gbmh' in final_m or (calibrate and model_obj=='class') or 'knn' in final_m: print_coef = False
    else: print_coef = print_coef

    best_model, stack_scores, stack_pred = skm.best_stack(pipe, params,
                                                          X_stack, y_stack, n_iter=run_params['n_iters'], 
                                                          run_adp=run_adp, print_coef=print_coef,
                                                          sample_weight=False, proba=proba,
                                                          num_k_folds=num_k_folds, alpha=alpha,
                                                          random_state=(i*12)+(i*17))

    best_models.append(best_model)
    scores.append(stack_scores['stack_score'])
    stack_val_pred = pd.concat([stack_val_pred, pd.Series(stack_pred['stack_pred'], name=final_m)], axis=1)

    if show_plots:
        mf.show_scatter_plot(stack_pred['stack_pred'], stack_pred['y'], r2=True)


    return best_models, scores, stack_val_pred


def fit_and_predict(m, df_predict, X, y, proba):
    import sklearn

    # try: 
    #     check_is_fitted(m)
    # except: 
    #     m.fit(X, y)
    m.fit(X,y)

    if proba and 'calibrate' in vers and type(m.steps[-1][1])!=sklearn.calibration.CalibratedClassifierCV:
        m = CalibratedClassifierCV(m, cv='prefit')
        m.fit(X, y)

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

    return X_predict


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


def show_calibration_curve(y_true, y_pred, n_bins=10, strategy='quantile'):

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

def val_std_dev(model_output_path, output, best_val, best_val_class=None, best_val_quant=None,
                metrics={'pred_fp_per_game': 1}, iso_spline='iso', show_plot=True):

    _, _, _, _, oof_data = mf.load_all_pickles(model_output_path, 'reg')
    val_data = oof_data['reg_adp'][['player', 'team', 'year', 'week', 'y_act']].reset_index(drop=True)
    
    val_data['pred_fp_per_game'] = best_val.mean(axis=1)

    if 'pred_fp_per_game_class' in metrics.keys() and best_val_class is not None:
        val_data['pred_fp_per_game_class'] = best_val_class.mean(axis=1)
    
    if 'pred_fp_per_game_quantile' in metrics.keys() and best_val_quant is not None:
        val_data['pred_fp_per_game_quantile'] = best_val_quant.mean(axis=1)
        
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

def vegas_points(run_params, metrics={'implied_points_for': 1}):

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

    sd_m, max_m, min_m = get_std_splines(scores, metrics, show_plot=True, k=2, 
                                        min_grps_den=int(scores.shape[0]*0.08), 
                                        max_grps_den=int(scores.shape[0]*0.04),
                                        iso_spline='spline')

    output['std_dev'] = sd_m(sd_max_met)
    output['max_score'] = max_m(sd_max_met)
    output['min_score'] = min_m(sd_max_met)
    output = output.rename(columns={'player': 'team'})
    return output


def add_actual(df):
    if set_pos=='Defense': pl = 'defTeam'
    else: pl = 'player'

    if run_params['rush_pass'] != '': rush_pass = f"_{run_params['rush_pass']}"
    else: rush_pass = ''
    actual_pts = dm.read(f'''SELECT {pl} player, week, season year, fantasy_pts{rush_pass} actual_pts
                            FROM {set_pos}_Stats 
                            WHERE week>={run_params['set_week']} 
                                and season={run_params['set_year']}''', 'FastR')
    
    if len(actual_pts) > 0:
        df = pd.merge(df, actual_pts, on=['player', 'week', 'year'])
    return df



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
    
 
    dm.delete_from_db('Simulation', table_name, del_str, create_backup=False)
    dm.write_to_db(df, 'Simulation', table_name, 'append')

  

def save_test_to_db(df, run_params):

    df['pos'] = set_pos
    df['pred_version'] = vers
    df['ensemble_vers'] = ensemble_vers
    df['model_type'] = model_type
    df['set_week'] = run_params['set_week']
    df['set_year'] = run_params['set_year']

    df = df[['player', 'set_week', 'set_year', 'pos', 'pred_fp_per_game', 'actual_pts',
             'pred_version', 'ensemble_vers', 'model_type']]

    del_str = f'''pos='{set_pos}' 
                  AND pred_version='{vers}'
                  AND ensemble_vers='{ensemble_vers}' 
                  AND set_week={run_params['set_week']} 
                  AND set_year={run_params['set_year']}
                  AND model_type='{model_type}'
                '''
    dm.delete_from_db('Simulation', f'Model_Test_Validations', del_str, create_backup=False)
    dm.write_to_db(df, 'Simulation', 'Model_Test_Validations', 'append')


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

min_include = 2
show_plot= True
print_coef = True
num_k_folds = 3

r2_wt = 0
sera_wt = 1
mse_wt = 0
brier_wt = 1
matt_wt = 1

calibrate = False

# set the model version
set_weeks = [
         # 1, 2, 3, 4, 5, 
         # 6, 7, 8, 9, 10,
        #  13, 14, 15, 16
        # 12, 13
        16
                 ]

pred_versions = [
                'sera1_rsq0_brier1_matt1_lowsample_perc_ffa_fc',
                #  'sera1_rsq0_brier1_matt1_lowsample_perc',
                #   'sera1_rsq0_brier1_matt1_lowsample_perc',
                #    'sera1_rsq0_brier1_matt1_lowsample_perc',
                #     'sera1_rsq0_brier1_matt1_lowsample_perc',
                #      'sera1_rsq0_brier1_matt1_lowsample_perc',
                #       'sera1_rsq0_brier1_matt1_lowsample_perc',
                #        'sera1_rsq0_brier1_matt1_lowsample_perc',
                #        'sera1_rsq0_brier1_matt1_lowsample_perc',
                #        'sera1_rsq0_brier1_matt1_lowsample_perc'
                ]

ensemble_versions = [
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    # 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    # 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    # 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    # 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    # 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    # 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    # 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    # 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    # 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3'
                     ]

std_dev_type = 'pred_spline_class80_q80_matt1_brier1_kfold3'

for w, vers, ensemble_vers in zip(set_weeks, pred_versions, ensemble_versions):

    run_params['set_week'] = w
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
    for set_pos, model_type, rush_pass in runs:

        run_params['rush_pass'] = rush_pass

        # load data and filter down
        pkey, db_output, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
        df, run_params = load_data(model_type, set_pos, run_params)
        df, run_params = create_game_date(df, run_params)
        df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

        #------------
        # Run the Stacking Models and Generate Output
        #------------

        # get the stack cuts
        _, _, best_models_class, _, full_hold_class = mf.load_all_pickles(model_output_path, 'class')

        run_params['cuts'] = sorted(list(set([int(c[-2:]) for c in full_hold_class.keys()])))
        class_cut = run_params['cuts'][-2]

        # get the training data for stacking and prediction data after stacking
        X_stack, y_stack, y_stack_class, models_reg, models_class, models_quant = load_all_stack_pred(model_output_path, class_cut)
        X_predict = get_stack_predict_data(df_train, df_predict, df, run_params, 
                                           models_reg, models_class, models_quant)

        # if set_pos=='QB':
        #     try: X_stack, X_predict, y_stack, y_stack_class, run_params = \
        #             add_rush_pass(X_stack, X_predict, y_stack, y_stack_class, run_params)
        #     except: print('No rush / pass data available')

        X_stack = X_stack.drop(['player', 'week', 'year'], axis=1).dropna(axis=0)
        y_stack = y_stack[y_stack.index.isin(X_stack.index)].y_act
        y_stack_class = y_stack_class[y_stack_class.index.isin(X_stack.index)].y_act
        X_stack, y_stack, y_stack_class = X_stack.reset_index(drop=True), y_stack.reset_index(drop=True), y_stack_class.reset_index(drop=True)

        # class metrics
        final_models = ['lr_c', 'lgbm_c', 'rf_c', 'gbm_c', 'gbmh_c', 'xgb_c', 'knn_c']
        stack_val_pred = pd.DataFrame(); scores = []; best_models = []
        for i, fm in enumerate(final_models):
            best_models, scores, stack_val_pred = run_stack_models(fm, i, X_stack, y_stack_class, best_models, 
                                                                    scores, stack_val_pred, model_obj='class',
                                                                    run_adp=False, show_plots=False, 
                                                                    calibrate=calibrate, num_k_folds=num_k_folds,
                                                                    print_coef=print_coef)

            if show_plot: show_calibration_curve(y_stack_class, stack_val_pred[fm], n_bins=8)

        predictions = stack_predictions(X_predict, best_models, final_models, model_obj='class')
        best_val_class, best_predictions_class, _ = average_stack_models(scores, final_models, y_stack_class, stack_val_pred, 
                                                                            predictions, model_obj='class', show_plot=show_plot, 
                                                                            min_include=min_include)

        if show_plot: show_calibration_curve(y_stack_class, best_val_class.mean(axis=1), n_bins=8)
        save_val_to_db(model_output_path, best_val_class, run_params, 'class', table_name='Model_Validations_Class')

        # quantile regression metrics
        final_models = ['qr_q', 'gbm_q', 'lgbm_q', 'rf_q', 'knn_q']
        stack_val_pred = pd.DataFrame(); scores = []; best_models = []
        for i, fm in enumerate(final_models):
            best_models, scores, stack_val_pred = run_stack_models(fm, i, X_stack, y_stack, best_models, 
                                                                    scores, stack_val_pred, model_obj='quantile',
                                                                    run_adp=False, show_plots=show_plot, 
                                                                    calibrate=calibrate, num_k_folds=num_k_folds,
                                                                    print_coef=print_coef)

        predictions = stack_predictions(X_predict, best_models, final_models, model_obj='quantile')
        best_val_quant, best_predictions_quant, _ = average_stack_models(scores, final_models, y_stack, stack_val_pred, 
                                                                        predictions, model_obj='quantile', show_plot=show_plot, 
                                                                        min_include=min_include)


        # create the stacking models
        final_models = ['ridge', 'lasso', 'huber', 'lgbm', 'xgb', 'rf', 'bridge', 'gbm', 'gbmh', 'knn']
        stack_val_pred = pd.DataFrame(); scores = []; best_models = []
        for i, fm in enumerate(final_models):
            best_models, scores, stack_val_pred = run_stack_models(fm, i, X_stack, y_stack, best_models, 
                                                                   scores, stack_val_pred, show_plots=show_plot,
                                                                   num_k_folds=num_k_folds, print_coef=print_coef)

        predictions = stack_predictions(X_predict, best_models, final_models)
        best_val, best_predictions, best_score = average_stack_models(scores, final_models, y_stack, stack_val_pred, 
                                                                      predictions, model_obj='reg',
                                                                      show_plot=show_plot, min_include=min_include)
        save_val_to_db(model_output_path, best_val, run_params, 'reg', table_name='Model_Validations')
        
        # create the output and add standard devations / max scores
        output = create_output(output_start, best_predictions, best_predictions_class, best_predictions_quant)
        save_prob_to_db(output, run_params, 'Predicted_Probability')

        metrics = {'pred_fp_per_game': 1, 'pred_fp_per_game_class': 1, 'pred_fp_per_game_quantile': 1}
        output = val_std_dev(model_output_path, output, best_val, best_val_class, best_val_quant, metrics=metrics, 
                             iso_spline='spline', show_plot=show_plot)
        
        try:  
            output = add_actual(output)
            print(output.loc[:50, ['player', 'week', 'year', 'dk_salary', 'dk_rank', 'pred_fp_per_game', 'pred_fp_per_game_class',
                                   'pred_fp_per_game_quantile', 'actual_pts', 'std_dev', 'min_score', 'max_score']])
            
            save_test_to_db(output, run_params)
            
            if show_plot: 
                mf.show_scatter_plot(output.pred_fp_per_game, output.actual_pts)
                skm_score, _, _ = get_skm(df_train, model_obj='reg', to_drop=[])
                print('Score:', np.round(skm_score.custom_score(output.pred_fp_per_game, output.actual_pts),2))
            
            output = output.drop('actual_pts', axis=1)
        except:
            print(output.loc[:50, ['player', 'dk_salary','dk_rank', 'pred_fp_per_game', 'pred_fp_per_game_class', 
                                   'pred_fp_per_game_quantile', 'std_dev', 'min_score', 'max_score']])

        save_output_to_db(output, run_params)

vp = vegas_points(run_params, metrics={'implied_points_for': 1})
dm.delete_from_db('Simulation', 'Vegas_Points', f"week={run_params['set_week']} AND year={run_params['set_year']}", create_backup=False)
dm.write_to_db(vp, 'Simulation', 'Vegas_Points', 'append')
print('All Runs Finished')

#%%

results = []
for reg_wt in [0,0.5,1,1.5]:
    for class_wt in [0,0.5, 1, 1.5]:
        for quant_wt in [0,0.5,1,1.5]: 
            if reg_wt + class_wt + quant_wt > 0:
                # create the output and add standard devations / max scores
                output = create_output(output_start, best_predictions, best_predictions_class, best_predictions_quant)

                metrics = {'pred_fp_per_game': reg_wt, 'pred_fp_per_game_class': class_wt, 'pred_fp_per_game_quantile': quant_wt}
                output = val_std_dev(model_output_path, output, best_val, best_val_class, best_val_quant, metrics=metrics, 
                                        iso_spline='spline', show_plot=False)

                output = add_actual(output)
                output = output.rename(columns={'actual_pts': 'y_act'})
                output, _ = create_game_date(output, run_params)

                # set up the target variable to be categorical based on Xth percentile
                cut_perc = output.groupby('game_date')['y_act'].apply(lambda x: np.percentile(x, 80))
                output = pd.merge(output, cut_perc.reset_index().rename(columns={'y_act': 'cut_perc'}), on=['game_date'])
                output['y_act_class'] = np.where(output.y_act >= output.cut_perc, 1, 0)

                # output = output.sort_values(by='pred_fp_per_game_class', ascending=False).reset_index(drop=True)
                # show_calibration_curve(output.y_act_class, output.pred_fp_per_game_class, n_bins=8)

                # display(output[['player', 'week', 'year', 'y_act', 'y_act_class', 'cut_perc', 
                #         'pred_fp_per_game', 'pred_fp_per_game_class', 'pred_fp_per_game_quantile',
                #          'std_dev', 'min_score', 'max_score']].iloc[:50])

                # display(output.loc[output.week==8, ['player', 'week', 'year', 'y_act', 'y_act_class', 'cut_perc', 
                #         'pred_fp_per_game', 'pred_fp_per_game_class', 'pred_fp_per_game_quantile',
                #         'std_dev', 'min_score', 'max_score']])

                pct_min = output[output.y_act < output.min_score].shape[0] / output.shape[0]
                pct_max = output[output.y_act > output.max_score].shape[0] / output.shape[0]

                one_std = output[(output.y_act < (output.pred_fp_per_game + 1*output.std_dev)) & \
                    (output.y_act > (output.pred_fp_per_game - 1*output.std_dev))].shape[0] / output.shape[0]
                one_std -= .68
                
                two_std = output[(output.y_act < (output.pred_fp_per_game + 2*output.std_dev)) & \
                    (output.y_act > (output.pred_fp_per_game - 2*output.std_dev))].shape[0] / output.shape[0]
                two_std -= .95

                three_std = output[(output.y_act < (output.pred_fp_per_game + 3*output.std_dev)) & \
                    (output.y_act > (output.pred_fp_per_game - 3*output.std_dev))].shape[0] / output.shape[0]
                three_std -= .997

                results.append([reg_wt, class_wt, quant_wt, pct_min, pct_max, one_std, two_std, three_std])
                results_df = pd.DataFrame(results, columns=['reg_wt', 'class_wt', 'quant_wt', 'pct_min', 'pct_max', 'one_std', 'two_std', 'three_std'])
                results_df['total_error'] = results_df[['pct_min', 'pct_max', 'one_std', 'two_std', 'three_std']].abs().sum(axis=1)
results_df.sort_values(by='total_error')

# %%

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
    run_params['n_splits'] = 4

    return df_train_mil, df_predict_mil, min_samples_mil, run_params


for w, vers, ensemble_vers in zip(set_weeks, pred_versions, ensemble_versions):

    run_params['set_week'] = w
    runs = [
        ['QB', 'full_model', ''],
        ['RB', 'full_model', ''],
        ['WR', 'full_model', ''],
        ['TE', 'full_model', ''],
        ['Defense', 'full_model', ''],
        ['QB', 'backfill', ''],
        ['RB', 'backfill', ''],
        ['WR', 'backfill', ''],
        ['TE', 'backfill', '']
    ]
    for set_pos, model_type, rush_pass in runs:

        print(f'\n----------\n{set_pos} week {w} {model_type}\n----------\n')
        run_params['rush_pass'] = rush_pass

        # load data and filter down
        pkey, db_output, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
        df, run_params = load_data(model_type, set_pos, run_params)
        df, run_params = create_game_date(df, run_params)
        df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)
            
        df_train_mil, df_predict_mil, min_samples_mil, run_params = predict_million_df(df, run_params)

        # load the regregression predictions
        pred, actual, models_mil, scores, full_hold_mil = mf.load_all_pickles(model_output_path, 'million')
        X_stack_mil, y_stack_mil, _ = mf.X_y_stack('mil', full_hold_mil, pred, actual)


        _, X, y = get_skm(df_train_mil, 'class', to_drop=run_params['drop_cols'])
        X_predict_mil = create_stack_predict(df_predict_mil, models_mil, X, y, proba=True)


        show_plot=True

        # class metrics
        final_models = ['lr_c', 'lgbm_c', 'rf_c', 'gbm_c', 'gbmh_c', 'xgb_c', 'knn_c']
        stack_val_pred = pd.DataFrame(); scores = []; best_models = []
        for i, fm in enumerate(final_models):
            best_models, scores, stack_val_pred = run_stack_models(fm, i, X_stack_mil, y_stack_mil, best_models, 
                                                                    scores, stack_val_pred, model_obj='class',
                                                                    run_adp=False, show_plots=False, 
                                                                    calibrate=calibrate, num_k_folds=num_k_folds,
                                                                    print_coef=print_coef)

            if show_plot: show_calibration_curve(y_stack_mil, stack_val_pred[fm], n_bins=8)

        predictions = stack_predictions(X_predict_mil, best_models, final_models, model_obj='class')
        best_val_mil, best_predictions_mil, _ = average_stack_models(scores, final_models, y_stack_mil, stack_val_pred, 
                                                                            predictions, model_obj='class', show_plot=show_plot, 
                                                                            min_include=min_include)

        if show_plot: show_calibration_curve(y_stack_mil, best_val_mil.mean(axis=1), n_bins=8)

        test_output = pd.concat([df_predict_mil[['player', 'team', 'week', 'year']], 
                                 pd.Series(best_predictions_mil.mean(axis=1), name='pred_fp_per_game_class')], 
                                 axis=1).sort_values(by='pred_fp_per_game_class', ascending=False)
        display(test_output)

        save_val_to_db(model_output_path, best_val_mil, run_params, 'million', table_name='Model_Validations_Million')
        save_prob_to_db(test_output, run_params, 'Predicted_Million')

# %%

for w, vers, ensemble_vers in zip(set_weeks, pred_versions, ensemble_versions):

    run_params['set_week'] = w
    runs = [
        # ['QB', 'full_model', ''],
        # ['RB', 'full_model', ''],
        # ['WR', 'full_model', ''],
        # ['TE', 'full_model', ''],
        # ['Defense', 'full_model', ''],
        # ['QB', 'backfill', ''],
        # ['RB', 'backfill', ''],
        # ['WR', 'backfill', ''],
        # ['TE', 'backfill', '']
    ]
    for set_pos, model_type, rush_pass in runs:

        run_params['rush_pass'] = rush_pass

        # load data and filter down
        pkey, db_output, model_output_path = create_pkey_output_path(set_pos, run_params, model_type, vers)
        df, run_params = load_data(model_type, set_pos, run_params)
        df, run_params = create_game_date(df, run_params)
        df_train, df_predict, output_start, min_samples = train_predict_split(df, run_params)

        df_train_mil, df_predict_mil, min_samples_mil, run_params = predict_million_df(df, run_params)
                
        pred, actual, models_mil, scores, full_hold_mil = mf.load_all_pickles(model_output_path, 'million')
        X_stack_mil, y_stack_mil, _ = mf.X_y_stack('mil', full_hold_mil, pred, actual)
        _, X, y = get_skm(df_train_mil, 'class', to_drop=run_params['drop_cols'])
        X_predict_mil = create_stack_predict(df_predict_mil, models_mil, X, y, proba=True)
        scores = [s[1] for s in scores]

        final_models = ['lr_c', 'lgbm_c', 'rf_c', 'gbm_c', 'gbmh_c', 'xgb_c', 'knn_c']
        final_models = ['class_'+m+'_million' for m in final_models]
        best_val_mil, best_predictions_mil, _ = average_stack_models(scores, final_models, y_stack_mil, X_stack_mil, 
                                                                            X_predict_mil, model_obj='class', show_plot=True, 
                                                                            min_include=min_include)

        test_output = pd.concat([df_predict_mil[['player', 'team', 'week', 'year']], 
                                 pd.Series(best_predictions_mil.mean(axis=1), name='pred_fp_per_game_class')], 
                                 axis=1).sort_values(by='pred_fp_per_game_class', ascending=False)
        display(test_output)
        save_val_to_db(model_output_path, best_val_mil, run_params, 'million', table_name='Model_Validations_Million')
        save_prob_to_db(test_output, run_params, 'Predicted_Million')

# %%


pred, actual, models_reg, scores, full_hold_reg = mf.load_all_pickles(model_output_path, 'reg')
scores = [s[1] for s in scores.values()]

final_models = [c for c in X_stack if 'reg' in c]
best_val_class, best_predictions_reg, _ = average_stack_models(scores, final_models, y_stack, X_stack, 
                                                                    X_predict, model_obj='reg', show_plot=show_plot, 
                                                                    min_include=min_include)

output_no_stack = pd.concat([df_predict[['player']], pd.Series(best_predictions_reg.mean(axis=1), name='reg_pred')], axis=1).sort_values(by='reg_pred', ascending=False)
output_no_stack = output_no_stack.assign(week=w, year=run_params['set_year'])
output_no_stack = add_actual(output_no_stack)
display(output_no_stack)
if show_plot: mf.show_scatter_plot(output_no_stack.reg_pred, output_no_stack.actual_pts)


#%%

df = dm.read('''SELECT * FROM Model_Predictions 
                 WHERE version= 'sera1_rsq0_brier1_matt1_lowsample_perc'
                       AND ensemble_vers='no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3'
                       AND std_dev_type='pred_spline_class80_q80_matt1_brier1_kfold3'
                       AND week=11
                       AND year =2022
                       
''', 'Simulation')

df['ensemble_vers'] = 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3'
# df['version'] = 'sera1_rsq0_brier1_matt1_lowsample_perc'

del_str = f'''week=11
              AND year=2022
              AND version='sera1_rsq0_brier1_matt1_lowsample_perc'
              AND ensemble_vers='no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3'
                '''
dm.delete_from_db('Simulation', 'Model_Predictions', del_str, create_backup=False)
dm.write_to_db(df, 'Simulation', 'Model_Predictions', 'append')


# %%
week=12
year=2022

df = dm.read(f'''SELECT player_id, AVG(pred_fp_per_game) projection, pred_ownership ownership
                FROM Model_Predictions
                JOIN (
                      SELECT player, player_id
                      FROM Player_Ids
                      WHERE league={week}
                            AND year={year}
                     ) USING (player)
                JOIN (
                      SELECT player, pred_ownership
                      FROM Predicted_Ownership
                      WHERE week={week}
                            AND year={year}
                    ) USING (player)
                WHERE week={week}
                      AND year={year}
                GROUP BY player
             ''', 'Simulation')

df['ownership'] = np.exp(df.ownership) * 100
df.to_csv('c:/Users/mborysia/Downloads/sim_saber.csv', index=False)


# %%
