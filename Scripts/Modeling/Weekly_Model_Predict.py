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
import yaml
from wakepy import keep
from sklearn.pipeline import Pipeline
import optuna

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

def create_pkey_output_path(set_pos, run_params, model_type):


    pkey = f"{set_pos}_year{run_params['set_year']}_week{run_params['set_week']}_{model_type}{run_params['pred_vers']}"
    model_output_path = f"{root_path}/Model_Outputs/{run_params['set_year']}/{pkey}/"
    if not os.path.exists(model_output_path): os.makedirs(model_output_path)
    run_params['model_output_path'] = model_output_path
    
    return pkey, run_params, model_output_path

def load_data(model_type, set_pos, run_params):

    # load data and filter down
    if model_type=='full_model': df = dm.read(f'''SELECT * 
                                                  FROM {set_pos}_Data{run_params['rush_pass']}_Week{run_params['set_week']}
                                                ''', f"Model_Features_{run_params['set_year']}")
    elif model_type=='backfill': df = dm.read(f'''SELECT * 
                                                  FROM Backfill_{set_pos}_Week{run_params['set_week']}
                                                  WHERE pos='{set_pos}' 
                                                  ''', 
                                                  f"Model_Features_{run_params['set_year']}")

    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * 
                          FROM {set_pos}_Data{run_params['rush_pass']}_Week{run_params['set_week']}_v2
                       ''', f"Model_Features_{run_params['set_year']}")
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

    df_predict = df[(df.game_date == run_params['train_time_split']) & (df.week<17)].reset_index(drop=True)
    output_start = df_predict[['player', 'team', 'week', 'year', 'dk_salary']].copy().drop_duplicates()

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.game_date < run_params['cv_time_input']].shape[0])  
    print('Shape of Train Set', df_train.shape)

    return df_train, df_predict, output_start, min_samples


def get_class_data(df, cut, run_params):

    # set up the training and prediction datasets for the classification 
    df_train_class = df[df.game_date < run_params['train_time_split']].reset_index(drop=True)
    df_predict_class = df[(df.game_date == run_params['train_time_split']) & (df.week<17)].reset_index(drop=True)

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
        'reg': SciKitModel(skm_df, model_obj='reg', r2_wt=r2_wt, sera_wt=sera_wt, mse_wt=mse_wt, logloss_wt=log_wt, hp_algo=run_params['hp_algo']),
        'class': SciKitModel(skm_df, model_obj='class', brier_wt=brier_wt, matt_wt=matt_wt, hp_algo=run_params['hp_algo']),
        'quantile': SciKitModel(skm_df, model_obj='quantile', hp_algo=run_params['hp_algo'])
    }
    
    skm = skm_options[model_obj]
    X, y = skm.Xy_split(y_metric='y_act', to_drop=to_drop)

    return skm, X, y


def get_full_pipe(skm, m, alpha=None, stack_model=False, min_samples=10, bayes_rand='rand'):

    if skm.model_obj=='class': 
        kb = 'k_best_c'
        sp = 'select_perc_c'
    else: 
        kb = 'k_best'
        sp = 'select_perc'

    if 'team_stats' in stack_model: stack_model = stack_model.replace('_team_stats', '')

    stack_models = {

        'full_stack': skm.model_pipe([
                                      skm.piece('std_scale'),
                                      skm.piece(sp), 
                                      skm.feature_union([
                                                    skm.piece('agglomeration'), 
                                                    skm.piece(f'{kb}'),
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
                                                    skm.piece(f'{kb}_fu'),
                                                    skm.piece('pca')
                                                    ]),
                                      skm.piece(kb),
                                      skm.piece(m)
                                      ]),

        'random_full_stack_newp': skm.model_pipe([
                                      skm.piece('random_sample'),
                                      skm.piece('std_scale'), 
                                      skm.feature_union([
                                                    skm.piece('agglomeration'), 
                                                    skm.piece(f'{kb}_fu'),
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
                                        ]),

        'corr_collinear': skm.model_pipe([
                                 skm.piece('corr_collinear'),
                                #  skm.piece('std_scale'),
                                 skm.piece(m)
                                 ]),

    }

    pipe = stack_models[stack_model]
    params = skm.default_params(pipe, bayes_rand=bayes_rand, min_samples=min_samples)
    
    if skm.model_obj == 'quantile':
        if m in ('qr_q', 'gbmh_q'): pipe.set_params(**{f'{m}__quantile': alpha})
        elif m in ('rf_q', 'knn_q'): pipe.set_params(**{f'{m}__q': alpha})
        elif m == 'cb_q': pipe.set_params(**{f'{m}__loss_function': f'Quantile:alpha={alpha}'})
        else: pipe.set_params(**{f'{m}__alpha': alpha})

    if stack_model=='random_full_stack' and run_params['opt_type']=='bayes': 
        params['random_sample__frac'] = hp.uniform('random_sample__frac', 0.5, 1)
        # params['feature_union__pca__n_components'] = scope.int(hp.quniform('feature_union__pca__n_components', 2, 15, 1))
    
    if stack_model=='full_stack' and run_params['opt_type']=='bayes': 
        params[f'{sp}__percentile'] = hp.uniform(f'{sp}__percentile', 50, 100)
        # params[f'{sp}__percentile'] = skm.param_range('int', 20, 100, 5, bayes_rand, sp),
    
    elif stack_model=='random_full_stack' and run_params['opt_type']=='rand':
        params['random_sample__frac'] = np.arange(0.5, 1, 0.05)

    elif stack_model=='random_full_stack_newp' and run_params['opt_type']=='optuna':
        params['random_sample__frac'] = ['real', 0.2, 1]
        params['feature_union__agglomeration__n_clusters'] = ['int', 3, 15]
        params['feature_union__pca__n_components'] = ['int', 3, 15]
        params[f'feature_union__{kb}_fu__k'] = ['int', 3, 50]
        params[f'{kb}__k'] = ['int', 5, 50]

    elif stack_model=='random_kbest_newp' and run_params['opt_type']=='optuna':
        params['random_sample__frac'] = ['real', 0.2, 1]
        params[f'{kb}__k'] = ['int', 5, 50]
 
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


def fit_and_predict(m_label, m, df_predict, X, y, proba):

    try:
    
        cols = m.steps[0][-1].columns
        cols = [c for c in cols if c in X.columns]
        X = X[cols]
        X_predict = df_predict[cols]
        m = Pipeline(m.steps[1:])
    except:
        X_predict = df_predict[X.columns]
        
    try:
        m.fit(X,y)

        if proba: cur_predict = m.predict_proba(X_predict)[:,1]
        else: cur_predict = m.predict(X_predict)
    
    except:
        cur_predict = []

    cur_predict = pd.DataFrame(cur_predict, columns=['pred'])
    cur_predict['model'] = m_label

    return cur_predict



def create_stack_predict(df_predict, models, X, y, proba=False):
    # create the full stack pipe with meta estimators followed by stacked model
    all_models = []
    for k, ind_models in models.items():
        for m in ind_models:
            all_models.append([k, m])

    predictions = Parallel(n_jobs=-1, verbose=0)(delayed(fit_and_predict)(model_name, m, df_predict, X, y, proba) for model_name, m in all_models)
    preds = pd.concat([p for p in predictions], axis=0)
    X_predict = pd.pivot_table(preds, values='pred', index=preds.index,columns='model', aggfunc='mean')
    X_predict = X_predict.rename_axis(None, axis=1)

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
    X_predict = create_stack_predict(df_predict, models_mil, X, y, proba=True)
    X_predict = pd.concat([df_predict[['player', 'team', 'week', 'year']], X_predict], axis=1)
    return X_predict


def stack_predictions(X_predict, best_models, final_models, model_obj='reg'):
    
    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):

        try: start_cols = bm.steps[0][1].start_columns
        except: start_cols = X_predict.columns

        X_predict = X_predict[start_cols]
        
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


def weighted_average_models(scores, y_stack, stack_val_pred, predictions, model_obj):
    
    skm, _, _ = get_skm(df_train, model_obj=model_obj, to_drop=[])
    # scores = np.argsort(np.argsort([-s for s in scores]))
    scores = np.array(scores)
    scores = -(scores - np.max(scores))
    
    print(scores)
    best_val =(stack_val_pred*scores).sum(axis=1)/np.sum(scores)
    best_predictions = (predictions*scores).sum(axis=1)/np.sum(scores)
    best_score = skm.custom_score(y_stack, best_val.values)
    print(best_score)

    best_val =  pd.DataFrame(best_val, columns=['wt_avg'])
    best_predictions =  pd.DataFrame(best_predictions, columns=['wt_avg_pred'])

    return best_val, best_predictions, best_score



def average_stack_models(scores, final_models, y_stack, stack_val_pred, predictions, model_obj, show_plot=True, min_include=3):

    # best_val, best_predictions, best_score = weighted_average_models(scores, y_stack, stack_val_pred, predictions, model_obj)

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

    output.loc[output.std_dev < 0, 'std_dev'] = 1

    output.loc[(output.max_score < output.pred_fp_per_game), 'max_score'] = \
        output.loc[(output.max_score < output.pred_fp_per_game), 'pred_fp_per_game'] * 2
    
    output.loc[output.max_score < 0, 'max_score'] = 1
    
    output.loc[(output.pred_fp_per_game < output.min_score), 'min_score'] = \
        output.loc[(output.pred_fp_per_game < output.min_score), 'pred_fp_per_game'] / 2
    
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

    scores = dm.read("SELECT * FROM Scores_Lines", f"Model_Features_{run_params['set_year']}")
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


def add_sal_columns(df, df_sal, run_params):

    if 'team_stats' in run_params['million_ens_vers']: sal_multiply = 1000
    else: sal_multiply = 1

    df = pd.merge(df, df_sal[['player', 'week', 'year', 'dk_salary']], on=['player', 'week', 'year'])

    for c in df.columns:
        if 'expert' in c: df[c+'_salary'] = df[c] * df.dk_salary
        if 'rank' in c: df[c+'_salary'] = df[c] * df.dk_salary
        if 'proj' in c: df[c+'_salary'] = df[c] / df.dk_salary
        if 'ffa' in c: df[c+'_salary'] = df[c] / df.dk_salary
        if 'reg' in c: df[c+'_salary'] = sal_multiply*df[c] / df.dk_salary
        if 'quant' in c: df[c+'_salary'] = sal_multiply*df[c] / df.dk_salary
        if 'implied' in c: df[c+'_salary'] = sal_multiply*df[c] / df.dk_salary
    
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
    save_mil_to_db(test_output, run_params, 'Predicted_Million')

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

    if 'Million' in table_name: ens_vers = 'million_ens_vers'
    else: ens_vers = 'reg_ens_vers'

    df['pos'] = set_pos
    df['pred_vers'] = run_params['pred_vers']
    df[ens_vers] = run_params[ens_vers]
    df['model_type'] = model_type
    df['set_week'] = run_params['set_week']
    df['set_year'] = run_params['set_year']

    df = df[['player', 'team', 'year', 'week',  'y_act', 'pred_fp_per_game', 'pos',
             'pred_vers', ens_vers, 'model_type', 'set_week', 'set_year']]

    del_str = f'''pos='{set_pos}' 
                  AND pred_vers='{run_params['pred_vers']}'
                  AND {ens_vers}='{run_params[ens_vers]}' 
                  AND set_week={run_params['set_week']} 
                  AND set_year={run_params['set_year']}
                  AND model_type='{model_type}'
                '''
    
    dm.delete_from_db('Validations', table_name, del_str, create_backup=False)
    dm.write_to_db(df, 'Validations', table_name, 'append')

  


def save_mil_to_db(output, run_params, table_name):

    
    df = output[['player', 'team', 'week', 'year', 'pred_fp_per_game_class']].copy()
    df = df.rename(columns={'pred_fp_per_game_class': 'pred_fp_per_game'})
    df['std_dev'] = df.pred_fp_per_game * 0.25
    df['min_score'] = 0
    df['max_score'] = df.pred_fp_per_game * 1.5
    df['max_score'] = df.max_score.apply(lambda x: np.min([x, 1]))

    df['pos'] = set_pos
    df['pred_vers'] = run_params['pred_vers']
    df['million_ens_vers'] = run_params['million_ens_vers']
    df['model_type'] = model_type
    df['week'] = run_params['set_week']
    df['year'] = run_params['set_year']
    df['date_run'] = dt.datetime.now().strftime('%m-%d-%Y %H:%M')

    df = df[['player', 'week', 'year', 'pos', 'pred_fp_per_game', 
             'std_dev', 'min_score', 'max_score',
             'pred_vers', 'million_ens_vers', 'model_type', 'date_run']]

    del_str = f'''pos='{set_pos}' 
                  AND pred_vers='{run_params['pred_vers']}'
                  AND million_ens_vers='{run_params['million_ens_vers']}' 
                  AND week={run_params['set_week']} 
                  AND year={run_params['set_year']}
                  AND model_type='{model_type}'
                '''
    dm.delete_from_db('Simulation', table_name, del_str, create_backup=False)
    dm.write_to_db(df, 'Simulation', table_name, 'append')


def save_output_to_db(output, run_params):

    output['pos'] = set_pos
    output['pred_vers'] = run_params['pred_vers']
    output['reg_ens_vers'] = run_params['reg_ens_vers']
    output['std_dev_type'] = std_dev_type
    output['model_type'] = model_type
    output['week'] = run_params['set_week']
    output['year'] = run_params['set_year']
    output['date_run'] = dt.datetime.now().strftime('%m-%d-%Y %H:%M')

    output = output[['player', 'dk_salary', 'pred_fp_per_game', 'std_dev',
                     'dk_rank', 'pos', 'pred_vers', 'model_type', 'max_score', 'min_score',
                     'week', 'year', 'reg_ens_vers', 'std_dev_type', 'date_run']]

    del_str = f'''pos='{set_pos}' 
                AND pred_vers='{run_params['pred_vers']}'
                AND reg_ens_vers='{run_params['reg_ens_vers']}' 
                AND std_dev_type='{std_dev_type}'
                AND week={run_params['set_week']} 
                AND year={run_params['set_year']}
                AND model_type='{model_type}'
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


def get_trial_times(root_path, fname, run_params, set_pos, model_type):

    yr = run_params['set_year']
    wk = run_params['set_week']
    if wk == 1: 
        wk = 16
        yr -= 1
    else:
        wk -= 1

    pred_vers = run_params['pred_vers']
    recent_save = f"{root_path}/Model_Outputs/{yr}/{set_pos}_year{yr}_week{wk}_{model_type}{pred_vers}"

    all_trials = load_pickle(recent_save, fname)['trials']

    times = []
    for k,v in all_trials.items():
        if k!='reg_adp':
            max_trial = len(v.trials) - 1
            trial_times = []
            for i in range(max_trial-50, max_trial):
                trial_times.append(v.trials[i]['refresh_time'] - v.trials[i]['book_time'])
            trial_time = np.mean(trial_times).seconds
            times.append([k, np.round(trial_time / 60, 4)])

    time_per_trial = pd.DataFrame(times, columns=['model', 'time_per_trial']).sort_values(by='time_per_trial', ascending=False)
    time_per_trial['total_time'] = time_per_trial.time_per_trial * 50
    return time_per_trial


def calc_num_trials(time_per_trial, run_params):

    n_iters = run_params['n_iters']
    time_per_trial['percentile_90_time'] = time_per_trial.time_per_trial.quantile(0.6)
    time_per_trial['num_trials'] = n_iters * (time_per_trial.percentile_90_time + 0.001) / (time_per_trial.time_per_trial +  0.001)
    time_per_trial['num_trials'] = time_per_trial.num_trials.apply(lambda x: np.min([n_iters, np.max([x, n_iters/10])])).astype('int')
    
    return {k:v for k,v in zip(time_per_trial.model, time_per_trial.num_trials)}


def get_proba_adp_coef(model_obj, final_m, run_params):
    if model_obj == 'class': proba = True
    else: proba = False

    if model_obj in ('class', 'quantile'): run_adp = False
    else: run_adp = True

    if ('gbmh' in final_m 
        or 'knn' in final_m 
        or 'full_stack' in run_params['stack_model'] 
        or run_params['opt_type']=='bayes'): print_coef = False
    else: print_coef = run_params['print_coef']

    return proba, run_adp, print_coef

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

def get_last_run_week(w, run_params):
    if w == 1: 
        run_params['last_run_year'] = run_params['set_year'] - 1
        run_params['last_run_week'] = 16
    else: 
        run_params['last_run_year'] = run_params['set_year']
        run_params['last_run_week'] = w - 1
    
    return run_params

def get_trials(fname, final_m, bayes_rand):

    yr = run_params['set_year']
    wk = run_params['set_week']
    if wk == 1: 
        wk = 16
        yr -= 1
    else:
        wk -= 1
    pred_vers = run_params['pred_vers']
    recent_save = f"{root_path}/Model_Outputs/{yr}/{set_pos}_year{yr}_week{wk}_{model_type}{pred_vers}"

    if recent_save is not None and bayes_rand=='bayes': 
        try:
            trials = load_pickle(recent_save, fname)
            trials = trials['trials'][final_m]
            print('Loading previous trials')
            trials = get_recent_trials(trials, run_params['num_recent_trials'])
        except:
            print('No Previous Trials Exist')
            trials = Trials()

    elif bayes_rand=='bayes':
        print('Creating new Trials object')
        trials = Trials()

    else:
        trials = None

    return trials


def rename_existing(old_study_db, new_study_db, study_name):

    import datetime as dt
    new_study_name = study_name + '_' + dt.datetime.now().strftime('%Y%m%d%H%M%S')
    optuna.copy_study(from_study_name=study_name, from_storage=old_study_db, to_storage=new_study_db, to_study_name=new_study_name)
    optuna.delete_study(study_name=study_name, storage=new_study_db)


def get_new_study(old_db, new_db, old_name, new_name, num_trials):

    old_storage = optuna.storages.RDBStorage(
                                url=old_db,
                                engine_kwargs={"pool_size": 32, 
                                            "connect_args": {"timeout": 60},
                                            },
                                )
    
    new_storage = optuna.storages.RDBStorage(
                                url=new_db,
                                engine_kwargs={"pool_size": 32, 
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
    
def get_optuna_study(fname, final_m, run_params):
    last_db = f"{run_params['last_study_db']}_{final_m}.sqlite3"
    new_db = f"{run_params['study_db']}_{final_m}.sqlite3"
    old_name = f"{final_m}_{fname}_{run_params['set_pos']}_{run_params['model_type']}_{run_params['last_run_year']}_{run_params['last_run_week']}"
    new_name = f"{final_m}_{fname}_{run_params['set_pos']}_{run_params['model_type']}_{run_params['set_year']}_{run_params['set_week']}"
    next_study = get_new_study(last_db, new_db, old_name, new_name, run_params['num_recent_trials'])
    return next_study

# 'class_random_full_stack_newp_c80_matt0_brier1_include1_kfold3'
# 'gbm_c'
def run_stack_models(fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params, num_trials, is_million):

    print(f'\n{final_m}')

    min_samples = int(len(y_stack)/10)
    proba, run_adp, print_coef = get_proba_adp_coef(model_obj, final_m, run_params)

    skm, _, _ = get_skm(pd.concat([X_stack, y_stack], axis=1), model_obj, to_drop=[])

    if is_million: sm = run_params['stack_model_million']
    else: sm = run_params['stack_model']
    pipe, params = get_full_pipe(skm, final_m, stack_model=sm, alpha=alpha, 
                                 min_samples=min_samples, bayes_rand=run_params['opt_type'])
    
    
    if run_params['opt_type'] == 'bayes': 
        trials = get_trials(fname, final_m, run_params['opt_type'])
        trials = update_trials_params(trials, final_m, params, pipe)
    elif run_params['opt_type'] == 'optuna': 
        trials = get_optuna_study(fname, final_m, run_params)
    else: trials = None

    try: n_iter = num_trials[final_m]
    except: n_iter = run_params['n_iters']
        
    best_model, stack_scores, stack_pred, trial = skm.best_stack(pipe, params, X_stack, y_stack, 
                                                                n_iter=n_iter, alpha=alpha,
                                                                trials=trials, bayes_rand=run_params['opt_type'],
                                                                run_adp=run_adp, print_coef=print_coef,
                                                                proba=proba, num_k_folds=run_params['num_k_folds'],
                                                                random_state=(i*2)+(i*7), optuna_timeout=run_params['optuna_timeout'])
    
    return best_model, stack_scores, stack_pred, trial

def get_func_params(model_obj, alpha):

    model_list = {
        'reg': ['rf', 'gbm', 'gbmh', 'mlp', 'cb', 'huber', 'xgb', 'lgbm', 'knn', 'ridge', 'lasso', 'bridge', 'enet'],
        'class': ['rf_c', 'gbm_c', 'gbmh_c', 'xgb_c','lgbm_c', 'knn_c', 'lr_c', 'mlp_c', 'cb_c'],
        'quantile': ['qr_q', 'gbm_q', 'lgbm_q', 'gbmh_q', 'rf_q', 'cb_q']#, 'knn_q']
    }

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

def remove_knn_rf_q(X):
    return X[[c for c in X.columns if 'knn_q' not in c and 'rf_q' not in c]]


def remove_low_preds(predictions, stack_val_pred, model_list, scores):
    
    preds_mean_check = pd.DataFrame(predictions.median(), columns=['preds'])
    val_mean_check = pd.DataFrame(stack_val_pred.median(), columns=['vals'])
    mean_checks = pd.merge(preds_mean_check, val_mean_check, left_index=True, right_index=True)
    mean_checks['pct_diff'] = (mean_checks.preds - mean_checks.vals) / (mean_checks.vals + 0.1)
    
    print(mean_checks)
    models_pre = mean_checks.index
    for cut in np.arange(0.2, 20, 0.2):
        mean_checks_idx = mean_checks[abs(mean_checks.pct_diff) <= cut].index
        if len(mean_checks_idx) >= run_params['min_include']: break

    print("models removed:", [m for m in models_pre if m not in mean_checks_idx])

    # auto remove any predictions that are negative or 0
    good_col = []
    good_idx = []

    for i, col in enumerate(predictions.columns):
        if col in mean_checks_idx:
            good_col.append(col)
            good_idx.append(i)

    predictions = predictions[good_col]
    stack_val_pred = stack_val_pred[good_col]
    model_list = list(np.array(model_list)[good_idx])
    scores = list(np.array(scores)[good_idx])

    return predictions, stack_val_pred, model_list, scores


def load_run_models(run_params, X_stack, y_stack, X_predict, model_obj, alpha=None, is_million=False):
    
    if model_obj=='reg': ens_vers = run_params['reg_ens_vers']
    elif model_obj=='class': ens_vers = run_params['class_ens_vers']
    elif model_obj=='quantile': ens_vers = run_params['quant_ens_vers']

    if is_million: 
        model_obj_label = 'million'
        ens_vers = run_params['million_ens_vers']
    else: 
        model_obj_label = model_obj

    path = run_params['model_output_path']
    fname = f"{model_obj_label}_{ens_vers}"    
    model_list, func_params = get_func_params(model_obj, alpha)

    try:
        time_per_trial = get_trial_times(root_path, fname, run_params, set_pos, model_type)
        print(time_per_trial)
        num_trials = calc_num_trials(time_per_trial, run_params)
    except: 
        num_trials = {m: run_params['n_iters'] for m in model_list}
    print(num_trials)

    print(path, fname)

    if os.path.exists(f"{path}/{fname}.p"):
        best_models, scores, stack_val_pred = load_stack_runs(path, fname)
    
    else:
        results = Parallel(n_jobs=-1, verbose=50)(
                        delayed(run_stack_models)
                        (fname, final_m, i, model_obj, alpha, X_stack, y_stack, run_params, num_trials, is_million) 
                        for final_m, i, model_obj, alpha in func_params
                        )
        best_models, scores, stack_val_pred, trials = unpack_results(model_list, results)
        save_stack_runs(path, fname, best_models, scores, stack_val_pred, trials)

    X_predict = X_predict[X_stack.columns]
    predictions = stack_predictions(X_predict, best_models, model_list, model_obj=model_obj)
    # predictions, stack_val_pred, model_list, scores = remove_low_preds(predictions, stack_val_pred, model_list, scores)
    
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
        show_calibration_curve(mil_results['y_act'], mil_results['pred_mil'], n_bins=5)
        display(mf.show_scatter_plot(mil_results['pred_mil'], mil_results['y_act'], r2=True))
    except:
        display(output_mil.sort_values(by='pred_mil', ascending=False).iloc[:50])

    return output_mil
            

def display_output(output, show_plot=True):
     
    try:  
        output = add_actual(output)
        output = output[~output.week.isin([17,18])].reset_index(drop=True)
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
        
#----------------
# Team Functions
#----------------
        
def concatenate_team_data(backfill_runs):
    team_data = pd.DataFrame()
    for pos in backfill_runs.keys():
        team_data = pd.concat([team_data, backfill_runs[pos]['X_stack_player'].assign(pos=pos)], axis=0)
        team_data = pd.concat([team_data, backfill_runs[pos]['X_predict_player'].assign(pos=pos)], axis=0)
    return team_data

def sort_and_rank_team_data(team_data):
    team_data = team_data.sort_values(by=['team', 'year', 'week', 'reg_adp'],
                                      ascending=[True, True, True, False]).reset_index(drop=True)
    team_data['team_rank'] = team_data.groupby(['team', 'year', 'week']).cumcount()
    team_data['team_pos_rank'] = team_data.groupby(['team', 'pos', 'year', 'week']).cumcount()
    return team_data

def summarize_team_data(team_data):
    ignore_cols = ['player', 'team', 'week', 'year', 'pos', 'team_rank', 'team_pos_rank']
    team_sum = team_data[team_data.team_rank <= 5].groupby(['team', 'year', 'week']).agg({c:'sum' for c in team_data.columns if c not in ignore_cols})
    
    for model_stat in ['reg', 'class_95', 'class_80', 'class_33', 'quant_0.8', 'quant_0.95']:
        stat_cols = [c for c in team_sum.columns if model_stat in c]
        team_sum[f'team_sum_{model_stat}'] = team_sum[stat_cols].mean(axis=1)
        team_sum = team_sum.drop(stat_cols, axis=1)
        
    return team_sum.reset_index()

def merge_team_sum(X, team_sum):
    X = pd.merge(X, team_sum, on=['team','week','year'],how='left')
    X = X.sort_values(by=['team', 'year', 'week']).reset_index(drop=True)
    X = X.groupby(['team']).apply(lambda x: x.ffill()).reset_index(drop=True)
    return X

def calc_team_frac(X):
    for model_stat in ['reg', 'class_95', 'class_80', 'class_33', 'quant_0.8', 'quant_0.95']:
        stat_cols = [c for c in X.columns if model_stat in c and 'team_sum' not in c]
        X[f'team_frac_{model_stat}'] = X[stat_cols].mean(axis=1) / X[f'team_sum_{model_stat}']
    return X

#%%
#==========
# General Setting
#==========

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


def process_config(config):
    config['pred_vers'] = config['pred_vers'].format(**config['pred_params'])
    # config['ensemble_vers'] = config['ensemble_vers'].format(**config['ensemble_params'])
    # config['std_dev_type'] = config['std_dev_type'].format(**config['std_dev_params'])

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
    'set_year': 2024,

    # set beginning of validation period
    'val_year_min': 2020,
    'val_week_min': 1,

    'cuts': [33, 80, 95],

    'stack_model': 'random_full_stack_newp',
    'stack_model_million': 'random_full_stack_newp',

    # opt params
    'n_iters': 50,
    
    'n_splits': 5,
    'num_k_folds': 3,
    'show_plot': True,
    'print_coef': True,
    'min_include': 2,

    # other parameters
    'use_sample_weight': False,
    
    'met': 'y_act',   

    'opt_type': 'optuna',
    'hp_algo': 'tpe',
    'num_recent_trials': 100,
    'optuna_timeout': 60*2.5
}

s_mod = run_params['stack_model']
mil_mod = run_params['stack_model_million']
min_inc = run_params['min_include']
kfold = run_params['num_k_folds']

r2_wt = 0
sera_wt = 0
mse_wt = 1
brier_wt = 1
matt_wt = 0
log_wt = 0

alpha = 80
class_cut = 80

# set_weeks = [1,2,3,4]
# set_weeks = [5,6,7,8]
# set_weeks = [9,10,11,12]
# set_weeks = [13,14,15,16]
set_weeks = [12]

pred_vers = 'sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_higherkb'
reg_ens_vers = f"{s_mod}_sera{sera_wt}_rsq{r2_wt}_mse{mse_wt}_include{min_inc}_kfold{kfold}"
quant_ens_vers = f"{s_mod}_q{alpha}_include{min_inc}_kfold{kfold}"
class_ens_vers = f"{s_mod}_c{class_cut}_matt{matt_wt}_brier{brier_wt}_include{min_inc}_kfold{kfold}"
million_ens_vers = f"{mil_mod}_matt{matt_wt}_brier{brier_wt}_include{min_inc}_kfold{kfold}"

run_params['pred_vers'] = pred_vers
run_params['reg_ens_vers'] = reg_ens_vers
run_params['quant_ens_vers'] = quant_ens_vers
run_params['class_ens_vers'] = class_ens_vers
run_params['million_ens_vers'] = million_ens_vers

std_dev_types = [
    f'spline_pred_class{class_cut}_q{alpha}_matt{matt_wt}_brier{brier_wt}_kfold{kfold}',
    f'spline_pred_class{class_cut}_matt{matt_wt}_brier{brier_wt}_kfold{kfold}',
    f'spline_pred_q{alpha}_matt{matt_wt}_brier{brier_wt}_kfold{kfold}',
    f'spline_class{class_cut}_q{alpha}_matt{matt_wt}_brier{brier_wt}_kfold{kfold}'
]

std_dev_type = std_dev_types[0]

with keep.running() as m:
    
    if not m.success:
        print('Fell Asleep')
            
    for w in set_weeks:
        run_params['set_week'] = w
        run_params = get_last_run_week(w, run_params)

        if not os.path.exists(f"{root_path}/Scripts/Modeling/optuna/{run_params['set_year']}/week{run_params['set_week']}/"):
            os.makedirs(f"{root_path}/Scripts/Modeling/optuna/{run_params['set_year']}/week{run_params['set_week']}/")
        if not os.path.exists(f"{root_path}/Scripts/Modeling/optuna/{run_params['last_run_year']}/week{run_params['last_run_week']}/"):
            os.makedirs(f"{root_path}/Scripts/Modeling/optuna/{run_params['last_run_year']}/week{run_params['last_run_week']}/")

        run_params['last_study_db'] = f"sqlite:///optuna/{run_params['last_run_year']}/week{run_params['last_run_week']}/weekly_predict"
        run_params['study_db'] = f"sqlite:///optuna/{run_params['set_year']}/week{run_params['set_week']}/weekly_predict"

        if 'team_stats' in run_params['stack_model']:
            vegas_scores = dm.read("SELECT * FROM Scores_Lines", f"Model_Features_{run_params['set_year']}")
            vegas_scores = vegas_scores[['team', 'week', 'year', 'over_under', 'implied_points_for', 'implied_points_against', 'is_home']]

            model_type = 'backfill'
            run_params['rush_pass'] = ''

            backfill_runs = {}
            for set_pos in ['QB', 'RB', 'WR', 'TE']:
             
                # load data and filter down
                pkey, run_params, model_output_path = create_pkey_output_path(set_pos, run_params, model_type)
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
                
                X_stack = remove_knn_rf_q(X_stack)
                X_predict = remove_knn_rf_q(X_predict)
                X_predict_player = remove_knn_rf_q(X_predict_player)

                # cleanup the X and y datasets
                X_stack_player, X_stack, y_stack = cleanup_X_y(X_stack, y_stack)
                _, _, y_stack_class = cleanup_X_y(X_stack_player, y_stack_class)

                backfill_runs[set_pos] = {
                                          'X_stack_player': X_stack_player, 
                                          'X_stack': X_stack, 
                                          'y_stack': y_stack, 
                                          'y_stack_class': y_stack_class, 
                                          'X_predict': X_predict, 
                                          'X_predict_player': X_predict_player
                                          }

            # Example usage:
            team_data = concatenate_team_data(backfill_runs)
            team_data = sort_and_rank_team_data(team_data)
            team_sum = summarize_team_data(team_data)
            team_pos_rank = team_data[['player', 'team', 'week', 'year',  'team_rank', 'team_pos_rank']]

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

            run_params['rush_pass'] = rush_pass
            run_params['set_pos'] = set_pos
            run_params['model_type'] = model_type
            
            # load data and filter down
            pkey, run_params, model_output_path = create_pkey_output_path(set_pos, run_params, model_type)
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
        
            if 'team_stats' in run_params['stack_model']:      
                X_stack = merge_team_sum(X_stack, team_sum)
                X_predict_player = merge_team_sum(X_predict_player, team_sum)
                X_stack = calc_team_frac(X_stack)
                X_predict_player = calc_team_frac(X_predict_player)

                # X_stack = pd.merge(X_stack, team_pos_rank, on=['player', 'team', 'week', 'year'], how='left').fillna(5)
                # X_predict_player = pd.merge(X_predict_player, team_pos_rank, on=['player', 'team', 'week', 'year'], how='left').fillna(5)

                # X_stack = pd.merge(X_stack, vegas_scores, on=['team', 'week', 'year'], how='left').fillna(X_stack.mean())
                # X_predict_player = pd.merge(X_predict_player, vegas_scores, on=['team', 'week', 'year'], how='left').fillna(X_predict_player.mean())

                X_predict = X_predict_player.drop(['player', 'team', 'week', 'year'], axis=1).copy()

                y_stack = pd.merge(X_stack[['player', 'week', 'year']], y_stack, 
                                   on=['player', 'week', 'year']).reset_index(drop=True)
                y_stack_class = pd.merge(X_stack[['player', 'week', 'year']], y_stack_class, 
                                   on=['player', 'week', 'year']).reset_index(drop=True)
                
                output_start = pd.merge(X_predict_player[['player', 'team', 'week', 'year']], output_start, on=['player', 'team', 'week', 'year'], how='left')

            X_stack = remove_knn_rf_q(X_stack)
            X_predict = remove_knn_rf_q(X_predict)
            X_predict_player = remove_knn_rf_q(X_predict_player)

            # cleanup the X and y datasets
            X_stack_player, X_stack, y_stack = cleanup_X_y(X_stack, y_stack)
            _, _, y_stack_class = cleanup_X_y(X_stack_player, y_stack_class)


            # run the class, quant, and reg models
            best_val_class, best_predictions_class = load_run_models(run_params, X_stack, y_stack_class, X_predict, 'class')
            best_val_quant, best_predictions_quant = load_run_models(run_params, X_stack, y_stack, X_predict, 'quantile', alpha=alpha/100)
            best_val_reg, best_predictions_reg = load_run_models(run_params, X_stack, y_stack, X_predict, 'reg')
            save_val_to_db(X_stack_player, y_stack, best_val_reg, run_params, table_name='Model_Validations')

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

            X_stack_mil = remove_knn_rf_q(X_stack_mil)
            X_predict_mil = remove_knn_rf_q(X_predict_mil)

            X_stack_mil = add_sal_columns(X_stack_mil, df_train_mil, run_params)
            X_predict_mil = add_sal_columns(X_predict_mil, df_predict_mil, run_params)

            X_stack_mil = X_stack_mil.dropna(axis=0).reset_index(drop=True)
            y_stack_mil = pd.merge(X_stack_mil[['player', 'week', 'year']], 
                                   y_stack_mil[y_stack_mil.player!='Ryan Griffin'], 
                                   on=['player', 'week', 'year']).reset_index(drop=True)
            
            X_stack_mil = pd.merge(X_stack_mil, 
                                   y_stack_mil.loc[y_stack_mil.player!='Ryan Griffin', ['player','week', 'year']], 
                                   on=['player', 'week', 'year']).reset_index(drop=True)

            X_predict_mil = X_predict_mil.drop(['player', 'team', 'week', 'year'], axis=1)
            X_stack_mil = X_stack_mil.drop(['player', 'team', 'week', 'year'], axis=1)
            y_stack_mil = y_stack_mil.y_act
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

week = 1
year = 2024
model = 'lgbm_c'
model_obj = 'class'
pos = 'QB'
model_type = 'full_model'

if model_obj=='reg': ens_vers = run_params['reg_ens_vers']
elif model_obj == 'class': ens_vers = run_params['class_ens_vers']
elif model_obj=='quantile': ens_vers = run_params['quant_ens_vers']
elif model_obj=='million': ens_vers = run_params['million_ens_vers']
from optuna.visualization import plot_parallel_coordinate

study = optuna.create_study(
            study_name=f'{model}_{model_obj}_{ens_vers}_{pos}_{model_type}_{year}_{week}',
            storage=f'sqlite:///optuna/{year}/week{week}/weekly_predict_{model}.sqlite3',
            load_if_exists=True
        )

plot_parallel_coordinate(study)

#%%
import os
import shutil

def copy_and_rename_files(year, keyword):
    base_dir = r"C:\Users\borys\OneDrive\Documents\GitHub\Daily_Fantasy\Model_Outputs"
    year_dir = os.path.join(base_dir, str(year))

    for root, dirs, files in os.walk(year_dir):
        if keyword in root:  # Only process folders containing the keyword
            for file in files:
                if "all_" not in file and "include2" in file:
                    old_file_path = os.path.join(root, file)
                    new_file_name = file.replace("include2", "include1")
                    new_file_path = os.path.join(root, new_file_name)
                    shutil.copy(old_file_path, new_file_path)

# Call the function with the year and keyword you want
copy_and_rename_files(2023, "higherkb")



# %%
