#%%
# core packages
from random import Random
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt

from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from skmodel import SciKitModel

import pandas_bokeh
pandas_bokeh.output_notebook()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

pd.set_option('display.max_columns', 999)

from sklearn import set_config
set_config(display='diagram')

#==========
# General Setting
#==========

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
np.random.seed(1234)

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE', 'Defense'
for set_pos in ['QB', 'WR', 'RB','TE', 'Defense'
                ]:

    model_type = 'full_model'
    vers = '3_25_3_percentile_reg_3_25_3_perc_class_25iter'

    # set year to analyze
    set_year = 2021
    set_week = 5

    print(f'\n==================\n{set_pos} {model_type} {set_year} {set_week}\n====================')

    # set the earliest date to begin the validation set
    val_year_min = 2020
    val_week_min = 10

    met = 'y_act'

    def_cuts = [33, 75, 90]
    off_cuts = [33, 80, 95]
    if set_pos == 'Defense': cuts = def_cuts
    else: cuts = off_cuts

    #-------------
    # Set up output dataset
    #-------------

    all_vars = [set_pos, set_year, set_week]

    pkey = f'{set_pos}_year{set_year}_week{set_week}_{model_type}{vers}'
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

    def add_result_db_output(model_type, model, results, db_output):
        db_output['pkey'].append(pkey)
        db_output['set_pos'].append(set_pos)
        db_output['set_year'].append(set_year)
        db_output['set_week'].append(set_week)
        db_output['model_type'].append(model_type)
        db_output['model'].append(model)
        db_output['validation_score'].append(results[0])
        db_output['test_score'].append(results[1])

        return db_output

    model_output_path = f'{root_path}/Model_Outputs/{set_year}/{pkey}/'
    if not os.path.exists(model_output_path): os.makedirs(model_output_path)

    def save_pickle(obj, path, fname, protocol=-1):
        with gzip.open(f"{path}/{fname}.p", 'wb') as f:
            pickle.dump(obj, f, protocol)

        print(f'Saved {fname} to path {path}')

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

    #==========
    # Pull and clean compiled data
    #==========

    # load data and filter down
    if model_type=='full_model': df = dm.read(f'''SELECT * FROM {set_pos}_Data''', 'Model_Features')
    elif model_type=='backfill': df = dm.read(f'''SELECT * FROM Backfill WHERE pos='{set_pos}' ''', 'Model_Features')

    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * FROM {set_pos}_Data2''', 'Model_Features')
        df = pd.concat([df, df2], axis=1)

    drop_cols = list(df.dtypes[df.dtypes=='object'].index)
    print(drop_cols)

    # set up the date column for sorting
    def year_week_to_date(x):
        return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))

    df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
    cv_time_input = int(dt.datetime(val_year_min, 1, val_week_min).strftime('%Y%m%d'))
    train_time_split = int(dt.datetime(set_year, 1, set_week).strftime('%Y%m%d'))

    # # get the train / predict dataframes and output dataframe
    df_train = df[df.game_date < train_time_split].reset_index(drop=True)
    df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)
    df_predict = df[df.game_date == train_time_split].reset_index(drop=True)
    output_start = df_predict[['player', 'dk_salary']].copy()

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.game_date < cv_time_input].shape[0])  
    print('Shape of Train Set', df_train.shape)

    #===========================================================================================

    # set up blank dictionaries for all metrics
    pred = {}; actual = {}; scores = {}; models = {}
    met = 'y_act'

    print(f'\nRunning Metric {met}\n=========================\n')
    print('ADP only\n============\n')

    skm = SciKitModel(df_train)
    X, y = skm.Xy_split(y_metric='y_act', to_drop=drop_cols)

    # set up the ADP model pipe
    pipe = skm.model_pipe([skm.piece('feature_select'), skm.piece('std_scale'), skm.piece('lr')])
    params = skm.default_params(pipe)
    params['feature_select__cols'] = [[ 'dk_salary'], ['dk_salary', 'year'] ]

    # fit and append the ADP model
    best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                    col_split='game_date', 
                                                    time_split=cv_time_input)

    # append all of the metric outputs
    pred[f'{met}_adp'] = oof_data['combined']; actual[f'{met}_adp'] = oof_data['actual']
    scores[f'{met}_adp'] = r2; models[f'{met}_adp'] = best_models

    db_output = add_result_db_output('reg', 'adp', r2, db_output)
    #---------------
    # Model Training loop
    #---------------

    # loop through each potential model
    model_list = ['lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'rf']
    for m in model_list:

        print('\n============\n')
        print(m)

        # set up the model pipe and get the default search parameters
        pipe = skm.model_pipe([skm.piece('std_scale'), 
                                skm.piece('select_perc'),
                                skm.feature_union([
                                            skm.piece('agglomeration'), 
                                            skm.piece('k_best'),
                                            skm.piece('pca')
                                            ]),
                                skm.piece('k_best'),
                                skm.piece(m)])
        
        params = skm.default_params(pipe, 'rand')
        params['select_perc__percentile'] = range(3, 25, 3)

        if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)

        # run the model with parameter search
        best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                        col_split='game_date', time_split=cv_time_input)

        # append the results and the best models for each fold
        pred[f'{met}_{m}'] = oof_data['combined']; actual[f'{met}_{m}'] = oof_data['actual']
        scores[f'{met}_{m}'] = r2; models[f'{met}_{m}'] = best_models

        db_output = add_result_db_output('reg', m, r2, db_output)

    save_pickle(pred, model_output_path, 'reg_pred')
    save_pickle(actual, model_output_path, 'reg_actual')
    save_pickle(models, model_output_path, 'reg_models')
    save_pickle(scores, model_output_path, 'reg_scores')

    #===========================================================================================

    # set up blank dictionaries for all metrics
    pred = {}; actual = {}; scores = {}; models = {}

    for cut in cuts:

        print(f"\n--------------\nPercentile {cut}\n--------------\n")

        df_train_class, df_predict_class = get_class_data(df, cut)
        skm_class = SciKitModel(df_train_class, model_obj='class')
        X_class, y_class = skm_class.Xy_split(y_metric='y_act', 
                                            to_drop=drop_cols)

        # print the value-counts
        print('Training Value Counts:', y_class.value_counts()[0], '|', y_class.value_counts()[1])

        # loop through each potential model
        model_list = ['lr_c', 'xgb_c',  'lgbm_c', 'gbm_c', 'rf_c', 'knn_c']
        for m in model_list:

            print('\n============\n')
            print(m)

            # set up the model pipe and get the default search parameters
            pipe = skm_class.model_pipe([skm_class.piece('std_scale'), 
                                        skm_class.piece('select_perc_c'),
                                        skm_class.feature_union([
                                                            skm_class.piece('agglomeration'), 
                                                            skm_class.piece('k_best_c'), 
                                                            ]),
                                        skm_class.piece('k_best_c'),
                                        skm_class.piece(m)])
            
            params = skm_class.default_params(pipe, 'rand')
            params['select_perc_c__percentile'] = range(3, 25, 3)
            if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)

            # run the model with parameter search
            best_models, score_results, oof_data = skm_class.time_series_cv(pipe, X_class, y_class, 
                                                                            params, n_iter=25,
                                                                            col_split='game_date',
                                                                            time_split=cv_time_input)

            # append the results and the best models for each fold
            pred[f'class_{m}_{cut}'] = oof_data['combined']
            actual[f'class_{m}_{cut}'] = oof_data['actual']
            scores[f'class_{m}_{cut}'] = score_results 
            models[f'class_{m}_{cut}'] = best_models

            db_output = add_result_db_output(f'class_{cut}', m, score_results, db_output)

        save_pickle(pred, model_output_path, 'class_pred')
        save_pickle(actual, model_output_path, 'class_actual')
        save_pickle(models, model_output_path, 'class_models')
        save_pickle(scores, model_output_path, 'class_scores')


    #------------
    # Make the Class Predictions
    #------------

    pred_class = load_pickle(model_output_path, 'class_pred')
    actual_class = load_pickle(model_output_path, 'class_actual')
    models_class = load_pickle(model_output_path, 'class_models')
    scores_class = load_pickle(model_output_path, 'class_scores')

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict_class = pd.DataFrame()
    for cut in cuts:

        print(f"\n--------------\nPercentile {cut}\n--------------\n")

        df_train_class, df_predict_class = get_class_data(df, cut)

        skm_class_final = SciKitModel(df_train_class, model_obj='class')
        X_stack_class, y_stack_class = skm_class_final.X_y_stack('class', pred_class, actual_class)
        X_class_final, y_class_final = skm_class_final.Xy_split(y_metric='y_act', to_drop=drop_cols)
        
    pred = load_pickle(model_output_path, 'reg_pred')
    actual = load_pickle(model_output_path, 'reg_actual')
    models = load_pickle(model_output_path, 'reg_models')
    scores = load_pickle(model_output_path, 'reg_scores')

    #------------
    # Make the Regression Predictions
    #------------

    output = output_start[['player', 'dk_salary']].copy()

    df_predict_stack = df_predict.copy()
    df_predict_stack = df_predict_stack.drop('y_act', axis=1).fillna(0)
    skm_stack = SciKitModel(df_train)

    # get the X and y values for stack trainin for the current metric
    X_stack, y_stack = skm_stack.X_y_stack(met, pred, actual)
    X_stack = pd.concat([X_stack, X_stack_class], axis=1)

    best_models = []
    final_models = [
                    'ridge',
                    'lasso',
                    'lgbm', 
                    'xgb', 
                    'rf', 
                    'bridge'
                    ]
    for final_m in final_models:

        print(f'\n{final_m}')

        # get the model pipe for stacking setup and train it on meta features
        if final_m in ['ridge', 'lasso', 'bridge']:
            stack_pipe = skm_stack.model_pipe([
                                    skm_stack.piece('std_scale'), 
                                    skm_stack.piece('k_best'), 
                                    skm_stack.piece(final_m)
                                ])
            stack_params = skm_stack.default_params(stack_pipe)
            stack_params['k_best__k'] = range(1, X_stack.shape[1])

        else:
            stack_pipe = skm_stack.model_pipe([
                                    skm_stack.piece(final_m)
                                ])
            stack_params = skm_stack.default_params(stack_pipe)

        best_model, stack_score, adp_score = skm_stack.best_stack(stack_pipe, stack_params,
                                                                X_stack, y_stack, n_iter=50, 
                                                                run_adp=True, print_coef=True)

        db_output = add_result_db_output('final', final_m, [adp_score, stack_score], db_output)

    # write out tracking results to tracking DB
    db_output_pd = pd.DataFrame(db_output)
    dm.delete_from_db('Results', 'Model_Tracking',f"pkey='{pkey}'")
    dm.write_to_db(db_output_pd, 'Results', 'Model_Tracking', 'append')

# %%

for i in range(10):

    x = 2
    # %reset -f

x
# %%
