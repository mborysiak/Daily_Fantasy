#%%
# core packages
from operator import mod
from random import Random
from threading import current_thread
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt
import matplotlib.pyplot as plt
import gc

from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from skmodel import SciKitModel
import sys

import pandas_bokeh
pandas_bokeh.output_notebook()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

pd.set_option('display.max_columns', 999)
# from sklearn import set_config
# set_config(display='diagram')

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
set_week = 17
# set_week = int(sys.argv[1])

# model_type = 'full_model'
vers = 'standard_proba_sera_brier_lowsample'
use_sample_weight = False
opt_type = 'custom_rand'

n_iters = 50
to_keep = 25

drop_words = ['ProjPts', 'recv', 'fantasyPoints', 'expert', 'fp_rank', 'proj', 'projected_points', 'salary']
keep_words = ['def', 'qb', 'team']

for model_type in ['full_model', 'backfill']:

    if model_type == 'full_model': positions = ['QB', 'RB', 'WR', 'TE', 'Defense']
    elif model_type == 'backfill': positions =  ['QB', 'RB', 'WR', 'TE']

    for set_pos in positions:

        # set the earliest date to begin the validation set
        val_year_min = 2020
        if set_pos in ('WR', 'TE'): val_week_min = 10
        else: val_week_min = 10

        kbs = {
            'QB': [25, 200, 5],
            'RB': [25, 200, 5],
            'WR': [25, 200, 5],
            'TE': [25, 200, 5],
            'Defense': [25, 200, 5]
        }

        prc = {
            'QB': [5, 35, 5],
            'RB': [5, 35, 5],
            'WR': [5, 35, 5],
            'TE': [5, 35, 5],
            'Defense': [3, 20, 3]
        }
        
        print(f'\n==================\n{set_pos} {model_type} {set_year} {set_week} {vers}\n====================')

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

        def load_data(model_type, set_pos):

            # load data and filter down
            if model_type=='full_model': df = dm.read(f'''SELECT * FROM {set_pos}_Data''', 'Model_Features')
            elif model_type=='backfill': df = dm.read(f'''SELECT * FROM Backfill WHERE pos='{set_pos}' ''', 'Model_Features')

            if df.shape[1]==2000:
                df2 = dm.read(f'''SELECT * FROM {set_pos}_Data2''', 'Model_Features')
                df = pd.concat([df, df2], axis=1)

            df = df.sort_values(by=['year', 'week']).reset_index(drop=True)
            if set_pos == 'Defense':
                df['team'] = df.player
                
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


        def save_param_scores(df, obj_type, model):
            df = df.assign(model=model, year=set_year, week=set_week, pos=set_pos, model_type=model_type)
            exist = dm.read(f"SELECT * FROM {obj_type}_{model}", 'Results')
            df = pd.concat([exist, df], axis=0, sort=False)
            dm.write_to_db(df, 'Results', f'{obj_type}_{model}', 'replace')


        def update_output_dict(label, m, suffix, out_dict, oof_data, best_models):

            # append all of the metric outputs
            lbl = f'{label}_{m}{suffix}'
            out_dict['pred'][lbl] = oof_data['hold']
            out_dict['actual'][lbl] = oof_data['actual']
            out_dict['scores'][lbl] = oof_data['scores']
            out_dict['models'][lbl] = best_models
            out_dict['full_hold'][lbl] = oof_data['full_hold']

            return out_dict


        def save_output_dict(out_dict, label):

            save_pickle(out_dict['pred'], model_output_path, f'{label}_pred')
            save_pickle(out_dict['actual'], model_output_path, f'{label}_actual')
            save_pickle(out_dict['models'], model_output_path, f'{label}_models')
            save_pickle(out_dict['scores'], model_output_path, f'{label}_scores')
            save_pickle(out_dict['full_hold'], model_output_path, f'{label}_full_hold')


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

        def update_feature_drop(params):
            if model_type == 'backfill':
                params['feature_drop__col'] = [None]

            elif set_pos in ('QB', 'RB'):
                params['feature_drop__col'] = [list(np.random.choice(X.columns, int(0.5*X.shape[1]), replace=False)) for _ in range(10)]
            else:
                to_drop = [c for c in X.columns if any(dw in c for dw in drop_words) and not any(kw in c for kw in keep_words)]
                params['feature_drop__col'] = [list(np.random.choice(to_drop, len(to_drop)-to_keep, replace=False)) for _ in range(10)]
            return params

        #==========
        # Pull and clean compiled data
        #==========

        # load data and filter down
        df, drop_cols = load_data(model_type, set_pos)
        df, cv_time_input, train_time_split = create_game_date(df)
        df_train, df_predict, output_start, min_samples = train_predict_split(df, train_time_split, cv_time_input)

        #===========================================================================================

        # set up blank dictionaries for all metrics
        out_dict = {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}
        met = 'y_act'

        print(f'\nRunning Metric {met}\n=========================\n')
        print('ADP only\n=========================\n')

        skm = SciKitModel(df_train)
        X, y = skm.Xy_split(y_metric='y_act', to_drop=drop_cols)

        # set up the ADP model pipe
        pipe = skm.model_pipe([skm.piece('std_scale'), 
                               skm.piece('k_best'),
                               skm.piece('lr')])
        params = skm.default_params(pipe, bayes_rand='rand')
        params['k_best__k'] = range(1,5)

        # fit and append the ADP model
        adp_cols = ['game_date', 'ProjPts', 'dk_salary', 'fantasyPoints', 'year', 'week']
        best_models, oof_data, param_scores = skm.time_series_cv(pipe, X[adp_cols], y, 
                                                                 params, n_iter=25,
                                                                 col_split='game_date', 
                                                                 time_split=cv_time_input,
                                                                 sample_weight=use_sample_weight, 
                                                                 bayes_rand=opt_type,
                                                                 random_seed=1234)

        save_param_scores(param_scores, 'reg', 'adp')
        out_dict = update_output_dict('reg', 'adp', '', out_dict, oof_data, best_models)
        db_output = add_result_db_output('reg', 'adp', oof_data['scores'], db_output)

        #---------------
        # Model Training loop
        #---------------

        # loop through each potential model
        model_list = ['lgbm', 'ridge', 'svr', 'lasso', 'enet','rf', 'xgb', 'gbm']
        if not use_sample_weight: model_list.append('knn')
        
        for i, m in enumerate(model_list):

            print('\n=========================\n')
            print(m)

            # set up the model pipe and get the default search parameters
            pipe = skm.model_pipe([ skm.piece('random_sample'),
                                    skm.piece('std_scale'), 
                                    #skm.piece('select_perc'),
                                    skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best'),
                                                ]),
                                    skm.piece('k_best'),
                                    skm.piece(m)])
            
            # set params
            params = skm.default_params(pipe, 'rand')

            # params = update_feature_drop(params)
            if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)

            # run the model with parameter search
            best_models, oof_data, param_scores = skm.time_series_cv(pipe, X, y, params, n_iter=n_iters,
                                                                    col_split='game_date', 
                                                                    time_split=cv_time_input,
                                                                    bayes_rand=opt_type,
                                                                    sample_weight=use_sample_weight,
                                                                    random_seed=(i+7)*19+(i*12)+6)

            # append the results and the best models for each fold
            save_param_scores(param_scores, 'reg', m)
            out_dict = update_output_dict('reg', m, '', out_dict, oof_data, best_models)
            db_output = add_result_db_output('reg', m, oof_data['scores'], db_output)
            gc.collect()

        save_output_dict(out_dict, 'reg')

        #===========================================================================================

        # set up blank dictionaries for all metrics
        out_dict = {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}

        for cut in cuts:

            print(f"\n--------------\nPercentile {cut}\n--------------\n")

            df_train_class, df_predict_class = get_class_data(df, cut)
            skm_class = SciKitModel(df_train_class, model_obj='class')
            X_class, y_class = skm_class.Xy_split(y_metric='y_act', 
                                                to_drop=drop_cols)

            # print the value-counts
            print('Training Value Counts:', y_class.value_counts()[0], '|', y_class.value_counts()[1])

            # loop through each potential model
            model_list = ['lr_c', 'lgbm_c', 'xgb_c', 'gbm_c', 'rf_c', 'knn_c']
            for i, m in enumerate(model_list):

                print('\n============\n')
                print(m)

                # set up the model pipe and get the default search parameters
                pipe = skm_class.model_pipe([skm_class.piece('random_sample'),
                                            skm_class.piece('std_scale'), 
                                        #   skm_class.piece('select_perc_c'),
                                            skm_class.feature_union([
                                                            skm_class.piece('agglomeration'), 
                                                            skm_class.piece('k_best_c'),
                                                            ]),
                                            skm_class.piece('k_best_c'),
                                            skm_class.piece(m)])
                
                # set params
                params = skm_class.default_params(pipe, 'rand')
                if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)

                # run the model with parameter search
                best_models, oof_data, param_scores = skm_class.time_series_cv(pipe, X_class, y_class, 
                                                                                params, n_iter=n_iters,
                                                                                col_split='game_date',
                                                                                bayes_rand=opt_type,
                                                                                time_split=cv_time_input,
                                                                                proba=True, random_seed=(i+7)*19+(i*12)+6)

                # append the results and the best models for each fold
                try: param_scores[f'{m}__class_weight'] = param_scores[f'{m}__class_weight'].apply(lambda x: x[0])
                except: pass
                save_param_scores(param_scores, 'class', m)
                out_dict = update_output_dict('class', m, f'_{cut}', out_dict, oof_data, best_models)
                db_output = add_result_db_output(f'class_{cut}', m, oof_data['scores'], db_output)
                gc.collect()

        # save all the outputs
        save_output_dict(out_dict, 'class')
        
    #=====================================================================================

        # set up blank dictionaries for all metrics
        out_dict = {'pred': {}, 'actual': {}, 'scores': {}, 'models': {}, 'full_hold':{}}

        for alpha in [0.8, 0.95]:

            print(f"\n--------------\nAlpha {alpha}\n--------------\n")

            skm_quantile = SciKitModel(df_train, model_obj='quantile')
            X_quant, y_quant = skm_quantile.Xy_split(y_metric='y_act',  to_drop=drop_cols)

            # loop through each potential model
            model_list = ['gbm_q']
            for m in model_list:

                print('\n============\n')
                print(m)

                # set up the model pipe and get the default search parameters
                pipe = skm_quantile.model_pipe([
                                                skm_quantile.piece('random_sample'),
                                                skm_quantile.piece(m)
                                                ])
                
                # set params
                pipe.steps[-1][-1].alpha = alpha
                params = skm_quantile.default_params(pipe, 'rand')
                params['random_sample__frac'] = np.arange(0.02, 0.1, 0.01)

                # run the model with parameter search
                best_models, oof_data, param_scores = skm_quantile.time_series_cv(pipe, X_quant, y_quant, 
                                                                                 params, n_iter=n_iters,
                                                                                 bayes_rand=opt_type,
                                                                                 col_split='game_date',
                                                                                 time_split=cv_time_input)

                # append the results and the best models for each fold
                save_param_scores(param_scores, 'class', m)
                out_dict = update_output_dict('quant', m, f'_{alpha}', out_dict, oof_data, best_models)
                gc.collect()

        save_output_dict(out_dict, 'quant')

        #=============================================================================================
#%%
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


        def show_scatter_plot(y_pred, y, label='Total', r2=True):
            plt.scatter(y_pred, y)
            plt.xlabel('predictions');plt.ylabel('actual')
            plt.show()

            from sklearn.metrics import r2_score
            if r2: print(f'{label} R2:', r2_score(y, y_pred))
            else: print(f'{label} Corr:', np.corrcoef(y, y_pred)[0][1])

        
        def top_predictions(y_pred, y, r2=False):

            val_high_score = pd.concat([pd.Series(y_pred), pd.Series(y)], axis=1)
            val_high_score.columns = ['predictions','y_act']
            val_high_score = val_high_score[val_high_score.predictions >= \
                                            np.percentile(val_high_score.predictions, 75)]
            show_scatter_plot(val_high_score.predictions, val_high_score.y_act, label='Top', r2=r2)

        #------------
        # Make the Class Predictions
        #------------

        # load the class predictions
        pred_class, actual_class, _, _, full_hold_class = load_all_pickles(model_output_path, 'class')
        X_stack_class, y_stack_class = X_y_stack('class', full_hold_class, pred_class, actual_class)

        # load the class predictions
        pred_quant, actual_quant, _, _, full_hold_quant = load_all_pickles(model_output_path, 'quant')
        X_stack_quant, y_stack_quant = X_y_stack('quant', full_hold_quant, pred_quant, actual_quant)
        
        # get the X and y values for stack trainin for the current metric
        pred, actual, _, _, full_hold_reg = load_all_pickles(model_output_path, 'reg')
        X_stack, y_stack = X_y_stack('reg', full_hold_reg, pred, actual)
        X_stack = pd.concat([X_stack, X_stack_class, X_stack_quant], axis=1)

        best_models = []
        final_models = [
                        'ridge',
                        'lasso',
                        'lgbm', 
                        'xgb', 
                        'rf', 
                        'bridge',
                        'gbm'
                        ]
        skm_stack = SciKitModel(df_train, model_obj='reg')
        preds = pd.DataFrame()
        scores = []
        for final_m in final_models:

            print(f'\n{final_m}')

            # get the model pipe for stacking setup and train it on meta features
            stack_pipe = skm_stack.model_pipe([
                                    skm_stack.piece('k_best'), 
                                    skm_stack.piece(final_m)
                                ])
            
            stack_params = skm_stack.default_params(stack_pipe)
            stack_params['k_best__k'] = range(1, X_stack.shape[1])

            best_model, stack_scores, stack_pred = skm_stack.best_stack(stack_pipe, stack_params,
                                                                        X_stack, y_stack, n_iter=50, 
                                                                        run_adp=True, print_coef=True,
                                                                        sample_weight=use_sample_weight)
            best_models.append(best_model)
            scores.append(stack_scores['stack_score'])
            preds = pd.concat([preds, pd.Series(stack_pred['stack_pred'], name=final_m)], axis=1)
            
            # show_scatter_plot(stack_pred['stack_pred'], stack_pred['y'], r2=False)
            # top_predictions(stack_pred['stack_pred'], stack_pred['y'], r2=False)

            db_output = add_result_db_output('final', final_m, 
                                            [stack_scores['adp_score'], stack_scores['stack_score']], 
                                            db_output)

        # print('\nShowing Ensemble\n===============\n')
        # top_3 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        # model_idx = np.array(final_models)[top_3]
        # show_scatter_plot(preds[model_idx].mean(axis=1), stack_pred['y'], r2=False)
        # top_predictions(preds[model_idx].mean(axis=1), stack_pred['y'], r2=False)
        # gc.collect()


        # write out tracking results to tracking DB
        db_output_pd = pd.DataFrame(db_output)
        db_output_pd = db_output_pd.assign(perc_low=prc[set_pos][0]).assign(perc_high=prc[set_pos][1]).assign(perc_range=prc[set_pos][2])
        db_output_pd = db_output_pd.assign(kb_low=kbs[set_pos][0]).assign(kb_high=kbs[set_pos][1]).assign(kb_range=kbs[set_pos][2])
        db_output_pd = db_output_pd.assign(n_iters=n_iters).assign(to_keep=to_keep).assign(val_week=val_week_min).assign(val_year=val_year_min)
        dm.delete_from_db('Results', 'Model_Tracking',f"pkey='{pkey}'")
        dm.write_to_db(db_output_pd, 'Results', 'Model_Tracking', 'append')


#%%

# def load_all_pickles(model_output_path, label):
#     pred = load_pickle(model_output_path, f'{label}_pred')
#     actual = load_pickle(model_output_path, f'{label}_actual')
#     models = load_pickle(model_output_path, f'{label}_models')
#     scores = load_pickle(model_output_path, f'{label}_scores')
#     try: full_hold = load_pickle(model_output_path, f'{label}_full_hold')
#     except: full_hold = None
#     return pred, actual, models, scores, full_hold

# def X_y_stack_full(full_hold):
#     i = 0
#     for k, v in full_hold.items():
#         if i == 0:
#             df = v.copy()
#             df = df.rename(columns={'pred': k})
#         else:
#             df_cur = v.rename(columns={'pred': k}).drop('y_act', axis=1)
#             df = pd.merge(df, df_cur, on=['player', 'team', 'week','year'])
#         i+=1
#     return df

# _, _, _, _, full_hold = load_all_pickles(model_output_path, 'reg')
# df = X_y_stack_full(full_hold)

# df['mean_pred'] = df.iloc[:, 5:].mean(axis=1)
# df['pred_diff'] = df.mean_pred - df.y_act

# df.plot.scatter(x='mean_pred', y='y_act')
# from sklearn.metrics import r2_score, mean_squared_error
# print(r2_score(df.y_act, df.mean_pred), mean_squared_error(df.y_act, df.mean_pred))
# df.sort_values(by='pred_diff').iloc[-25:]
