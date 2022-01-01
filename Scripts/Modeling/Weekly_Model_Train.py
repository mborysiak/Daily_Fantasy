#%%
# core packages
from random import Random
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

model_type = 'backfill'
vers = 'standard'

n_iters = 25
to_keep = 25

drop_words = ['ProjPts', 'recv', 'fantasyPoints', 'expert', 'fp_rank', 'proj', 'projected_points', 'salary']
keep_words = ['def', 'qb', 'team']

if model_type == 'full_model': positions = ['QB', 'RB', 'WR', 'TE', 'Defense']
elif model_type == 'backfill': positions = ['QB', 'RB', 'WR', 'TE']

for set_pos in positions:

    # set the earliest date to begin the validation set
    val_year_min = 2020
    if set_pos in ('WR', 'TE'): val_week_min = 12
    else: val_week_min = 10

    kbs = {
        'QB': [5, 50, 5],
        'RB': [10, 100, 10],
        'WR': [5, 50, 5],
        'TE': [5, 50, 5],
        'Defense': [5, 50, 5]
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


    # # test log
    # df = df[df.y_act > 1].reset_index(drop=True)


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

    # # test log
    # y = np.log(y)

    # set up the ADP model pipe
    pipe = skm.model_pipe([skm.piece('feature_select'), 
                           skm.piece('std_scale'), 
                           skm.piece('k_best'),
                           skm.piece('lr')])
    params = skm.default_params(pipe)
    params['k_best__k'] = range(5)
    params['feature_select__cols'] = [['ProjPts', 'dk_salary', 'fantasyPoints', 'year', 'week']]

    # fit and append the ADP model
    best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                   col_split='game_date', 
                                                    time_split=cv_time_input)

    # append all of the metric outputs
    pred[f'{met}_adp'] = oof_data['hold']
    actual[f'{met}_adp'] = oof_data['actual']
    scores[f'{met}_adp'] = r2
    models[f'{met}_adp'] = best_models

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
        pipe = skm.model_pipe([ skm.piece('feature_drop'),
                                skm.piece('std_scale'), 
                                skm.piece('select_perc'),
                                skm.feature_union([
                                            skm.piece('agglomeration'), 
                                            skm.piece('k_best'),
                                            # skm.piece('pca')
                                            ]),
                                skm.piece('k_best'),
                                skm.piece(m)])
        
        # set params
        params = skm.default_params(pipe, 'rand')
        params['select_perc__percentile'] = range(prc[set_pos][0],  prc[set_pos][1], prc[set_pos][2])
        params['k_best__k'] = range(kbs[set_pos][0],kbs[set_pos][1], kbs[set_pos][2])
        if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)
    
        if model_type=='backfill':
            params['feature_drop__col'] = [[None]]

        if set_pos in ('QB', 'RB'):
            params['feature_drop__col'] = [list(np.random.choice(X.columns, int(0.5*X.shape[1]), replace=False)) for _ in range(10)]
        else:
            to_drop = [c for c in X.columns if any(dw in c for dw in drop_words) and not any(kw in c for kw in keep_words)]
            params['feature_drop__col'] = [list(np.random.choice(to_drop, len(to_drop)-to_keep, replace=False)) for _ in range(10)]

        # run the model with parameter search
        best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=n_iters,
                                                        col_split='game_date', time_split=cv_time_input)

        # append the results and the best models for each fold
        pred[f'{met}_{m}'] = oof_data['hold']
        actual[f'{met}_{m}'] = oof_data['actual']
        scores[f'{met}_{m}'] = r2
        models[f'{met}_{m}'] = best_models

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
            pipe = skm.model_pipe([skm.piece('feature_drop'),
                                   skm.piece('std_scale'), 
                                   skm.piece('select_perc_c'),
                                   skm.feature_union([
                                                skm.piece('agglomeration'), 
                                                skm.piece('k_best_c'),
                                                # skm.piece('pca')
                                                ]),
                                    skm.piece('k_best_c'),
                                    skm.piece(m)])
            
            # set params
            params = skm.default_params(pipe, 'rand')
            params['select_perc_c__percentile'] = range(prc[set_pos][0],  prc[set_pos][1], prc[set_pos][2])
            params['k_best_c__k'] = range(kbs[set_pos][0], kbs[set_pos][1], kbs[set_pos][2])
            if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)


            if model_type=='backfill':
                params['feature_drop__col'] = [[None]]
                
            if set_pos in ('QB', 'RB'):
                params['feature_drop__col'] = [list(np.random.choice(X.columns, int(0.5*X.shape[1]), replace=False)) for _ in range(10)]
            else:
                to_drop = [c for c in X.columns if any(dw in c for dw in drop_words) and not any(kw in c for kw in keep_words)]
                params['feature_drop__col'] = [list(np.random.choice(to_drop, len(to_drop)-to_keep, replace=False)) for _ in range(10)]
                

            # run the model with parameter search
            best_models, score_results, oof_data = skm_class.time_series_cv(pipe, X_class, y_class, 
                                                                            params, n_iter=n_iters,
                                                                            col_split='game_date',
                                                                            time_split=cv_time_input)

            # append the results and the best models for each fold
            pred[f'class_{m}_{cut}'] = oof_data['hold']
            actual[f'class_{m}_{cut}'] = oof_data['actual']
            scores[f'class_{m}_{cut}'] = score_results 
            models[f'class_{m}_{cut}'] = best_models

            db_output = add_result_db_output(f'class_{cut}', m, score_results, db_output)

        save_pickle(pred, model_output_path, 'class_pred')
        save_pickle(actual, model_output_path, 'class_actual')
        save_pickle(models, model_output_path, 'class_models')
        save_pickle(scores, model_output_path, 'class_scores')


    #=============================================================================================

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
        stack_pipe = skm_stack.model_pipe([
                                skm_stack.feature_union([
                                                skm_stack.piece('agglomeration'), 
                                                skm_stack.piece('k_best'),
                                                skm_stack.piece('pca')
                                                ]),
                                skm_stack.piece('k_best'), 
                                skm_stack.piece(final_m)
                            ])
        stack_params = skm_stack.default_params(stack_pipe)
        stack_params['k_best__k'] = range(1, X_stack.shape[1])

        best_model, stack_score, adp_score = skm_stack.best_stack(stack_pipe, stack_params,
                                                                X_stack, y_stack, n_iter=50, 
                                                                run_adp=True, print_coef=False)
        best_models.append(best_model)
        db_output = add_result_db_output('final', final_m, [adp_score, stack_score], db_output)


        # get the final output:
        X_fp, y_fp = skm_stack.Xy_split(y_metric='y_act', to_drop=drop_cols)

        val_predictions = pd.DataFrame()
        for bm, fm in zip(best_models, final_models):
            val_predict = pd.Series(skm_stack.cv_predict(bm, X_fp, y_fp), name=f'pred_{met}_{fm}')
            val_predictions = pd.concat([val_predictions, val_predict], axis=1)

        #------------
        # Scatter and Metrics for Overall Results
        #------------
        plt.scatter(val_predict, y_fp)
        plt.xlabel('predictions');plt.ylabel('actual')
        plt.show()
        from sklearn.metrics import r2_score
        print('Total R2:', r2_score(y_fp, val_predict))

        val_high_score = pd.concat([val_predict, y_fp], axis=1)
        val_high_score.columns = ['predictions','y_act']
        val_high_score = val_high_score[val_high_score.predictions >= \
                                        np.percentile(val_high_score.predictions, 75)]

        val_high_score.plot.scatter(x='predictions', y='y_act')
        plt.show()
        print('High Score R2:', r2_score(val_high_score.y_act, val_high_score.predictions))


    #------------
    # Scatter and Metrics for Overall Results
    #------------
    plt.scatter(val_predictions.mean(axis=1), y_fp)
    plt.xlabel('predictions');plt.ylabel('actual')
    plt.show()
    from sklearn.metrics import r2_score
    print('Total R2:', r2_score(y_fp, val_predictions.mean(axis=1)))

    val_high_score = pd.concat([val_predictions.mean(axis=1), y_fp], axis=1)
    val_high_score.columns = ['predictions','y_act']
    val_high_score = val_high_score[val_high_score.predictions >= \
                                    np.percentile(val_high_score.predictions, 75)]

    val_high_score.plot.scatter(x='predictions', y='y_act')
    plt.show()
    print('High Score R2:', r2_score(val_high_score.y_act, val_high_score.predictions))


    # write out tracking results to tracking DB
    db_output_pd = pd.DataFrame(db_output)
    db_output_pd = db_output_pd.assign(perc_low=prc[set_pos][0]).assign(perc_high=prc[set_pos][1]).assign(perc_range=prc[set_pos][2])
    db_output_pd = db_output_pd.assign(kb_low=kbs[set_pos][0]).assign(kb_high=kbs[set_pos][1]).assign(kb_range=kbs[set_pos][2])
    db_output_pd = db_output_pd.assign(n_iters=n_iters).assign(to_keep=to_keep).assign(val_week=val_week_min).assign(val_year=val_year_min)
    dm.delete_from_db('Results', 'Model_Tracking',f"pkey='{pkey}'")
    dm.write_to_db(db_output_pd, 'Results', 'Model_Tracking', 'append')


#%%

#=====================================================


# df_train_subset = df_train[df_train.ProjPts > np.percentile(df_train.ProjPts, 20)].reset_index(drop=True)
df_train_subset = df_train[df_train.y_act > 0.5].reset_index(drop=True)

skm = SciKitModel(df_train_subset)
X, y = skm.Xy_split(y_metric='y_act', to_drop=drop_cols)

plt.hist(y)
plt.show()

# y = y + abs(y.min()) + 1
plt.hist(np.log1p(y))
plt.show()

# y = np.log1p(y)

#%%
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import PoissonRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

np.random.seed(1234)

# set up the model pipe and get the default search parameters
m = 'ridge'
pipe = skm.model_pipe([
                        skm.piece('feature_drop'),
                        skm.piece('std_scale'), 
                        skm.piece('select_perc'),
                        skm.feature_union([
                                    skm.piece('agglomeration'), 
                                    skm.piece('k_best'),
                                    # skm.piece('pca')
                                    ]),
                        skm.piece('k_best'),
                        # ('ridge', PoissonRegressor())
                        # ('lgbm', LGBMRegressor(n_jobs=1, objective='tweedie'))
                        # ('xgb', XGBRegressor(n_jobs=1, objective='reg:tweedie'))
                        skm.piece(m)
                        ])


# set params
params = skm.default_params(pipe, 'rand')
params['select_perc__percentile'] = range(prc[set_pos][0],  prc[set_pos][1], prc[set_pos][2])
params['k_best__k'] = range(kbs[set_pos][0],kbs[set_pos][1], kbs[set_pos][2])
if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)

if model_type=='backfill':
    params['feature_drop__col'] = [[None]]

if set_pos in ('QB', 'RB'):
    params['feature_drop__col'] = [list(np.random.choice(X.columns, int(0.5*X.shape[1]), replace=False)) for _ in range(10)]
else:
    to_drop = [c for c in X.columns if any(dw in c for dw in drop_words) and not any(kw in c for kw in keep_words)]
    params['feature_drop__col'] = [list(np.random.choice(to_drop, len(to_drop)-to_keep, replace=False)) for _ in range(10)]

transform = False

if transform:
    print('yes_transform')
    pipe = TransformedTargetRegressor(pipe, func=np.log1p, inverse_func=np.expm1)
    # pipe = TransformedTargetRegressor(pipe, transformer=QuantileTransformer(output_distribution="normal"))
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('full_pipe', pipe)])
    params = {'full_pipe__regressor__'+k: v for k, v in params.items()}
else:
    print('no_transform')

# fit and append the ADP model
best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                col_split='game_date', 
                                                time_split=cv_time_input)


from sklearn.metrics import r2_score, mean_squared_error

try: del oof_data['val']
except: pass
# results = np.log1p(pd.DataFrame(oof_data))
# results = np.exp(pd.DataFrame(oof_data))
results = pd.DataFrame(oof_data)

# oof_data['actual'] = np.exp(oof_data['actual'])
# oof_data['hold'] = np.exp(oof_data['hold'])
# oof_data['actual'] = np.log1p(oof_data['actual'])
# oof_data['hold'] = np.log1p(oof_data['hold'])

print(r2_score(oof_data['actual'], oof_data['hold']))
plt.scatter(oof_data['hold'], oof_data['actual'])

high_pred = results[results.hold > np.percentile(results.hold, 85)]
print(r2_score(high_pred.actual, high_pred.hold))
plt.scatter(high_pred.hold,high_pred.actual)
plt.show()

print(np.sqrt(mean_squared_error(high_pred.actual, high_pred.hold)))
results['resid'] = results.actual - results.hold
plt.scatter(results.hold, results.resid)

#%%
i = 4
try: coefs = best_models[i].named_steps[m].coef_
except: coefs = best_models[i].named_steps[m].feature_importances_
try:cols = X.drop(best_models[i].named_steps['feature_drop'].col, axis=1).columns[best_models[i].named_steps['k_best'].get_support()]
except: cols = X.columns[best_models[i].named_steps['k_best'].get_support()]
pd.Series(coefs, index=cols).sort_values().plot.barh(figsize=(5,10))
# %%
