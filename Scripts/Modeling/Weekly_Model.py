
#%%
# core packages
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
set_pos = 'Defense'

# set year to analyze
set_year = 2021
set_week = 1

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 9

met = 'y_act'

#-------------
# Set up output dataset
#-------------

all_vars = [set_pos, set_year, set_week]

pkey = f'{set_pos}_year{set_year}_week{set_week}'
db_output = {'set_pos': set_pos, 'set_year': set_year, 'set_week': set_week}
db_output['pkey'] = pkey

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

#==========
# Pull and clean compiled data
#==========

# load data and filter down
df = dm.read(f'''SELECT * FROM {set_pos}_Data''', 'Model_Features')
drop_cols = list(df.dtypes[df.dtypes=='object'].index)
print(drop_cols)

# set up the date column for sorting
def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))

df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
cv_time_input = int(dt.datetime(val_year_min, 1, val_week_min).strftime('%Y%m%d'))
train_time_split = int(dt.datetime(set_year, 1, set_week).strftime('%Y%m%d'))

# get the train / predict dataframes and output dataframe
df_train = df[df.game_date < train_time_split].reset_index(drop=True)
df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)
df_predict = df[df.game_date == train_time_split].reset_index(drop=True)
output_start = df_predict[['player', 'dk_salary']].copy()

# get the minimum number of training samples for the initial datasets
min_samples = int(df_train[df_train.game_date < cv_time_input].shape[0])  
print('Shape of Train Set', df_train.shape)

# set up the target variable to be categorical based on Xth percentile
df_class = df.copy()
if set_pos == 'Defense':
    cut_perc = np.percentile(df_class.y_act, 80)
else:
    cut_perc = np.percentile(df_class.y_act, 95)
df_class['y_act'] = np.where(df_class.y_act >= cut_perc, 1, 0)

# set up the training and prediction datasets for the classification 
df_train_class = df_class[df_class.game_date < train_time_split].reset_index(drop=True)
df_predict_class = df_class[df_class.game_date == train_time_split].reset_index(drop=True)

#%%

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
params['feature_select__cols'] = [[ 'dk_salary'], ['dk_salary', 'year'], ['dk_salary', 'year', 'week'] ]

# fit and append the ADP model
best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=50,
                                                col_split='game_date', 
                                                time_split=cv_time_input)

# append all of the metric outputs
pred[f'{met}_adp'] = oof_data['combined']; actual[f'{met}_adp'] = oof_data['actual']
scores[f'{met}_adp'] = r2; models[f'{met}_adp'] = best_models

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
                                          skm.piece('pca')
                                          ]),
                              skm.piece('k_best'),
                              skm.piece(m)])
      params = skm.default_params(pipe, 'rand')
      params['select_perc__percentile'] = range(5, 30, 5)
      params['feature_drop__col'] = [

            ['fantasyPoints',  'fantasyPointsRank', 'ProjPts',
             'expertConsensus', 'expertNathanJahnke',
             'expertKevinCole', 'expertAndrewErickson','expertIanHartitz',
             'dk_salary', 'fd_salary', 'yahoo_salary'],

             ['fantasyPoints',  'fantasyPointsRank', 'ProjPts',
              'expertConsensus', 'expertNathanJahnke',
              'expertKevinCole', 'expertAndrewErickson','expertIanHartitz'],

             ['dk_salary', 'fd_salary', 'yahoo_salary'],
            

      ]
      if m=='knn': params['knn__n_neighbors'] = range(1, min_samples-1)

      # run the model with parameter search
      best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, params, n_iter=25,
                                                      col_split='game_date', time_split=cv_time_input)

      # append the results and the best models for each fold
      pred[f'{met}_{m}'] = oof_data['combined']; actual[f'{met}_{m}'] = oof_data['actual']
      scores[f'{met}_{m}'] = r2; models[f'{met}_{m}'] = best_models

save_pickle(pred, model_output_path, 'reg_pred')
save_pickle(actual, model_output_path, 'reg_actual')
save_pickle(models, model_output_path, 'reg_models')
save_pickle(scores, model_output_path, 'reg_scores')

# print the value-counts
print('Training Value Counts:', df_train_class.y_act.value_counts()[0], '|', df_train_class.y_act.value_counts()[1])

# set up blank dictionaries for all metrics
pred = {}; actual = {}; scores = {}; models = {}

skm_class = SciKitModel(df_train_class, model_obj='class')
X_class, y_class = skm_class.Xy_split(y_metric='y_act', 
                                      to_drop=drop_cols)

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
    params['select_perc_c__percentile'] = range(5, 30, 5)
    if m=='knn_c': params['knn_c__n_neighbors'] = range(1, min_samples-1)

    # run the model with parameter search
    best_models, score_results, oof_data = skm_class.time_series_cv(pipe, X_class, y_class, 
                                                                    params, n_iter=25,
                                                                    col_split='game_date',
                                                                    time_split=cv_time_input)

    # append the results and the best models for each fold
    pred[f'class_{m}'] = oof_data['combined']; actual[f'class_{m}'] = oof_data['actual']
    scores[f'class_{m}'] = score_results; models[f'class_{m}'] = best_models

save_pickle(pred, model_output_path, 'class_pred')
save_pickle(actual, model_output_path, 'class_actual')
save_pickle(models, model_output_path, 'class_models')
save_pickle(scores, model_output_path, 'class_scores')

#%%
#------------
# Make the Class Predictions
#------------

pred_class = load_pickle(model_output_path, 'class_pred')
actual_class = load_pickle(model_output_path, 'class_actual')
models_class = load_pickle(model_output_path, 'class_models')
scores_class = load_pickle(model_output_path, 'class_scores')

skm_class_final = SciKitModel(df_train_class, model_obj='class')
X_stack_class, y_stack_class = skm_class_final.X_y_stack('class', pred_class, actual_class)
X_class_final, y_class_final = skm_class_final.Xy_split(y_metric='y_act', to_drop=drop_cols)

# create the full stack pipe with meta estimators followed by stacked model
X_predict_class = pd.DataFrame()
for k, v in models_class.items():
    if 'svc' not in k:
        m = skm_class_final.ensemble_pipe(v)
        m.fit(X_class_final, y_class_final)
        cur_pred = pd.Series(m.predict_proba(df_predict_class[X_class_final.columns])[:,1], name=k)
        X_predict_class = pd.concat([X_predict_class, cur_pred], axis=1)

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
final_models = ['gbm', 
                'lgbm', 
                'xgb', 
                'rf', 
                'bridge'
                ]
for final_m in final_models:

    print(f'\n{final_m}')
    # get the model pipe for stacking setup and train it on meta features
    stack_pipe = skm_stack.model_pipe([
                            skm_stack.piece('std_scale'), 
                            skm_stack.piece('k_best'), 
                            skm_stack.piece(final_m)
                        ])
    best_model, stack_score, adp_score = skm_stack.best_stack(stack_pipe, X_stack, 
                                                        y_stack, n_iter=50, 
                                                        run_adp=True, print_coef=True)
    best_models.append(best_model)


# get the final output:
X_fp, y_fp = skm_stack.Xy_split(y_metric='y_act', to_drop=drop_cols)

# create the full stack pipe with meta estimators followed by stacked model
X_predict = pd.DataFrame()
for k, v in models.items():
    m = skm_stack.ensemble_pipe(v)
    m.fit(X_fp, y_fp)
    X_predict = pd.concat([X_predict, pd.Series(m.predict(df_predict[X_fp.columns]), name=k)], axis=1)

X_predict = pd.concat([X_predict, X_predict_class], axis=1)

predictions = pd.DataFrame()
for bm, fm in zip(best_models, final_m):
    prediction = pd.Series(np.round(bm.predict(X_predict), 2), name=f'pred_{met}_{fm}')
    predictions = pd.concat([predictions, prediction], axis=1)


output['pred_fp_per_game'] = predictions.mean(axis=1)
std_models = predictions.std(axis=1)
std_bridge = bm.predict(X_predict, return_std=True)[1]
output['std_dev'] = std_bridge

output = output.sort_values(by='dk_salary', ascending=False)
output['dk_rank'] = range(len(output))
output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
output.iloc[:50]

#%%
vers = 'v1'

output['pos'] = set_pos
output['version'] = vers
output['model_type'] = 'full_model'
output['max_score'] = 1.05*np.percentile(df_train.y_act.max(), 99)
output['week'] = set_week
output['year'] = set_year

del_str = f'''pos='{set_pos}' 
             AND version='{vers}' 
             AND week={set_week} 
             AND year={set_year}
             AND model_type='full_model'
             '''
dm.delete_from_db('Simulation', 'Model_Predictions', del_str)
dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')
# %%
