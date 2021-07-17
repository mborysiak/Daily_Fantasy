
#%%
# core packages
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt

from ff.db_operations import DataManage
from ff import general
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
root_path = general.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
np.random.seed(1234)

# set to position to analyze: 'RB', 'WR', 'QB', or 'TE'
set_pos = 'WR'

# set year to analyze
set_year = 2020
set_week = 14

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 6

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

# set up the date column for sorting
def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))

df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
cv_time_input = int(dt.datetime(val_year_min, 1, val_week_min).strftime('%Y%m%d'))

#%%

# set up blank dictionaries for all metrics
pred = {}; actual = {}; scores = {}; models = {}

met = 'y_act'

print(f'\nRunning Metric {met}\n=========================\n')
print('ADP only\n============\n')


df_train = df[df.week < set_week].reset_index(drop=True)
df_predict = df[df.week==set_week].reset_index(drop=True)

# get the minimum number of training samples for the initial datasets
min_samples = int(df_train[df_train.game_date < cv_time_input].shape[0])  

print('Shape of Train Set', df_train.shape)
skm = SciKitModel(df_train)
X, y = skm.Xy_split(y_metric='y_act', to_drop=['player', 'position', 'coach', 'team', 'defTeam'])

# set up the ADP model pipe
pipe = skm.model_pipe([skm.piece('feature_select'), skm.piece('std_scale'), skm.piece('lr')])
params = skm.default_params(pipe)
params['feature_select__cols'] = [
                                    ['fantasyPoints',  'fantasyPointsRank', 'ProjPts',
                                     'expertConsensus', 'expertNathanJahnke',
                                     'expertKevinCole', 'expertAndrewErickson','expertIanHartitz',
                                     'dk_salary', 'fd_salary', 'yahoo_salary'], 

                                    ['dk_salary', 'fantasyPoints',  'fantasyPointsRank', 'ProjPts']
                              ]

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
      params['select_perc__percentile'] = range(5, 30, 3)
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
# %%

# 
df_train_class = df[df.week < set_week].reset_index(drop=True)
df_predict_class = df[df.week == set_week].reset_index(drop=True)

# print the value-counts
print('Training Value Counts:', df_train_class.y_act.value_counts()[0], '|', df_train_class.y_act.value_counts()[1])
print(f'Number of Features: {df_train_class.shape[1]}')


#%%

pred = load_pickle(model_output_path, 'reg_pred')
actual = load_pickle(model_output_path, 'reg_actual')
models = load_pickle(model_output_path, 'reg_models')
scores = load_pickle(model_output_path, 'reg_scores')

output = output_start[['player', 'avg_pick']].copy()

# set up the stacking training + prediction dataset
df_train, df_predict = hf.get_train_predict(df, 'fp_per_game', pos, set_pos, 
                                            set_year-pos[set_pos]['test_years'], 
                                            pos[set_pos]['earliest_year'], pos[set_pos]['features'])
df_predict = df_predict.drop('y_act', axis=1).fillna(0)
skm = SciKitModel(df_train)

# get the X and y values for stack trainin for the current metric
X_stack = pd.DataFrame()
for met, pt in zip(pos[set_pos]['metrics'], pts_dict[set_pos]):
    
    if pos[set_pos]['all_stats']:
        X_s, y_s = skm.X_y_stack(met, pred, actual)
        X_stack = pd.concat([X_stack, X_s*pt], axis=1)
        if met=='fp_per_game': y_stack = y_s
    
    elif met == 'fp_per_game':
        X_s, y_s = skm.X_y_stack(met, pred, actual)
        X_stack = pd.concat([X_stack, X_s*pt], axis=1)
        y_stack = y_s

# X_stack = pd.concat([X_stack, X_stack_class], axis=1)

best_models = []
final_models = ['ridge', 'lgbm', 'xgb', 'rf', 'bridge']
for final_m in final_models:

    print(f'\n{final_m}')
    # get the model pipe for stacking setup and train it on meta features
    stack_pipe = skm.model_pipe([
                            skm.piece('std_scale'), 
                            skm.piece('k_best'), 
                            skm.piece(final_m)
                        ])
    best_model, stack_score, adp_score = skm.best_stack(stack_pipe, X_stack, 
                                                        y_stack, n_iter=50, 
                                                        run_adp=True, print_coef=True)
    best_models.append(best_model)

# get the final output:
X_fp, y_fp = skm.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])

# create the full stack pipe with meta estimators followed by stacked model
X_predict = pd.DataFrame()
for k, v in models.items():
    m = skm.ensemble_pipe(v)
    m.fit(X_fp, y_fp)
    X_predict = pd.concat([X_predict, pd.Series(m.predict(df_predict[X_fp.columns]), name=k)], axis=1)

X_predict = pd.concat([X_predict, X_predict_class], axis=1)

predictions = pd.DataFrame()
for bm, fm in zip(best_models, final_m):
    prediction = pd.Series(np.round(bm.predict(X_predict), 2), name=f'pred_{met}_{fm}')
    predictions = pd.concat([predictions, prediction], axis=1)

db_output['reg_stack_score'] = stack_score
db_output['adp_stack_score'] = adp_score

output['pred_fp_per_game'] = predictions.mean(axis=1)
std_models = predictions.std(axis=1)
std_bridge = bm.predict(X_predict, return_std=True)[1]
output['std_dev'] = std_models + std_bridge
output = output.sort_values(by='avg_pick')
output['adp_rank'] = range(len(output))
output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
output.iloc[:50]