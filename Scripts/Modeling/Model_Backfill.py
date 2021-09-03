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
from xgboost.sklearn import XGBModel, XGBRegressor
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

# set year to analyze
set_year = 2021
set_week = 1

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 4

met = 'y_act'

#-----------------
# Run Baseline Model
#-----------------

set_pos = 'WR'

# df = dm.read(f'''SELECT playerName player, week, a.year,
#                              fantasyPoints,  fantasyPointsRank,
#                              `Proj Pts` ProjPts,
#                              expertConsensus, expertNathanJahnke,
#                              expertKevinCole, expertAndrewErickson,
#                              expertIanHartitz,
#                              dk_salary, fd_salary, yahoo_salary
#                     FROM PFF_Proj_Ranks a
#                     JOIN (SELECT Name playerName, *
#                             FROM PFF_Expert_Ranks 
#                             WHERE Position='{set_pos}' )
#                             USING (playerName, week, year)
#                     LEFT JOIN (SELECT player playerName, week, year, 
#                                         dk_salary, fd_salary, yahoo_salary
#                                 FROM Daily_Salaries
#                                 WHERE Position='{set_pos}'
#                             ) USING (playerName, week, year)
#                     WHERE a.position='{set_pos.lower()}' 
#                     ''', 'Pre_PlayerData')

df = dm.read(f'''SELECT player, week, year, fp_rank, projected_points,
                        dk_salary, fd_salary, yahoo_salary
                 FROM FantasyPros
                 JOIN (SELECT *
                       FROM Daily_Salaries
                       WHERE Position='{set_pos}'
                            ) USING (player, week, year)
                    WHERE pos='{set_pos}' 
                    ''', 'Pre_PlayerData')
df.player = df.player.apply(dc.name_clean)

# fill in null expert rankings
df = df.sort_values(by=['player', 'year', 'week'])
df = df.groupby(['player'], as_index=False).apply(lambda group: group.ffill())
for c in ['dk_salary', 'fd_salary', 'yahoo_salary']:
    df[c] = df[c].fillna(df[c].min())
df = df.fillna(df.max())

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
df_predict = df[df.game_date == train_time_split].reset_index(drop=True)

y_acts = dm.read(f'''SELECT player, week, season year, y_act 
                     FROM {set_pos}_Stats''', 'FastR').dropna(subset=['y_act'])
y_acts.player = y_acts.player.apply(dc.name_clean)
df_train = pd.merge(df_train, y_acts, on=['player', 'week', 'year'])

to_fill = dm.read(f'''SELECT DISTINCT player FROM week{set_week}_year{set_year}''', 'Simulation')
df_predict = df_predict[~df_predict.player.isin(list(to_fill.player))].reset_index(drop=True)
output_start = df_predict[['player', 'dk_salary']].copy()

# get the minimum number of training samples for the initial datasets
min_samples = int(df_train[df_train.game_date < cv_time_input].shape[0])  
print('Shape of Train Set', df_train.shape)

#%%
# set up blank dictionaries for all metrics
pred = {}; actual = {}; scores = {}; models = {}
met = 'y_act'

skm = SciKitModel(df_train)
X, y = skm.Xy_split(y_metric='y_act', to_drop=drop_cols)

predictions = pd.DataFrame()
for m in ['lgbm', 'ridge', 'svr', 'lasso', 'enet', 'xgb', 'knn', 'gbm', 'rf']:
    
    print('\n============\n')
    print(m)
    
    pipe = skm.model_pipe([skm.piece('std_scale'), 
                            skm.piece('k_best'),
                            skm.piece(m)])

    params = skm.default_params(pipe)
    params['k_best__k'] = range(1, X.shape[1])

    # run the model with parameter search
    best_models, r2, oof_data = skm.time_series_cv(pipe, X, y, 
                                                   params, n_iter=25,
                                                   col_split='game_date',
                                                   time_split=cv_time_input)

    # append the results and the best models for each fold
    pred[f'{met}_{m}'] = oof_data['combined']; actual[f'{met}_{m}'] = oof_data['actual']
    scores[f'{met}_{m}'] = r2; models[f'{met}_{m}'] = best_models

#%%
output = output_start[['player', 'dk_salary']].copy()

df_predict_stack = df_predict.copy()
df_predict_stack = df_predict_stack
skm_stack = SciKitModel(df_train)

# get the X and y values for stack trainin for the current metric
X_stack, y_stack = skm_stack.X_y_stack(met, pred, actual)

best_models = []
final_models = ['lasso', 'lgbm', 'xgb', 'rf', 'bridge']
for final_m in final_models:

    print(f'\n{final_m}')
    # get the model pipe for stacking setup and train it on meta features
    stack_pipe = skm_stack.model_pipe([
                          #  skm_stack.piece('std_scale'), 
                            skm_stack.piece('k_best'), 
                            skm_stack.piece(final_m)
                        ])
    best_model, stack_score, adp_score = skm_stack.best_stack(stack_pipe, X_stack, 
                                                        y_stack, n_iter=50, 
                                                        run_adp=False, print_coef=True)
    best_models.append(best_model)


# get the final output:
X_fp, y_fp = skm_stack.Xy_split(y_metric='y_act', to_drop=drop_cols)

# create the full stack pipe with meta estimators followed by stacked model
X_predict = pd.DataFrame()
for k, v in models.items():
    m = skm_stack.ensemble_pipe(v)
    m.fit(X_fp, y_fp)
    X_predict = pd.concat([X_predict, pd.Series(m.predict(df_predict[X_fp.columns]), name=k)], axis=1)

predictions = pd.DataFrame()
for bm, fm in zip(best_models, final_models):
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

# %%

def create_distribution(player_data, num_samples=1000):
    
    print(player_data.player)
    import scipy.stats as stats

    # create truncated distribution
    lower, upper = np.percentile(df_train.y_act, 0.5),  np.percentile(df_train.y_act, 99.5) * 1.1
    lower_bound = (lower - player_data.pred_fp_per_game) / player_data.std_dev, 
    upper_bound = (upper - player_data.pred_fp_per_game) / player_data.std_dev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc= player_data.pred_fp_per_game, scale= player_data.std_dev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


def create_sim_output(output, num_samples=1000):
    sim_out = pd.DataFrame()
    for _, row in output.iterrows():
        cur_out = pd.DataFrame([row.player, set_pos]).T
        cur_out.columns=['player', 'pos']
        dists = pd.DataFrame(create_distribution(row, num_samples)).T
        cur_out = pd.concat([cur_out, dists], axis=1)
        sim_out = pd.concat([sim_out, cur_out], axis=0)
    
    return sim_out

# %%
output = create_sim_output(output)
dm.write_to_db(output, 'Simulation', f'week{set_week}_year{set_year}', 'append')


# %%
