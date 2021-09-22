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
set_week = 2

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 9

met = 'y_act'

#-----------------
# Run Baseline Model
#-----------------

set_pos = 'RB'
vers = 'v1'


df = dm.read(f'''SELECT * FROM Backfill WHERE pos='{set_pos}' ''', 'Model_Features')

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

to_fill = dm.read(f'''SELECT DISTINCT player FROM Model_Predictions 
                                WHERE pos='{set_pos}'
                                    AND version='{vers}'
                                    AND week={set_week}
                                    AND year={set_year}
                                    AND model_type != 'backfill' ''', 'Simulation')

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
models = ['lasso',
       'ridge',
        'lgbm', 
        'xgb', 
         'bridge'
          ]
for m in models:
    
    print('\n============\n')
    print(m)
    
    pipe = skm.model_pipe([skm.piece('std_scale'), 
                            skm.piece('k_best'),
                            skm.piece(m)])

    params = skm.default_params(pipe)
    params['k_best__k'] = range(1, X.shape[1])

    # run the model with parameter search
    cv_time = skm.cv_time_splits('game_date', X, cv_time_input)
    best_model = skm.random_search(pipe, X, y, params, cv=cv_time, n_iter=25)
    preds = skm.cv_predict_time(best_model, X, y, cv_time)
    
    from sklearn.metrics import r2_score
    print(round(r2_score(y[cv_time[0][1][0]:], preds), 3))

    pipe.fit(X, y)
    predictions = pd.concat([predictions, pd.Series(np.round(pipe.predict(df_predict[X.columns]),1), name=m)], axis=1)


output = output_start[['player', 'dk_salary']].copy()
output = pd.concat([output, predictions], axis=1)

output['pred_fp_per_game'] = predictions.mean(axis=1)
output['std_dev'] = dm.read(f'''SELECT avg(std_dev) 
                                FROM Model_Predictions 
                                WHERE pos='{set_pos}'
                                    AND version='{vers}'
                                    AND week={set_week}
                                    AND year={set_year}
                                    AND model_type != 'backfill' ''', 'Simulation').values[0][0]

output = output.sort_values(by='dk_salary', ascending=False)
output['dk_rank'] = range(len(output))
output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)

# mean_act = df_train.loc[df_train.fp_rank.isin(df_predict.fp_rank.sort_values().unique()[:12]), 'y_act'].mean() 
# ratio = mean_act / output.pred_fp_per_game[:12].mean()
# output['pred_fp_per_game'] = output['pred_fp_per_game'] * ratio

output.iloc[:50]

#%%

output = output[~output.player.isin(list(to_fill.player))].reset_index(drop=True)
output.iloc[:50]


#%%

output = output.drop(models, axis=1)

output['pos'] = set_pos
output['version'] = vers
output['model_type'] = 'backfill'
output['max_score'] = 1.05*np.percentile(df_train.y_act.max(), 99)
output['week'] = set_week
output['year'] = set_year

del_str = f'''pos='{set_pos}'
              AND version='{vers}' 
              AND model_type='backfill'
              AND week={set_week} 
              AND year={set_year}'''

dm.delete_from_db('Simulation', 'Model_Predictions', del_str)
dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')

# %%



preds = dm.read(f'''SELECT * 
                    FROM Model_Predictions 
                    WHERE version='{vers}'
                          AND week = '{set_week}'
                          AND year = '{set_year}' ''', 'Simulation')

preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'mean', 
                                                              'std_dev': 'mean',
                                                              'max_score': 'mean'})

drop_teams = ['WAS', 'NYG', 'BAL', 'KC', 'DET', 'GB']

teams = dm.read(f'''SELECT CASE WHEN pos!='DST' THEN player ELSE team END player, team 
                    FROM FantasyPros
                    WHERE week={set_week} AND year={set_year}''', 'Pre_PlayerData')

preds = pd.merge(preds, teams, on=['player'])
preds = preds[~preds.team.isin(drop_teams)].drop('team', axis=1).reset_index(drop=True)

preds.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50]
# %%

def create_distribution(player_data, num_samples=1000):

    import scipy.stats as stats

    # create truncated distribution
    lower, upper = 0,  player_data.max_score
    lower_bound = (lower - player_data.pred_fp_per_game) / player_data.std_dev, 
    upper_bound = (upper - player_data.pred_fp_per_game) / player_data.std_dev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc= player_data.pred_fp_per_game, scale= player_data.std_dev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


def create_sim_output(output, num_samples=1000):
    sim_out = pd.DataFrame()
    for _, row in output.iterrows():
        cur_out = pd.DataFrame([row.player, row.pos]).T
        cur_out.columns=['player', 'pos']
        dists = pd.DataFrame(create_distribution(row, num_samples)).T
        cur_out = pd.concat([cur_out, dists], axis=1)
        sim_out = pd.concat([sim_out, cur_out], axis=0)
    
    return sim_out


def plot_distribution(estimates):

    from IPython.core.pylabtools import figsize
    import seaborn as sns
    import matplotlib.pyplot as plt

    print('\n', estimates.player)
    estimates = estimates.iloc[2:]

    # Plot all the estimates
    plt.figure(figsize(8, 8))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws = {'linewidth' : 4},
                 label = 'Estimated Dist.')

    # Plot the mean estimate
    plt.vlines(x = estimates.mean(), ymin = 0, ymax = 0.01, 
                linestyles = '--', colors = 'red',
                label = 'Pred Estimate',
                linewidth = 2.5)

    plt.legend(loc = 1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel('Grade'); plt.ylabel('Density');

    # Prediction information
    sum_stats = (np.percentile(estimates, 5), np.percentile(estimates, 95), estimates.std() /estimates.mean())
    print('Average Estimate = %0.4f' % estimates.mean())
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f    Std Error = %0.4f' % sum_stats) 

output = create_sim_output(preds).reset_index(drop=True)

#%%

idx = output[output.player=="Dalvin Cook"].index[0]
plot_distribution(output.iloc[idx])
# %%

dm.write_to_db(output, 'Simulation', f'week{set_week}_year{set_year}', 'replace')
# %%
