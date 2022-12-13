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

#==========
# General Setting
#==========

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
np.random.seed(1234)

# set year to analyze
set_year = 2022
set_week = 14
showdown_teams = ['MIA', 'LAC']

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 14

met = 'y_act'

# full_model or backfill
vers = 'sera1_rsq0_brier1_matt1_lowsample_perc'

# %%

preds = dm.read(f'''SELECT * 
                    FROM Model_Predictions 
                    WHERE version='{vers}'
                          AND week = '{set_week}'
                          AND year = '{set_year}' 
                          AND player != 'Ryan Griffin'
            ''', 'Simulation')

kickers = preds.loc[preds.pos=='K', 'player'].unique()

preds['weighting'] = 1
preds.loc[preds.model_type=='full_model', 'weighting'] = 1

score_cols = ['pred_fp_per_game', 'std_dev', 'max_score']
for c in score_cols: preds[c] = preds[c] * preds.weighting

# Groupby and aggregate with namedAgg [1]:
preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                              'std_dev': 'sum',
                                                              'max_score': 'sum',
                                                              'min_score': 'sum',
                                                              'weighting': 'sum'})



for c in score_cols: preds[c] = preds[c] / preds.weighting
preds = preds.drop('weighting', axis=1)


teams = dm.read(f'''SELECT player, team
                    FROM (
                    SELECT CASE WHEN pos!='DST' THEN player ELSE team END player, 
                        team,
                        row_number() OVER (PARTITION BY player ORDER BY year, week, projected_points DESC) rn 
                    FROM FantasyPros
                    WHERE week={set_week} AND year={set_year}
                    ) WHERE rn=1''', 'Pre_PlayerData')

preds = pd.merge(preds, teams, on=['player'], how='left')
preds = preds[(preds.team.isin(showdown_teams)) | \
    (preds.player.isin(kickers))].drop('team', axis=1).reset_index(drop=True)

captain = preds.copy()
captain.pos = 'CPT'
for c in ['pred_fp_per_game', 'std_dev', 'max_score', 'min_score']:
    captain[c] = captain[c] * 1.5

preds['pos'] = 'FLEX'

preds = pd.concat([captain, preds], axis=0)

preds.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50]


#%%

def create_distribution(player_data, num_samples=1000):

    import scipy.stats as stats

    # create truncated distribution
    lower, upper = player_data.min_score,  player_data.max_score
    lower_bound = (lower - player_data.pred_fp_per_game) / (player_data.std_dev+1)
    upper_bound = (upper - player_data.pred_fp_per_game) / (player_data.std_dev+1)
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc= player_data.pred_fp_per_game, scale= player_data.std_dev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


def create_sim_output(output, num_samples=1000):
    sim_out = pd.DataFrame()
    for _, row in output.iterrows():
        print(row.player)
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

sim_dist = create_sim_output(preds).reset_index(drop=True)


# %%

dm.write_to_db(sim_dist, 'Simulation', f'showdown_week{set_week}_year{set_year}', 'replace')

# %%
