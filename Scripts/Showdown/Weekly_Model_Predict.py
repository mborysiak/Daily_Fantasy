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
set_year = 2021
set_week = 11

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 10

met = 'y_act'

# full_model or backfill
vers = 'roll8_fullhist_kbestallstack_WRTEDEFkeep25_QBRBdrophalf'

# %%

preds = dm.read(f'''SELECT * 
                    FROM Model_Predictions 
                    WHERE version='{vers}'
                          AND week = '{set_week}'
                          AND year = '{set_year}' 
                          AND player != 'Ryan Griffin'
            ''', 'Simulation')

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

drop_teams = ['NYG', 'TB']

teams = dm.read(f'''SELECT player, team
                    FROM (
                    SELECT CASE WHEN pos!='DST' THEN player ELSE team END player, 
                        team,
                        row_number() OVER (PARTITION BY player ORDER BY projected_points DESC) rn 
                    FROM FantasyPros
                    WHERE week={set_week} AND year={set_year}
                    ) WHERE rn=1''', 'Pre_PlayerData')

preds = pd.merge(preds, teams, on=['player'])
preds = preds[preds.team.isin(drop_teams)].drop('team', axis=1).reset_index(drop=True)

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

sim_dist = create_sim_output(preds).reset_index(drop=True)

#%%

idx = sim_dist[sim_dist.player=="Jakobi Meyers"].index[0]
plot_distribution(sim_dist.iloc[idx])

# %%

dm.write_to_db(sim_dist, 'Simulation', f'showdown_week{set_week}_year{set_year}', 'replace')


# %%

for i in range(12):

    players = dm.read('''SELECT * FROM Best_Lineups WHERE week=7''', 'Results').iloc[i, :9].values

    cur_team = preds[preds.player.isin(players)].copy()

    cur_team['variance'] = cur_team.std_dev ** 2
    sum_variance = np.sum(cur_team.variance)
    sum_mean_var = np.var(cur_team.pred_fp_per_game)

    team_var = np.sqrt(sum_variance + sum_mean_var)
    team_mean = cur_team.pred_fp_per_game.sum()

    import seaborn as sns
    estimates = np.random.normal(team_mean, team_var, 10000)
    
    print(i, team_mean, team_var, np.percentile(estimates, 80), np.percentile(estimates, 99))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws = {'linewidth' : 4},
                 label = 'Estimated Dist.')
    
# %%

players = dm.read('''SELECT * FROM Best_Lineups''', 'Results')
players = players.loc[1:, ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'week', 'year']]
players['lineup_num'] = range(len(players))
best_lineups = pd.melt(players, id_vars=['week', 'year', 'lineup_num'])
best_lineups = best_lineups.rename(columns={'value': 'player'}).drop('variable', axis=1)

y_act = dm.read(f'''SELECT defTeam player, week, season year, fantasy_pts y_act
                    FROM Defense_Stats
                    WHERE season >= 2020''', 'FastR')

for p in ['QB', 'RB', 'WR', 'TE']:
    y_act_cur = dm.read(f'''SELECT player, week, season year, fantasy_pts y_act
                            FROM {p}_Stats
                            WHERE season >= 2020''', 'FastR')
    y_act = pd.concat([y_act, y_act_cur], axis=0)

best_lineups = pd.merge(best_lineups, y_act, on=['player', 'week', 'year'])
team_scores = best_lineups.groupby('lineup_num').agg({'y_act': 'sum'}).y_act
for i in range(50, 100, 5):
    print(f'Percentile {i}:', np.percentile(team_scores, i))

print(f'Percentile 99:', np.percentile(team_scores, 99))
sns.distplot(team_scores, hist = True, kde = True, bins = 15,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws = {'linewidth' : 4},
                 label = 'Estimated Dist.')


# %%


df = dm.read('''SELECT * FROM Model_Predictions''', 'Simulation')


 # pull in the salary and actual results data
dk_sal = dm.read('''SELECT team player, team, week, year, projected_points sd_metric
                    FROM FantasyPros
                    WHERE pos='DST' ''', 'Pre_PlayerData')

stats = dm.read(f'''SELECT defTeam player, defTeam team, week, season year, fantasy_pts y_act
                    FROM {pos}_Stats ''', 'FastR')

# pull in the salary and actual results data
dk_sal = dm.read('''SELECT player, offTeam team, week, year, projected_points, fantasyPoints, ProjPts
                    FROM PFF_Proj_Ranks
                    JOIN (SELECT player, team offTeam, week, year, projected_points 
                            FROM FantasyPros)
                            USING (player, offTeam, week, year)
                    JOIN (SELECT player, offTeam, week, year, `Proj Pts` ProjPts 
                            FROM PFF_Expert_Ranks)
                            USING (player, offTeam, week, year)''', 'Pre_PlayerData')
# %%
