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

# set year to analyze
set_year = 2020
set_week = 16

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 5

met = 'y_act'

#-----------------
# Run Baseline Model
#-----------------

set_pos = 'TE'

df = dm.read(f'''SELECT playerName player, week, a.year,
                             fantasyPoints,  fantasyPointsRank,
                             `Proj Pts` ProjPts,
                             expertConsensus, expertNathanJahnke,
                             expertKevinCole, expertAndrewErickson,
                             expertIanHartitz,
                             dk_salary, fd_salary, yahoo_salary
                    FROM PFF_Proj_Ranks a
                    JOIN (SELECT Name playerName, *
                            FROM PFF_Expert_Ranks 
                            WHERE Position='{set_pos}' )
                            USING (playerName, week, year)
                    LEFT JOIN (SELECT player playerName, week, year, 
                                        dk_salary, fd_salary, yahoo_salary
                                FROM Daily_Salaries
                                WHERE Position='{set_pos}'
                            ) USING (playerName, week, year)
                    WHERE a.position='{set_pos.lower()}' 
                    ''', 'Pre_PlayerData')

fp = dm.read('''SELECT player, week, year, fp_rank, projected_points
                FROM FantasyPros''', 'Pre_PlayerData')
y_acts = dm.read(f'''SELECT player, week, season year, y_act FROM {set_pos}_Stats''', 'FastR')

fp.player = fp.player.apply(dc.name_clean)
y_acts.player = y_acts.player.apply(dc.name_clean)
df.player = df.player.apply(dc.name_clean)

df = pd.merge(df, fp, on=['player', 'week', 'year'])
df = pd.merge(df, y_acts, on=['player', 'week', 'year'])

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

# get the train / predict dataframes and output dataframe
df_train = df[(df.week < set_week) & (df.year <= set_year)].reset_index(drop=True)
df_predict = df[(df.week == set_week) & (df.year == set_year)].reset_index(drop=True)

to_fill = dm.read(f'''SELECT DISTINCT player FROM week{set_week}_year{set_year}''', 'Simulation')
df_predict = df_predict[~df_predict.player.isin(list(to_fill.player))].reset_index(drop=True)
output_start = df_predict[['player', 'dk_salary']].copy()

# get the minimum number of training samples for the initial datasets
min_samples = int(df_train[df_train.game_date < cv_time_input].shape[0])  
print('Shape of Train Set', df_train.shape)

#%%

df = df.sort_values(by=['year', 'week']).reset_index(drop=True)

skm = SciKitModel(df)
X_base, y_base = skm.Xy_split(y_metric='y_act', to_drop=drop_cols)
cv_time_base = skm.cv_time_splits('game_date', X_base, cv_time_input)

model_base = skm.model_pipe([skm.piece('std_scale'), 
                             skm.piece('k_best'),
                             skm.piece('bridge')])

params = skm.default_params(model_base)
params['k_best__k'] = range(1, X_base.shape[1])

best_model_base = skm.random_search(model_base, X_base, y_base, params, cv=cv_time_base, n_iter=25)
_, _ = skm.val_scores(best_model_base, X_base, y_base, cv_time_base)

imp_cols = X_base.columns[best_model_base['k_best'].get_support()]
skm.print_coef(best_model_base, imp_cols)
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

X_predict = df_predict[X_base.columns]
pred, pred_std = best_model_base.predict(X_predict, return_std=True)
output = output_start.copy()
output['pred_fp_per_game'] = pred
output['std_dev'] = pred_std

output = create_sim_output(output)

# %%
dm.write_to_db(output, 'Simulation', f'week{set_week}_year{set_year}', 'append')
