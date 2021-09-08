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
val_week_min = 9

met = 'y_act'

#-----------------
# Run Baseline Model
#-----------------

set_pos = 'TE'

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

experts = dm.read(f'''SELECT player, week, year,
                             fantasyPoints,  fantasyPointsRank,
                             `Proj Pts` ProjPts,
                             expertConsensus
                    FROM PFF_Proj_Ranks a
                    JOIN (SELECT *
                            FROM PFF_Expert_Ranks )
                            USING (player, week, year)
                    WHERE a.position='{set_pos.lower()}' 
                          AND expertConsensus IS NOT NULL
                    ''', 'Pre_PlayerData')

df = pd.merge(df, experts, on=['player','week','year'])

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
# df_predict = df_predict[~df_predict.player.isin(list(to_fill.player))].reset_index(drop=True)
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
        # 'rf',
         # 'lgbm', 
        #  'xgb', 
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
    best_model = skm.random_search(pipe, X, y, params, cv=cv_time)
    preds = skm.cv_predict_time(best_model, X, y, cv_time)
    
    from sklearn.metrics import r2_score
    print(round(r2_score(y[cv_time[0][1][0]:], preds), 3))

    pipe.fit(X, y)
    predictions = pd.concat([predictions, pd.Series(np.round(pipe.predict(df_predict[X.columns]),1), name=m)], axis=1)

(_, std) = pipe.predict(df_predict[X.columns], return_std=True)

output = output_start[['player', 'dk_salary']].copy()
output = pd.concat([output, predictions], axis=1)
output['pred_fp_per_game'] = predictions.mean(axis=1)
output['std_dev'] = std
output = output.sort_values(by='dk_salary', ascending=False)
output['dk_rank'] = range(len(output))
output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
output.iloc[:50]

#%%

vers = 'v1'


full = dm.read(f'''SELECT player, pred_fp_per_game full_pred, std_dev full_std
                      FROM Model_Predictions
                      WHERE week={set_week}
                            AND year={set_year}
                            AND model_type='full_model'
                            AND version='{vers}'
                            AND pos = '{set_pos}'
                            AND dk_rank < 5
                        ''', 'Simulation')

full = pd.merge(full, output, on=['player'])
full['ratio'] = (full.full_pred / full.pred_fp_per_game) 
full['std_ratio'] = (full.full_std / full.std_dev) 

ratio = full.ratio.mean()
std_ratio = full.std_ratio.mean()
output['pred_fp_per_game'] = output['pred_fp_per_game'] * ratio
output['std_dev'] = output['std_dev'] * std_ratio
output = output.drop(models, axis=1)
output.iloc[:50]

#%%
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
preds.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50]
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
        cur_out = pd.DataFrame([row.player, row.pos]).T
        cur_out.columns=['player', 'pos']
        dists = pd.DataFrame(create_distribution(row, num_samples)).T
        cur_out = pd.concat([cur_out, dists], axis=1)
        sim_out = pd.concat([sim_out, cur_out], axis=0)
    
    return sim_out

output = create_sim_output(preds)
# %%

dm.write_to_db(output, 'Simulation', f'week{set_week}_year{set_year}', 'replace')
# %%
