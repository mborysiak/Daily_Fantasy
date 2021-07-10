#%%
import pandas as pd 
import pyarrow.parquet as pq
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set the filepath and name for NFL Fast R data saved from R script
DATA_PATH = f'{root_path}/Data/OtherData/NFL_FastR/'
FNAME = 'raw_data_2000_2020.parquet'

pd.set_option('display.max_columns', 999)

#---------------
# Functions
#---------------

def one_hot_col(df, col):
    return pd.concat([df, pd.get_dummies(df[col])], axis=1).drop(col, axis=1)


def get_agg_stats(data, gcols, stat_cols, agg_type, prefix=''):

    agg_stat = {c: agg_type for c in stat_cols}
    df = data.groupby(gcols).agg(agg_stat)
    df.columns = [f'{prefix}_{c}_{agg_type}' for c in df.columns]

    return df.reset_index()


def get_coaches():
    hc = data[['season', 'week', 'home_team', 'home_coach']].drop_duplicates()
    ac = data[['season', 'week', 'away_team', 'away_coach']].drop_duplicates()
    
    new_cols = ['season', 'week', 'posteam', 'coach']
    hc.columns = new_cols
    ac.columns = new_cols

    return pd.concat([hc, ac], axis=0).reset_index(drop=True)


def window_max(df, w_col, gcols, agg_met, agg_type):
    
    gcols.append(w_col)
    agg_df = df.groupby(gcols, as_index=False).agg({agg_met: agg_type})
    
    gcols.remove(w_col)
    
    max_df = agg_df.groupby(gcols, as_index=False).agg({agg_met: 'max'})

    gcols.append(agg_met)
    return pd.merge(agg_df, max_df, on=gcols) 


def calc_fp(df, cols, pts):
    df['fantasy_pts'] = (df[cols] * pts).sum(axis=1)
    return df


#%%

#---------------
# Load Data
#---------------

# read in the data, filter to real players, and sort by value
data = pq.read_table(f'{DATA_PATH}/{FNAME}').to_pandas()
data = data.sort_values(by='epa', ascending=False).reset_index(drop=True)

#---------------
# CLean Data
#---------------

cols = ['season', 'week', 'season_type', 'game_id', 'spread_line', 'total_line', 'location', 'roof', 'surface',
        'vegas_wp', 'temp', 'defteam', 'posteam', 'home_team', 'away_team', 'home_coach', 'away_coach',
        'desc', 'play', 'down', 'drive', 'away_score', 'home_score', 'half_seconds_remaining',
        'passer_player_id', 'rusher_player_id', 'receiver_player_id',
        'receiver_player_name', 'rusher_player_name', 'passer_player_name',
        'receiver_player_position',  'rusher_player_position', 'passer_player_position',
        'play_type', 'play_type_nfl', 'shotgun', 'no_huddle', 
        'pass_attempt', 'rush_attempt', 'rush', 'qb_dropback', 'qb_scramble', 'penalty',
        'first_down', 'first_down_pass', 'first_down_rush', 'fourth_down_converted',
        'fourth_down_failed', 'third_down_converted', 'third_down_failed', 'goal_to_go', 'td_prob', 'drive',
        'drive_ended_with_score', 'drive_first_downs', 'drive_play_count', 'drive_inside20',
        'drive_game_clock_start', 'drive_game_clock_end', 'drive_yards_penalized',
        'drive_start_yard_line', 'drive_end_yard_line', 'drive_time_of_possession',
        'run_gap', 'run_location','rush_touchdown', 'tackled_for_loss',
        'pass_touchdown', 'pass_length', 'pass_location', 'qb_epa', 'qb_hit','sack',
        'cp', 'cpoe', 'air_epa', 'air_wpa', 'air_yards', 'comp_air_epa',
        'comp_air_wpa', 'comp_yac_epa', 'comp_yac_wpa','complete_pass', 'incomplete_pass',
        'interception', 'ep', 'epa', 'touchdown',
        'home_wp', 'home_wp_post', 'away_wp', 'away_wp_post',
        'fumble', 'fumble_lost',
        'wp', 'wpa', 'xyac_epa', 'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage',
        'xyac_success', 'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
        'yards_gained', 'ydsnet', 'ydstogo',
        'receiver_player_age', 'receiver_player_college_name', 
        'receiver_player_height', 'receiver_player_weight',
        'rusher_player_age', 'rusher_player_college_name',
        'rusher_player_height', 'rusher_player_weight',
        'passer_player_age', 'passer_player_college_name', 
        'passer_player_height', 'passer_player_weight']

data = data.loc[data.season_type=='REG', cols]

data.loc[data.run_location.isin(['left', 'right']), 'run_location'] = 'run_outside'
data.loc[data.run_location=='middle', 'run_location'] = 'run_middle'

data.loc[data.pass_location.isin(['left', 'right']), 'pass_location'] = 'pass_outside'
data.loc[data.pass_location=='middle', 'pass_location'] = 'pass_middle'

data.loc[(data.surface != 'grass') & ~(data.surface.isnull()), 'surface'] = 'synthetic'
data.loc[data.roof.isin(['outdoors', 'open']), 'roof'] = 'outdoors'
data.loc[data.roof.isin(['dome', 'closed']), 'roof'] = 'indoors'

for c in ['run_location', 'pass_location', 'surface', 'roof']:
    data = one_hot_col(data, c)

def time_convert(t):
    return int(t.split(':')[0]) + float(t.split(':')[1]) / 60

data.drive_time_of_possession = data.drive_time_of_possession.fillna('0:0').apply(time_convert)
data.drive_yards_penalized = data.drive_yards_penalized.fillna(0)

data.loc[data.posteam==data.home_team, 'spread_line'] = -data.loc[data.posteam==data.home_team, 'spread_line']
data['pos_is_home'] = np.where(data.posteam==data.home_team, 1, 0)

# temperature data doesn't exists for 2020
data = data.drop('temp', axis=1)

#%%



# #-------------
# # QB Stats
# #-------------

# # find who the QB was on a given week
# data['yards_gained_random'] = data.yards_gained.apply(lambda x: x + np.random.random(1))
# w_grp = ['season', 'week', 'posteam']
# (w_col, w_met, w_agg) = ('passer_player_name', 'yards_gained_random', 'sum')
# qbs = window_max(data[data.sack==0], w_col, w_grp, w_met, w_agg).drop('yards_gained_random', axis=1)


#--------------
# Receiving Stats
#--------------

sum_cols = ['shotgun', 'no_huddle', 'pass_attempt', 
            'qb_dropback', 'qb_scramble', 'penalty', 'first_down', 'first_down_pass',
            'fourth_down_converted', 'fourth_down_failed',
            'third_down_converted', 'goal_to_go','pass_middle',
            'pass_outside', 'pass_touchdown',
            'qb_hit',  'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
            'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
            'comp_yac_wpa', 'complete_pass', 'incomplete_pass', 'interception',
            'ep', 'epa', 'touchdown', 'fumble', 'fumble_lost',  'xyac_epa',
            'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
            'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
            'yards_gained', 'ydstogo']

mean_cols = ['spread_line', 'total_line', 'vegas_wp', 
             'grass', 'synthetic', 'indoors', 'outdoors',
             'td_prob', 'qb_epa', 'wp', 'wpa',
             'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
             'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
             'comp_yac_wpa',  'ep', 'epa', 'xyac_epa',
             'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
             'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
             'yards_gained', 'ydstogo', 'pos_is_home']

gcols =  ['week', 'season', 'posteam', 'receiver_player_name']
rec_sum = get_agg_stats(data, gcols, sum_cols, 'sum', prefix='rec')
rec_mean = get_agg_stats(data, gcols, mean_cols, 'mean', prefix='rec')
rec = pd.merge(rec_sum, rec_mean, on=gcols)

rec = rec.rename(columns={'receiver_player_name': 'player', 'posteam': 'team'})

#--------------
# Rushing Stats
#--------------

sum_cols = ['shotgun', 'no_huddle', 'rush_attempt', 'first_down',
            'first_down_rush', 'fourth_down_converted', 'fourth_down_failed',
            'third_down_converted', 'goal_to_go', 'run_middle', 'run_outside',
             'rush_touchdown', 'tackled_for_loss',
            'ep', 'epa', 'touchdown', 'fumble','yardline_100', 'yards_after_catch',
            'yards_gained', 'ydstogo']

mean_cols = ['td_prob', 'wp', 'wpa', 'ep', 'epa', 'yardline_100',
             'yards_gained', 'ydstogo']

gcols =  ['week', 'season', 'posteam', 'rusher_player_name']
rush_sum = get_agg_stats(data, gcols, sum_cols, 'sum', prefix='rush')
rush_mean = get_agg_stats(data, gcols, mean_cols, 'mean', prefix='rush')
rush = pd.merge(rush_sum, rush_mean, on=gcols)

rush = rush.rename(columns={'rusher_player_name': 'player', 'posteam': 'team'})

#%%

all_stats = pd.merge(rush, rec, on=['player', 'posteam', 'week', 'season'], how='outer')
all_stats = all_stats.fillna(0)

fp_cols = ['rec_complete_pass_sum', 'rec_yards_gained_sum',
           'rush_yards_gained_sum',  'rec_pass_touchdown_sum', 'rush_rush_touchdown_sum']
all_stats = calc_fp(all_stats, fp_cols, [1, 0.1, 0.1, 7, 7])

all_stats = all_stats.sort_values(by=['player', 'season', 'week']).reset_index(drop=True)
all_stats['y_act'] = all_stats.groupby('player')['fantasy_pts'].shift(-1)

# extract out the names and positions of each group
wr = data.loc[(data.receiver_player_position=='WR'), 
              ['receiver_player_name', 'season', 'receiver_player_position', 'posteam']].drop_duplicates()
wr.columns = ['player', 'season', 'position', 'team']

rb = data.loc[(data.rusher_player_position=='RB'), 
              ['rusher_player_name', 'season', 'rusher_player_position', 'posteam']].drop_duplicates()
rb.columns = ['player', 'season', 'position', 'team']

te = data.loc[(data.receiver_player_position=='TE'), 
              ['receiver_player_name', 'season', 'receiver_player_position', 'posteam']].drop_duplicates()
te.columns = ['player', 'season', 'position', 'team']

wr = pd.merge(wr, all_stats, on=['player', 'season', 'team'])
rb = pd.merge(rb, all_stats, on=['player', 'season', 'team'])
te = pd.merge(te, all_stats, on=['player', 'season', 'team'])

#%%

for df, t_name in zip([wr, rb, te], ['WR_Stats', 'RB_Stats', 'TE_Stats']):
    season_week = df[['season', 'week']].drop_duplicates()
    for _, sw in season_week.iterrows():
     seas = sw[0]
     wk = sw[1]
     dm.delete_from_db('FastR', t_name, f"season={seas} AND week={wk}")
     dm.write_to_db(df, 'FastR', t_name, if_exist='append')


#%%


sum_cols = ['shotgun', 'no_huddle', 'pass_attempt', 
            'qb_dropback', 'qb_scramble', 'penalty', 'first_down', 'first_down_pass',
            'fourth_down_converted', 'fourth_down_failed',
            'third_down_converted', 'third_down_failed', 'goal_to_go','pass_middle',
            'pass_outside', 'pass_touchdown',
            'qb_hit',  'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
            'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
            'comp_yac_wpa', 'complete_pass', 'incomplete_pass', 'interception',
            'ep', 'epa', 'touchdown', 'fumble', 'fumble_lost',  'xyac_epa',
            'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
            'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
            'yards_gained', 'ydstogo',
            'drive_ended_with_score', 'drive_first_downs', 'drive_play_count', 'drive_inside20',
            'drive_time_of_possession']

mean_cols = ['spread_line', 'total_line', 'vegas_wp', 
             'grass', 'synthetic', 'indoors', 'outdoors',
             'td_prob', 'qb_epa', 'wp', 'wpa',
             'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
             'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
             'comp_yac_wpa',  'ep', 'epa', 'xyac_epa',
             'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
             'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
             'yards_gained', 'ydstogo', 'pos_is_home',
             'drive_ended_with_score', 'drive_first_downs', 'drive_play_count', 'drive_inside20',
             'drive_time_of_possession']

gcols =  ['week', 'season', 'posteam', 'passer_player_name']
pass_sum = get_agg_stats(data, gcols, sum_cols, 'sum', prefix='pass')
pass_mean = get_agg_stats(data, gcols, mean_cols, 'mean', prefix='pass')
qb = pd.merge(pass_sum, pass_mean, on=gcols)

qb = qb.rename(columns={'passer_player_name': 'player', 'posteam': 'team'})
qb_names = data.loc[(data.passer_player_position=='QB'), 
              ['passer_player_name', 'season', 'passer_player_position', 'posteam']].drop_duplicates()
qb_names.columns = ['player', 'season', 'position', 'team']

qb = pd.merge(qb_names, qb, on=['player', 'season', 'team'])
qb = pd.merge(qb, rush, on=['player', 'season', 'week', 'team'], how='left')
qb = qb.fillna(0)

fp_cols = ['pass_yards_gained_sum', 'pass_pass_touchdown_sum', 'pass_interception_sum',
           'rush_yards_gained_sum', 'rush_rush_touchdown_sum']
qb = calc_fp(qb, fp_cols, [0.04, 4, -1, 0.1, 6])

qb = qb.sort_values(by=['player', 'season', 'week']).reset_index(drop=True)
qb['y_act'] = qb.groupby('player')['fantasy_pts'].shift(-1)


season_week = qb[['season', 'week']].drop_duplicates()
for _, sw in season_week.iterrows():
    seas = sw[0]
    wk = sw[1]
    dm.delete_from_db('FastR', 'QB_Stats', f"season={seas} AND week={wk}")
    dm.write_to_db(qb, 'FastR', 'QB_Stats', if_exist='append')

#%%

#===================
# Team, Coach, and QB Stats
#===================

#--------------
# Aggregate columns
#--------------

sum_cols = ['shotgun', 'no_huddle', 'pass_attempt', 'rush_attempt',
            'qb_dropback', 'qb_scramble', 'penalty', 'first_down', 'first_down_pass',
            'first_down_rush', 'fourth_down_converted', 'fourth_down_failed',
            'third_down_converted', 'goal_to_go', 'run_middle', 'run_outside', 'pass_middle',
            'pass_outside', 'rush_touchdown', 'tackled_for_loss', 'pass_touchdown',
            'qb_hit', 'sack', 'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
            'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
            'comp_yac_wpa', 'complete_pass', 'incomplete_pass', 'interception',
            'ep', 'epa', 'touchdown', 'fumble', 'fumble_lost',  'xyac_epa',
            'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
            'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
            'yards_gained', 'ydstogo']

mean_cols = ['spread_line', 'total_line', 'vegas_wp', 
             'grass', 'synthetic', 'indoors', 'outdoors',
             'td_prob', 'qb_epa', 'wp', 'wpa',
             'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
             'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
             'comp_yac_wpa',  'ep', 'epa', 'xyac_epa',
             'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
             'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
             'yards_gained', 'ydstogo', 'pos_is_home']

#--------------
# Team Stats
#--------------

gcols = ['season', 'week', 'posteam']

team_sum = get_agg_stats(data, gcols, sum_cols, 'sum', prefix='team')
team_mean = get_agg_stats(data, gcols, mean_cols, 'mean', prefix='team')

team = data[gcols].drop_duplicates()
team = pd.merge(team, team_sum, on=gcols)
team = pd.merge(team, team_mean, on=gcols)
team = team.rename(columns={'posteam': 'team'})
team = team.sort_values(by=['season', 'team', 'week'])

team = team[team.team!=''].reset_index(drop=True)

season_week = team[['season', 'week']].drop_duplicates()
for _, sw in season_week.iterrows():
    seas = sw[0]
    wk = sw[1]
    dm.delete_from_db('FastR', 'Team_Stats', f"season={seas} AND week={wk}")
    dm.write_to_db(team, 'FastR', 'Team_Stats', if_exist='append')

#--------------
# Coach Stats
#--------------

gcols = ['season', 'week', 'coach', 'posteam']

coach_labels = get_coaches()
coach_data = pd.merge(data, coach_labels, on=['season', 'week', 'posteam'])

coach_sum = get_agg_stats(coach_data, gcols, sum_cols, 'sum', prefix='coach')
coach_mean = get_agg_stats(coach_data, gcols, mean_cols, 'mean', prefix='coach')

coaches = coach_data[gcols].drop_duplicates()
coaches = pd.merge(coaches, coach_sum, on=gcols)
coaches = pd.merge(coaches, coach_mean, on=gcols)
coaches = coaches.sort_values(by=['season', 'coach', 'week'])
coaches = coaches.rename(columns={'posteam': 'team'})

season_week = coaches[['season', 'week']].drop_duplicates()
for _, sw in season_week.iterrows():
    seas = sw[0]
    wk = sw[1]
    dm.delete_from_db('FastR', 'Coach_Stats', f"season={seas} AND week={wk}")
    dm.write_to_db(coaches, 'FastR', 'Coach_Stats', if_exist='append')


#%%

# this is close but there seems to be some sort of duplicates for getting drive info
# drive_cols = ['season', 'week', 'posteam',
#               'drive', 'drive_ended_with_score', 'drive_first_downs', 'drive_play_count',
#               'drive_inside20', 'drive_yards_penalized', 'drive_time_of_possession']
# drive_agg_cols = ['drive_ended_with_score', 'drive_first_downs', 'drive_play_count',
#                   'drive_inside20', 'drive_yards_penalized', 'drive_time_of_possession']

# get_agg_stats(data[drive_cols].drop_duplicates(), gcols, drive_agg_cols, 'sum', prefix='team')