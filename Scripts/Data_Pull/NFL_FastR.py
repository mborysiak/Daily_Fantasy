#%%
import pandas as pd 
import pyarrow.parquet as pq
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from DataPull_Helper import *

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set the filepath and name for NFL Fast R data saved from R script
DATA_PATH = f'{root_path}/Data/OtherData/NFL_FastR/'
FNAME = 'raw_data_2022.parquet'

pd.set_option('display.max_columns', 999)

cur_season = 2022

#---------------
# Functions
#---------------

def one_hot_col(df, col):
    return pd.concat([df, pd.get_dummies(df[col])], axis=1).drop(col, axis=1)


def get_agg_stats(data, gcols, stat_cols, agg_type, prefix='', suffix=None):
    if suffix is None: suffix = f'_{agg_type}'
    agg_stat = {c: agg_type for c in stat_cols}
    df = data.groupby(gcols).agg(agg_stat)
    df.columns = [f'{prefix}_{c}{suffix}' for c in df.columns]

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


def calc_fp(df, pts_dict):
    cols = list(pts_dict.keys())
    pts = list(pts_dict.values())
    df['fantasy_pts'] = (df[cols] * pts).sum(axis=1)
    return df


def calc_fp_qb(df, pts_dict, rush_pass):
    cols = list(pts_dict.keys())
    pts = list(pts_dict.values())
    df[f'fantasy_pts_{rush_pass}'] = (df[cols] * pts).sum(axis=1)
    return df


#%%

#---------------
# Load Data
#---------------

# read in the data, filter to real players, and sort by value
data = pq.read_table(f'{DATA_PATH}/{FNAME}').to_pandas()
data = data.sort_values(by='epa', ascending=False).reset_index(drop=True)

defense = data.copy()
defense = defense.loc[defense.season_type=='REG'].reset_index(drop=True)
data = data[(data.play_type.isin(['run', 'pass'])) & \
            (data.season_type=='REG')].reset_index(drop=True)

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

#--------------
# Receiving Stats
#--------------

rec_sum_cols = ['shotgun', 'no_huddle', 'pass_attempt', 
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

rec_mean_cols = ['spread_line', 'total_line', 'vegas_wp', 
             'grass', 'synthetic', 'indoors', 'outdoors',
             'td_prob', 'qb_epa', 'wp', 'wpa',
             'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
             'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
             'comp_yac_wpa',  'ep', 'epa', 'xyac_epa',
             'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
             'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
             'yards_gained', 'ydstogo', 'pos_is_home']

gcols =  ['week', 'season', 'posteam', 'receiver_player_name']
rec_sum = get_agg_stats(data, gcols, rec_sum_cols, 'sum', prefix='rec')
rec_mean = get_agg_stats(data, gcols, rec_mean_cols, 'mean', prefix='rec')
rec = pd.merge(rec_sum, rec_mean, on=gcols)

rec = rec.rename(columns={'receiver_player_name': 'player', 'posteam': 'team'})

#--------------
# Rushing Stats
#--------------

rush_sum_cols = ['shotgun', 'no_huddle', 'rush_attempt', 'first_down',
            'first_down_rush', 'fourth_down_converted', 'fourth_down_failed',
            'third_down_converted', 'goal_to_go', 'run_middle', 'run_outside',
             'rush_touchdown', 'tackled_for_loss',
            'ep', 'epa', 'touchdown', 'fumble', 'fumble_lost','yardline_100', 'yards_after_catch',
            'yards_gained', 'ydstogo']

rush_mean_cols = ['td_prob', 'wp', 'wpa', 'ep', 'epa', 'yardline_100',
             'yards_gained', 'ydstogo']

gcols =  ['week', 'season', 'posteam', 'rusher_player_name']
rush_sum = get_agg_stats(data, gcols, rush_sum_cols, 'sum', prefix='rush')
rush_mean = get_agg_stats(data, gcols, rush_mean_cols, 'mean', prefix='rush')
rush = pd.merge(rush_sum, rush_mean, on=gcols)

rush = rush.rename(columns={'rusher_player_name': 'player', 'posteam': 'team'})


#%%

all_stats = pd.merge(rush, rec, on=['player', 'team', 'week', 'season'], how='outer')
all_stats = all_stats.fillna(0)

# add in the 100 yd bonsuses
all_stats['rec_yd_100_bonus'] = np.where(all_stats.rec_yards_gained_sum >= 100, 1, 0)
all_stats['rush_yd_100_bonus'] = np.where(all_stats.rush_yards_gained_sum >= 100, 1, 0)
all_stats['fumble_lost'] = all_stats.rush_fumble_lost_sum + all_stats.rec_fumble_lost_sum

fp_cols = {'rec_complete_pass_sum': 1, 
           'rec_yards_gained_sum': 0.1,
           'rush_yards_gained_sum': 0.1,  
           'rec_pass_touchdown_sum': 6, 
           'rush_rush_touchdown_sum': 6,
           'rec_yd_100_bonus': 3, 
           'rush_yd_100_bonus': 3,
           'fumble_lost': -1}
all_stats = calc_fp(all_stats, fp_cols)

all_stats = all_stats.sort_values(by=['player', 'season', 'week']).reset_index(drop=True)

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
    df.player = df.player.apply(dc.name_clean)
    df.team = df.team.map(team_map)

    dm.delete_from_db('FastR', t_name, f"season={cur_season}")
    dm.write_to_db(df, 'FastR', t_name, if_exist='append')


#%%


qb_sum_cols = ['shotgun', 'no_huddle', 'pass_attempt', 
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

qb_mean_cols = ['spread_line', 'total_line', 'vegas_wp', 
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

def get_sack(data, gcols):
    sacks = data[data.sack==1].reset_index(drop=True)
    sacks = sacks.groupby(gcols).agg(sack_yards_sum=('yards_gained', 'sum'),
                                     sack_sum=('sack', 'count')).reset_index()
    return sacks

gcols =  ['week', 'season', 'posteam', 'passer_player_name']
sacks = get_sack(data, gcols)

gcols =  ['week', 'season', 'posteam', 'passer_player_name']
pass_sum = get_agg_stats(data[data.sack==0], gcols, qb_sum_cols, 'sum', prefix='pass')
pass_mean = get_agg_stats(data[data.sack==0], gcols, qb_mean_cols, 'mean', prefix='pass')
qb = pd.merge(pass_sum, pass_mean, on=gcols)
qb = pd.merge(qb, sacks, on=gcols, how='left').fillna({'sack_yards_sum': 0, 'sack_sum': 0})

qb = qb.rename(columns={'passer_player_name': 'player', 'posteam': 'team'})
qb_names = data.loc[(data.passer_player_position=='QB'), 
              ['passer_player_name', 'season', 'passer_player_position', 'posteam']].drop_duplicates()
qb_names.columns = ['player', 'season', 'position', 'team']

qb = pd.merge(qb_names, qb, on=['player', 'season', 'team'])
qb = pd.merge(qb, rush, on=['player', 'season', 'week', 'team'], how='left')
qb = qb.fillna(0)

qb['rush_yd_100_bonus'] = np.where(qb.rush_yards_gained_sum >= 100, 1, 0)
qb['pass_yd_300_bonus'] = np.where(qb.pass_yards_gained_sum >= 300, 1, 0)
qb['fumble_lost'] = qb.pass_fumble_lost_sum + qb.rush_fumble_lost_sum

fp_cols = {'pass_yards_gained_sum': 0.04, 
           'pass_pass_touchdown_sum': 4, 
           'pass_interception_sum': -1,
           'fumble_lost': -1,
           'rush_yards_gained_sum': 0.1, 
           'rush_rush_touchdown_sum': 6,
           'rush_yd_100_bonus': 3,
           'pass_yd_300_bonus': 3}
qb = calc_fp(qb, fp_cols)

rush_fp_cols = {'rush_yards_gained_sum': 0.1, 
                'rush_rush_touchdown_sum': 6,
                'rush_yd_100_bonus': 3}

pass_fp_cols = {'pass_yards_gained_sum': 0.04, 
                'pass_pass_touchdown_sum': 4, 
                'pass_interception_sum': -1,
                'fumble_lost': -1,
                'pass_yd_300_bonus': 3}

qb = calc_fp_qb(qb, rush_fp_cols, 'rush')
qb = calc_fp_qb(qb, pass_fp_cols, 'pass')

qb = qb.sort_values(by=['player', 'season', 'week']).reset_index(drop=True)


qb = pd.merge(qb[['week', 'season']].drop_duplicates(), qb, on=['week', 'season'], how='left')

#%%
qb.player = qb.player.apply(dc.name_clean)
qb.team = qb.team.map(team_map)
dm.delete_from_db('FastR', 'QB_Stats', f"season={cur_season}")
dm.write_to_db(qb, 'FastR', 'QB_Stats', if_exist='append')

#%%

#===================
# Team, Coach, and QB Stats
#===================

#--------------
# Aggregate columns
#--------------

rush_team_sum_cols = [f'rush_{c}_sum' for c in rush_sum_cols]
rush_team_mean_cols = [f'rush_{c}_mean' for c in rush_mean_cols]

rec_team_sum_cols = [f'rec_{c}_sum' for c in rec_sum_cols]
rec_team_mean_cols = [f'rec_{c}_mean' for c in rec_mean_cols]

#--------------
# Team Stats
#--------------

gcols = ['season', 'week', 'posteam']

team_rush_sum = get_agg_stats(rush_sum, gcols, rush_team_sum_cols, 'sum', prefix='team', suffix='')
team_rush_mean = get_agg_stats(rush_mean, gcols, rush_team_mean_cols, 'mean', prefix='team', suffix='')
team_rec_sum = get_agg_stats(rec_sum, gcols, rec_team_sum_cols, 'sum', prefix='team', suffix='')
team_rec_mean = get_agg_stats(rec_mean, gcols, rec_team_mean_cols, 'mean', prefix='team', suffix='')

team = data[gcols].drop_duplicates()
for d in [team_rush_sum, team_rush_mean, team_rec_sum, team_rec_mean]:
    team = pd.merge(team, d, on=gcols)

team = team.rename(columns={'posteam': 'team'})
team = team.sort_values(by=['season', 'team', 'week'])

team = team[team.team!=''].reset_index(drop=True)

#%%

team['team'] = team['team'].map(team_map)
dm.delete_from_db('FastR', 'Team_Stats', f"season={cur_season}")
dm.write_to_db(team, 'FastR', 'Team_Stats', if_exist='append')


#%%

# calculate the pts scored for pts allowed on defense
def_allowed = defense.groupby(['defteam', 'week', 'season'], as_index=False).agg({'posteam_score_post': 'max'})
def_allowed = def_allowed.rename(columns={'posteam_score_post': 'def_pts_allowed'})
def_allowed['def_pts_allowed_score'] =  np.select(
    [
        def_allowed['def_pts_allowed']==0,
        def_allowed['def_pts_allowed'].between(1, 6, inclusive=True),
        def_allowed['def_pts_allowed'].between(7, 13, inclusive=True),
        def_allowed['def_pts_allowed'].between(14, 20, inclusive=True),
        def_allowed['def_pts_allowed'].between(21, 27, inclusive=True),
        def_allowed['def_pts_allowed'].between(28, 34, inclusive=True),
    ], 
    [10, 7, 4, 1,0, -1], 
    default=-4
)

# find touchdowns that aren't categorized as return touchdowns
defense['def_td'] = np.where((defense.touchdown==1) & \
                             (defense.rush_touchdown==0) & \
                             (defense.pass_touchdown==0) & \
                             (defense.return_touchdown==0), 1, 0)

# find all blocked kicks
defense['block_kick'] = np.where((defense.field_goal_result=='blocked') | \
                                 (defense.extra_point_result=='blocked') | \
                                 (defense.punt_blocked==1), 1, 0)

#
sum_pt_cols = ['sack', 'interception', 'fumble_lost','return_touchdown','def_td', 'block_kick', 'safety']
def_scoring = defense.groupby(['defteam','week', 'season'], as_index=False)[sum_pt_cols].sum()
def_scoring = pd.merge(def_scoring, def_allowed, on=['defteam', 'week', 'season'])

def_scoring_dict = {
    'sack': 1,
    'interception': 2,
    'fumble_lost': 2,
    'safety': 2,
    'return_touchdown': 6,
    'def_td': 6,
    'block_kick': 2,
    'def_pts_allowed_score': 1
}
def_scoring = calc_fp(def_scoring, def_scoring_dict)
def_scoring = def_scoring.sort_values(by='fantasy_pts', ascending=False)


def_sum_cols = ['pass_attempt', 'rush_attempt','first_down', 'first_down_pass',
                'first_down_rush', 'fourth_down_converted', 'fourth_down_failed',
                'third_down_converted', 'rush_touchdown', 'tackled_for_loss', 'pass_touchdown',
                'qb_hit', 'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
                'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
                'comp_yac_wpa', 'complete_pass', 'incomplete_pass',
                'ep', 'epa', 'touchdown', 'fumble_forced', 'fumble', 'xyac_epa',
                'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
                'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
                'yards_gained', 'ydstogo', 'def_td', 'return_yards']

def_mean_cols = ['spread_line', 'total_line', 'vegas_wp', 
                 'td_prob', 'qb_epa', 'wp', 'wpa',
                 'qb_epa', 'cp', 'cpoe', 'air_epa', 'air_wpa',
                 'air_yards', 'comp_air_epa', 'comp_air_wpa', 'comp_yac_epa',
                 'comp_yac_wpa',  'ep', 'epa', 'xyac_epa',
                 'xyac_fd', 'xyac_mean_yardage', 'xyac_median_yardage', 'xyac_success',
                 'yac_epa', 'yac_wpa', 'yardline_100', 'yards_after_catch',
                 'yards_gained', 'ydstogo', 'return_yards'
                ]

gcols = ['defteam', 'season', 'week']
defense_sum = get_agg_stats(defense, gcols, def_sum_cols, 'sum', prefix='def')
defense_mean = get_agg_stats(defense, gcols, def_mean_cols, 'mean', prefix='def')

def_scoring = pd.merge(def_scoring, defense_sum, on=['defteam', 'season', 'week'])
def_scoring = pd.merge(def_scoring, defense_mean, on=['defteam', 'season', 'week'])

def_scoring = def_scoring.sort_values(by=['defteam', 'season', 'week']).reset_index(drop=True)

def_scoring.defteam = def_scoring.defteam.map(team_map)
dm.delete_from_db('FastR', 'Defense_Stats', f"season={cur_season}")
dm.write_to_db(def_scoring, 'FastR', 'Defense_Stats', if_exist='append')



#%%
# #--------------
# # Coach Stats
# #--------------

# gcols = ['season', 'week', 'coach', 'posteam']

# coach_labels = get_coaches()
# coach_data = pd.merge(data, coach_labels, on=['season', 'week', 'posteam'])

# coach_sum = get_agg_stats(coach_data, gcols, sum_cols, 'sum', prefix='coach')
# coach_mean = get_agg_stats(coach_data, gcols, mean_cols, 'mean', prefix='coach')

# coaches = coach_data[gcols].drop_duplicates()
# coaches = pd.merge(coaches, coach_sum, on=gcols)
# coaches = pd.merge(coaches, coach_mean, on=gcols)
# coaches = coaches.sort_values(by=['season', 'coach', 'week'])
# coaches = coaches.rename(columns={'posteam': 'team'})

# coaches.team = coaches.team.map(team_map)
# dm.delete_from_db('FastR', 'Coach_Stats', f"season={cur_season}")
# dm.write_to_db(coaches, 'FastR', 'Coach_Stats', if_exist='append')
