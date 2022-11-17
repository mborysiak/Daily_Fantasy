#%%
import pandas as pd
import os
import numpy as np
import requests 
import datetime as dt
import sqlite3
from DataPull_Helper import *
pd.set_option('display.max_columns', 999)

# +
set_year = 2022
set_week = 10

from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#%%
# # #=========================
# # #    Post Game       
# # #=========================

# Pro Football Reference
rush_rz = pd.read_html(f'https://www.pro-football-reference.com/years/{set_year}/redzone-rushing.htm')
rec_rz = pd.read_html(f'https://www.pro-football-reference.com/years/{set_year}/redzone-receiving.htm')
pass_rz = pd.read_html(f'https://www.pro-football-reference.com/years/{set_year}/redzone-passing.htm')

# +
rec_rz_cols = ['player', 'team', 'rz_20_tgt', 'rz_20_receptions', 'rz_20_catch_pct', 'rz_20_rec_yds',
                 'rz_20_rec_tds', 'rz_20_tgt_pct', 'rz_10_tgt', 'rz_10_receptions', 'rz_10_catch_pct',
                 'rz_10_rec_yds', 'rz_10_rec_tds', 'rz_10_tgt_pct', 'link']

rush_rz_cols = ['player', 'team', 'rz_20_rush_att', 'rz_20_rush_yds', 'rz_20_rush_td', 'rz_20_rush_pct',
                'rz_10_rush_att', 'rz_10_rush_yds', 'rz_10_rush_td', 'rz_10_rush_pct', 'rz_5_rush_att',
                'rz_5_rush_yds', 'rz_5_rush_td', 'rz_5_rush_pct', 'link']

pass_rz_cols = ['player', 'team', 'rz_20_pass_complete', 'rz_20_pass_att', 'rz_20_complete_pct',
                'rz_20_pass_yds', 'rz_20_pass_td', 'rz_20_int', 'rz_10_pass_complete', 'rz_10_pass_att',
                'rz_10_complete_pct', 'rz_10_pass_yds', 'rz_10_pass_td', 'rz_10_int', 'link',]

rush_rz_df = rush_rz[0]
rush_rz_df.columns = rush_rz_cols

rec_rz_df = rec_rz[0]
rec_rz_df.columns = rec_rz_cols

pass_rz_df = pass_rz[0]
pass_rz_df.columns = pass_rz_cols

rush_rz_df = rush_rz_df.drop(['link'], axis=1)
rec_rz_df = rec_rz_df.drop(['link'], axis=1)
pass_rz_df = pass_rz_df.drop(['link'], axis=1)

rush_rz_df.team = rush_rz_df.team.map(team_map)
rec_rz_df.team = rec_rz_df.team.map(team_map)
pass_rz_df.team = pass_rz_df.team.map(team_map)

rush_rz_df['week'] = set_week
rec_rz_df['week'] = set_week
pass_rz_df['week'] = set_week

rush_rz_df['year'] = set_year
rec_rz_df['year'] = set_year
pass_rz_df['year'] = set_year

for df in [rush_rz_df, rec_rz_df, pass_rz_df]:
    for c in df.columns:
        try:
            df[c] = df[c].apply(lambda x: float(str(x).replace('%', '')))
        except:
            pass
        df[c] = df[c].fillna(0)
        
#%%
rush_rz_df.player = rush_rz_df.player.apply(dc.name_clean)
rec_rz_df.player = rec_rz_df.player.apply(dc.name_clean)
pass_rz_df.player = pass_rz_df.player.apply(dc.name_clean)

for df, t in zip([rush_rz_df, rec_rz_df, pass_rz_df], ['Rush', 'Rec', 'Pass']):
    dm.delete_from_db('Post_PlayerData', f'PFR_Redzone_{t}', f"week={set_week} AND year-{set_year}")
    dm.write_to_db(df, 'Post_PlayerData', f'PFR_Redzone_{t}', 'append')

#%%
# # Def Vs Position

def_vs_map = {'Arizona Cardinals': 'ARI',
             'Atlanta Falcons': 'ATL',
             'Baltimore Ravens': 'BAL',
             'Buffalo Bills': 'BUF',
             'Carolina Panthers': 'CAR',
             'Chicago Bears': 'CHI',
             'Cincinnati Bengals': 'CIN',
             'Cleveland Browns': 'CLE',
             'Dallas Cowboys': 'DAL',
             'Denver Broncos': 'DEN',
             'Detroit Lions': 'DET',
             'Green Bay Packers': 'GB',
             'Houston Texans': 'HOU',
             'Indianapolis Colts': 'IND',
             'Jacksonville Jaguars': 'JAC',
             'Kansas City Chiefs': 'KC',
             'Los Angeles Chargers': 'LAC',
             'Los Angeles Rams': 'LAR',
             'Miami Dolphins': 'MIA',
             'Minnesota Vikings': 'MIN',
             'New England Patriots': 'NE',
             'New Orleans Saints': 'NO',
             'New York Giants': 'NYG',
             'New York Jets': 'NYJ',
             'Oakland Raiders': 'LVR',
             'Las Vegas Raiders': 'LVR',
             'Philadelphia Eagles': 'PHI',
             'Pittsburgh Steelers': 'PIT',
             'San Francisco 49ers': 'SF',
             'Seattle Seahawks': 'SEA',
             'Tampa Bay Buccaneers': 'TB',
             'Tennessee Titans': 'TEN',
             'Washington Redskins': 'WAS',
             'Washington Football Team': 'WAS'}

def_vs_rb = pd.read_html(f'https://www.pro-football-reference.com/years/{set_year}/fantasy-points-against-RB.htm')
def_vs_te = pd.read_html(f'https://www.pro-football-reference.com/years/{set_year}/fantasy-points-against-TE.htm')
def_vs_qb = pd.read_html(f'https://www.pro-football-reference.com/years/{set_year}/fantasy-points-against-QB.htm')
def_vs_wr = pd.read_html(f'https://www.pro-football-reference.com/years/{set_year}/fantasy-points-against-WR.htm')

#%%
def_vs_rb_df_cols = ['team', 'games', 'rush_att_allowed', 'rush_yds_allowed', 'rush_td_allowed',
                     'tgts_allowed', 'rec_allowed', 'rec_yds_allowed', 'rec_td_allowed', 'rb_fp_allowed', 
                     'rb_dk_pts_allowed', 'rb_fd_pts_allowed', 'rb_fp_allowed_per_game', 'rb_dk_pts_allowed_per_game',
                     'rb_fd_pts_allowed_per_game']

def_vs_wr_df_cols = ['team', 'games', 'tgts_allowed', 'rec_allowed', 'rec_yds_allowed', 'rec_td_allowed', 'wr_fp_allowed', 
                     'wr_dk_pts_allowed', 'wr_fd_pts_allowed', 'wr_fp_allowed_per_game', 'wr_dk_pts_allowed_per_game',
                     'wr_fd_pts_allowed_per_game']

def_vs_qb_df_cols = ['team', 'games', 'complete_allowed', 'att_allowed', 'pass_yds_allowed', 'pass_td_allowed',
                     'int_taken', 'two_pts_allowed', 'sacks_given', 'qb_rush_att_allowed', 'qb_rush_yds_allowed',
                     'qb_rush_td_allowed', 'qb_fp_allowed', 'qb_dk_fp_allowed', 'qb_fd_fp_allowed',
                     'qb_fp_per_game_allowed', 'qb_dk_fp_per_game_allowed', 'qb_fd_fp_per_game_allowed']


def_vs_rb_df = def_vs_rb[0]
def_vs_wr_df = def_vs_wr[0]
def_vs_te_df = def_vs_te[0]
def_vs_qb_df = def_vs_qb[0]

def_vs_rb_df.columns = def_vs_rb_df_cols
def_vs_wr_df.columns = def_vs_wr_df_cols
def_vs_te_df.columns = def_vs_wr_df_cols
def_vs_qb_df.columns = def_vs_qb_df_cols

def_vs_te_df = def_vs_te_df.fillna(0)
def_vs_qb_df = def_vs_qb_df.fillna(0)

def_vs_rb_df['week']= set_week
def_vs_wr_df['week'] = set_week
def_vs_te_df['week'] = set_week
def_vs_qb_df['week']= set_week

def_vs_rb_df['year'] = set_year
def_vs_wr_df['year'] = set_year
def_vs_te_df['year'] = set_year
def_vs_qb_df['year'] = set_year

#%%

for d, t in zip([def_vs_rb_df, def_vs_wr_df, def_vs_te_df, def_vs_qb_df],
                ['RB', 'WR', 'TE', 'QB']):

    d.team = d.team.map(def_vs_map).map(team_map)
    dm.delete_from_db('Post_PlayerData', f'Def_Allowed_{t}', f"week={set_week} AND year={set_year}")
    dm.write_to_db(d, 'Post_PlayerData', f'Def_Allowed_{t}', 'append')

#%%
# # Advanced PFR Stats

ADV_URL = f'https://www.pro-football-reference.com/years/{set_year}'
ADV_PASS = f'{ADV_URL}/passing_advanced.htm#ks_passing_detailed_'
qb_adv = pd.read_html(f'{ADV_PASS}air_yards::none')
qb = pd.read_html(f'{ADV_URL}/passing.htm#passing::none')[0]


def qb_adv_clean(df):
    try:
        df.columns = [c[0]+c[1] if 'Unnamed' not in c[0] else c[1] for c in df.columns ]
        df.columns = [c.replace('/', '_').replace('%', '_pct') for c in df.columns]
    except:
        df.columns = [c.replace('/', '_').replace('%', '_pct') for c in df.columns]
        
    df = df.rename(columns={'Player': 'player', 'Tm': 'team'})
    df = df[df.Rk!='Rk'].reset_index(drop=True)
    return df

#------------
# Advanced QB stats cleanup
#------------

qb_ay = qb_adv_clean(qb_adv[0])
qb_acc = qb_adv_clean(qb_adv[1])
qb_pres = qb_adv_clean(qb_adv[2])
qb_pt = qb_adv_clean(qb_adv[3])

qb_pres.PassingPrss_pct = qb_pres.PassingPrss_pct.apply(fix_pct)

for c in ['PassingDrop_pct', 'PassingBad_pct', 'PassingOnTgt_pct']:
    qb_acc[c] = qb_acc[c].apply(fix_pct)

# +
#------------
# Standard QB stats cleanup
#------------

qb = qb_adv_clean(qb)

qb.QBrec = qb.QBrec.fillna('0-0-0')

qb.QBrec = qb.QBrec.apply(lambda x: x.replace('/', '-'))
qb['wins'] = qb.QBrec.apply(lambda x: float(x.split('-')[0]))
qb['losses'] = qb.QBrec.apply(lambda x: float(x.split('-')[1]))
qb['ties'] = qb.QBrec.apply(lambda x: float(x.split('-')[2]))

qb = qb.drop('QBrec', axis=1)
qb = qb.fillna(0)

#------------
# Save Adv RB/WR stats as CSV in Downloads
#------------

for pos in ['rb', 'wr']:
    try:
        os.replace(f"/Users/mborysia/Downloads/{pos}_week{set_week}.csv", 
                f'{root_path}/Data/OtherData/pfr_adv_stats/{set_year}/{pos}_week{set_week}.csv')
    except: 
        pass

rb = pd.read_csv(f'{root_path}/Data/OtherData/pfr_adv_stats/{set_year}/rb_week{set_week}.csv')
rec = pd.read_csv(f'{root_path}/Data/OtherData/pfr_adv_stats/{set_year}/wr_week{set_week}.csv')

# drop the td column added in middle of 2022
rb = rb.drop('Unnamed: 9', axis=1)
rb_cols = ['rank', 'player', 'team', 'age', 'position', 'games', 'games_started', 'rush_att', 'rush_yds',
           'first_downs', 'yds_before_contact', 'yds_before_contact_att', 'yds_after_contact', 'yac_att', 
           'broke_tackles', 'att_broken']
rb.columns = rb_cols
rb.position = rb.position.fillna('RB')
rb = rb.fillna(0)
rb = rb[rb['rank']!='Rk'].reset_index(drop=True)
rb = rb.dropna(axis=0)

rec_cols = ['rank', 'player', 'team', 'age', 'position', 'games', 'games_started', 'targets', 'rec', 
            'rec_yds', 'rec_td', 'first_downs', 'yds_before_catch', 'yds_before_catch_per_rec', 'yards_after_catch', 
            'yards_after_catch_per_rec', 'adot', 'broken_tackles', 'rec_per_broken', 'drops', 'drop_pct',
            'targ_int', 'targ_rating']

rec.columns = rec_cols

# drop columns that were added in 2020 (td, adot, int, rating)
rec = rec.drop(['position', 'rec_td', 'adot', 'targ_int', 'targ_rating'], axis=1).fillna(0)

#%%

for df, t in zip([qb, qb_ay, qb_acc, qb_pres, qb_pt, rb, rec],
                 ['QB', 'QB_AirYards', 'QB_Accuracy', 'QB_Pressure', 'QB_PlayType', 'RB', 'WR']):
    df['week'] = set_week
    df['year'] = set_year
    df.team = df.team.map(team_map)
    df.player = df.player.apply(dc.name_clean)

    dm.delete_from_db('Post_PlayerData', f'PFR_Advanced_{t}', f"year={set_year} AND week={set_week}")
    dm.write_to_db(df, 'Post_PlayerData', f'PFR_Advanced_{t}', 'append')

#%%
# # Snap Counts

# +
snaps = pd.read_html('https://www.fantasypros.com/nfl/reports/snap-counts/')[0]
snaps = snaps[['Player', 'Pos', 'Team', str(set_week), 'TTL', 'AVG']]
snaps.columns = ['player', 'position', 'team', 'snap_count', 'total_snap_count', 'avg_snap_count']

snap_pct = pd.read_html('https://www.fantasypros.com/nfl/reports/snap-counts/?show=perc')[0]
snap_pct = snap_pct[['Player', 'Pos', 'Team', str(set_week), 'AVG']]
snap_pct.columns = ['player', 'position', 'team', 'snap_pct', 'avg_snap_pct']
snaps = pd.merge(snaps, snap_pct, how='inner', 
                 left_on=['player', 'position', 'team'], right_on=['player', 'position', 'team'])
snaps['year'] = set_year
snaps['week'] = set_week

# snaps.snap_pct = snaps.snap_pct.apply(lambda x: x.replace('%', ''))
# snaps.avg_snap_pct = snaps.avg_snap_pct.apply(lambda x: x.replace('%', ''))
snaps = convert_float(snaps)

dm.delete_from_db('Post_PlayerData', f'Snap_Counts', f"year={set_year} AND week={set_week}")
dm.write_to_db(snaps, 'Post_PlayerData', f'Snap_Counts', 'append')


#%%

df = pd.read_html('https://www.fantasypros.com/nfl/reports/snap-counts/?year=2022')[0]
df = pd.melt(df, id_vars=['Player', 'Pos', 'Team'])
df = df[~df.variable.isin(['AVG', 'TTL'])].dropna().reset_index(drop=True)
df.columns = ['player', 'pos', 'team', 'week', 'snap_counts']
df['year'] = set_year
df.player = df.player.apply(dc.name_clean)

dm.delete_from_db('Post_PlayerData', f'Snap_Counts_V2', f"year={set_year}")
dm.write_to_db(df, 'Post_PlayerData', 'Snap_Counts_V2', 'append')


#%%
# # QBR

espn_map = {'ARI': 'ARI',
         'ATL': 'ATL',
         'BAL': 'BAL',
         'BUF': 'BUF',
         'CAR': 'CAR',
         'CHI': 'CHI',
         'CIN': 'CIN',
         'CLE': 'CLE',
         'DAL': 'DAL',
         'DEN': 'DEN',
         'DET': 'DET',
         'GB': 'GB',
         'HOU': 'HOU',
         'IND': 'IND',
         'JAX': 'JAC',
         'KC': 'KC',
         'LAC': 'LAC',
         'LAR': 'LAR',
         'MIA': 'MIA',
         'MIN': 'MIN',
         'NE': 'NE',
         'NO': 'NO',
         'NYG': 'NYG',
         'NYJ': 'NYJ',
         'OAK': 'LVR',
         'LVR': 'LVR',
         'PHI': 'PHI',
         'PIT': 'PIT',
         'SEA': 'SEA',
         'SF': 'SF',
         'TB': 'TB',
         'TEN': 'TEN',
         'WSH': 'WAS',}

qbr = pd.read_html('http://www.espn.com/nfl/qbr')
qbr = pd.concat([qbr[0], qbr[1]], axis=1)
qbr['week'] = set_week
qbr['year'] = set_year

#%%

dm.delete_from_db('Post_PlayerData', 'ESPN_QBR', f"week={set_week} AND year={set_year}")
dm.write_to_db(qbr, 'Post_PlayerData', 'ESPN_QBR', 'append')

#%%
#-----------
# PFF Defensive Player Downloads
#-----------

try:
    os.replace(f"/Users/mborysia/Downloads/defense_summary.csv", 
                f'{root_path}/Data/OtherData/PFF_Defense/{set_year}/defense_summary_week{set_week}.csv')
except: 
    pass

df = pd.read_csv(f'{root_path}/Data/OtherData/PFF_Defense/{set_year}/defense_summary_week{set_week}.csv')
df['week'] = set_week
df['year'] = set_year

df.player = df.player.apply(dc.name_clean)
df = df.rename(columns={'team_name': 'team'})
df.team = df.team.map(team_map)

dm.delete_from_db('Post_PlayerData', 'Defense_Players', f"week={set_week} AND year={set_year}")
dm.write_to_db(df, 'Post_PlayerData', 'Defense_Players', 'append')

# %%

try:
    os.replace(f"/Users/mborysia/Downloads/offense_blocking.csv", 
                f'{root_path}/Data/OtherData/PFF_Offensive_Line/{set_year}/offense_blocking_week{set_week}.csv')
except: 
    pass

df = pd.read_csv(f'{root_path}/Data/OtherData/PFF_Offensive_Line/{set_year}/offense_blocking_week{set_week}.csv')
df['week'] = set_week
df['year'] = set_year

df.player = df.player.apply(dc.name_clean)
df = df.rename(columns={'team_name': 'team'})
df.team = df.team.map(team_map)

dm.delete_from_db('Post_PlayerData', 'Offensive_Line_Players', f"week={set_week} AND year={set_year}")
dm.write_to_db(df, 'Post_PlayerData', 'Offensive_Line_Players', 'append')
# %%

import requests
import pandas as pd

def get_next_gen(data_type, year, week):
    headers = {
        'accept': 'application/json, text/plain, */*',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
        'referer': 'https://nextgenstats.nfl.com/',
        'accept-language': 'en-US,en;q=0.9,hi;q=0.8',
    }

    response = requests.get(f'https://appapi.ngs.nfl.com/statboard/{data_type}?season={year}&seasonType=REG&week={week}', headers=headers)

    df = pd.read_json(response.content)

    return df



for stat_type in ['receiving', 'rushing', 'passing']:

    next_gen_data = get_next_gen(stat_type, set_year, set_week)

    all_stats = []
    if stat_type == 'receiving': ignore = ('player', 'recTouchdowns', 'receptions', 'targets', 'yards', 'teamId')
    elif stat_type == 'rushing': ignore = ('player', 'rushTouchdowns', 'rushYards', 'rushAttempts', 'teamId')
    elif stat_type == 'passing': ignore = ('player', 'attempts', 'gamesPlayed', 'interceptions', 'nflId', 'season', 'seasonType', 'week', 'teamId', 'completions')

    for i, x in enumerate(next_gen_data.stats.values):
        parse_dict = {k:v for k,v in x.items() if k not in ignore}
        if i==0: 
            cols = list(parse_dict.keys())
            if stat_type == 'rushing': cols.append('playerName')
        
        parse_list = list(parse_dict.values())
        if stat_type == 'rushing': 
            parse_list.append(x['player']['displayName'])

        all_stats.append(parse_list)

    next_gen = pd.DataFrame(all_stats, columns=cols).rename(columns={'playerName': 'player', 'position': 'pos'})
    next_gen = next_gen.dropna()
    next_gen.player = next_gen.player.apply(dc.name_clean)
    next_gen = next_gen.assign(week=set_week, year=set_year)
    
    dm.delete_from_db('Post_PlayerData', f'NextGen_{stat_type.title()}', f"year={set_year} AND week={set_week}")
    dm.write_to_db(next_gen, 'Post_PlayerData', f'NextGen_{stat_type.title()}', 'append')
   
# %%
