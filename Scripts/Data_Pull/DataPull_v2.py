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
set_year = 2021
set_week = 1

from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#%%

#=============
# Fantasy Pros
#=============

for set_pos in ['QB', 'RB', 'WR', 'TE', 'DST']:

    try:
        os.replace(f"/Users/mborysia/Downloads/FantasyPros_{set_year}_Week_{set_week}_{set_pos}_Rankings.csv", 
                   f'{root_path}/Data/OtherData/Fantasy_Pros/{set_year}/FantasyPros_{set_year}_Week_{set_week}_{set_pos}_Rankings.csv')
    except:
        pass

    df = pd.read_csv(f'{root_path}/Data/OtherData/Fantasy_Pros/{set_year}/FantasyPros_{set_year}_Week_{set_week}_{set_pos}_Rankings.csv')

    df.columns = [c.lstrip().rstrip() for c in df.columns]
    df = df.rename(columns={
        'RK': 'fp_rank',
        'PLAYER NAME': 'player',
        'TEAM': 'team',
        'PROJ. FPTS': 'projected_points',
    })

    if '(' in df.player[0]:
        df['team'] = df.player.apply(lambda x: x.split('(')[1].replace(')', ''))
        df.player = df.player.apply(lambda x: x.split('(')[0].rstrip().lstrip())

    df['week'] = set_week
    df['year'] = set_year
    df['pos'] = set_pos

    df = df[['player', 'team', 'pos', 'week', 'year', 'fp_rank', 'projected_points']]
    df.player = df.player.apply(dc.name_clean)
    df.team = df.team.apply(lambda x: x.replace('COV_IR', '').lstrip().rstrip())
    df.team = df.team.map(team_map)

    dm.delete_from_db('Pre_PlayerData', 'FantasyPros', f"week={set_week} and year={set_year} and pos='{set_pos}'")
    dm.write_to_db(df, 'Pre_PlayerData', 'FantasyPros', if_exist='append')


#%%

dk_sal = pd.read_html('https://www.fantasypros.com/daily-fantasy/nfl/draftkings-salary-changes.php')[0]
fd_sal = pd.read_html('https://www.fantasypros.com/daily-fantasy/nfl/fanduel-salary-changes.php')[0]
yahoo_sal = pd.read_html('https://www.fantasypros.com/daily-fantasy/nfl/yahoo-salary-changes.php')[0]

def cleanup_sal(df, name):
    df['player'] = df.Player.apply(lambda x: x.split('(')[0].strip(' '))
    df['team'] = df.Player.apply(lambda x: x.split('(')[1].split('-')[0].strip(' '))
    df['position'] = df.Player.apply(lambda x: x.split('(')[1].split('-')[1].strip(' ').strip(')'))
    df = df.rename(columns={'This Week': name})
    df[name] = df[name].apply(lambda x: x.replace('$', '').replace(',', '')).astype('int')
    df = df[['player', 'team', 'position', name]]
    
    return df

dk = cleanup_sal(dk_sal, 'dk_salary')
fd = cleanup_sal(fd_sal, 'fd_salary')
yahoo = cleanup_sal(yahoo_sal, 'yahoo_salary')
salaries = pd.merge(dk, fd, how='inner', 
                    left_on=['player','position', 'team'],
                    right_on=['player', 'position', 'team'])

salaries = pd.merge(salaries, yahoo, how='inner',
                    left_on=['player','position', 'team'],
                    right_on=['player', 'position', 'team'])


salaries['week'] = set_week
salaries['year'] = set_year
player_salary = salaries[salaries.position!='DST']
player_salary.player = player_salary.player.apply(dc.name_clean)

team_salary = salaries[salaries.position=='DST']
team_salary.team = team_salary.team.map(team_map)

dm.delete_from_db('Pre_PlayerData', 'Daily_Salaries', f"week={set_week} and year={set_year}")
dm.write_to_db(player_salary, 'Pre_PlayerData', 'Daily_Salaries', if_exist='append')

dm.delete_from_db('Pre_TeamData', 'Daily_Salaries', f"week={set_week} and year={set_year}")
dm.write_to_db(team_salary, 'Pre_TeamData', 'Daily_Salaries', if_exist='append')

#%%
# Betting Lines
# Paste text into csv with no headers using classic view from here: https://bettingdata.com/nfl/odds

df = pd.read_csv(f'{root_path}/Data/OtherData/Betting_Lines/{set_year}/week{set_week}.csv', header=None)

i = 0
good_data = {
    'away_team': [],
    'home_team': [],
    'away_line': [],
    'home_line': [],
    'away_moneyline': [],
    'home_moneyline': [],
    'over_under': [],
    'gametime': [],
    'gametime_unix': []
}

for _, row in df.iterrows():
    
    if 'at' in row[1]: 
        good_data['home_team'].append(row[1].replace('at ', ''))
        good_data['home_line'].append(row[2])
        good_data['away_team'].append(row[3])
        good_data['away_line'].append(-row[2])
    
    else: 
        good_data['home_team'].append(row[3].replace('at ', ''))
        good_data['home_line'].append(-row[2])
        good_data['away_team'].append(row[1])
        good_data['away_line'].append(row[2])

    good_data['away_moneyline'].append(row[5])
    good_data['home_moneyline'].append(row[6])
    good_data['over_under'].append(row[4])
    good_data['gametime'].append(pd.to_datetime(row[0]))
    good_data['gametime_unix'].append(pd.to_datetime(row[0]).timestamp())

data = pd.DataFrame(good_data)   

data.away_team = data.away_team.map(team_map)
data.home_team = data.home_team.map(team_map)
data['year'] = set_year
data['week'] = set_week

teams = list(data.away_team)
teams.extend(list(data.home_team))
[t for t in team_map.values() if t not in teams]

from collections import Counter
cnter = Counter(teams)
if len([c for c in cnter.values() if c > 1]) > 0:
    print('Check counter for duplicate teams.')
else:
    print('No duplicate teams')

print('Missing Teams:', [t for t in teams if t not in team_map.values()])

#%%

dm.delete_from_db('Pre_TeamData', 'Gambling_Lines', f"week={set_week} and year={set_year}")
dm.write_to_db(data, 'Pre_TeamData', 'Gambling_Lines', if_exist='append')

#%%
# # Weather

set_week = 16
set_year = 2020

cities = dm.read('''SELECT * FROM City_LatLon''', 'Pre_TeamData')
city_data  = dm.read(f'''SELECT a.home_team, b.latitude, b.longitude, a.gametime_unix
                                     FROM Gambling_Lines a
                                     JOIN (SELECT * FROM City_LatLon) b
                                          ON a.home_team = b.team
                                     WHERE week={set_week} 
                                           AND year={set_year}''', 'Pre_TeamData')

# +
KEY = '6c500b03257f351161dbe1fea1aa6558'

weather = []
for _, row in city_data.iterrows():
    city = row.home_team
    print(city)

    gt = row.gametime_unix
    LATITUDE = row.latitude
    LONGITUDE = row.longitude

    ds = requests.get('https://api.darksky.net/forecast/{}/{},{}'.format(KEY, LATITUDE, LONGITUDE))
    ds_data = ds.json()['daily']['data'].extend(ds.json()['hourly']['data'])
    ds_data = ds.json()['daily']['data']
    ds_data.extend(ds.json()['hourly']['data'])

    min_time_diff = 100000000
    for d in ds_data:
        if abs(d['time'] - gt) < min_time_diff:
            best_t = d['time']
            precip_prob = d['precipProbability']
            precip_intensity = d['precipIntensity']
            try:
                precip_type = d['precipType']
            except:
                precip_type = None
            try:
                temp_high = d['temperatureHigh']
                temp_low = d['temperatureLow']
            except:
                temp_high = d['temperature']
                temp_low = d['temperature']
            humidity = d['humidity']
            wind_speed = d['windSpeed']
            wind_gust = d['windGust']
            uv_index = d['uvIndex']
            min_time_diff = abs(d['time'] - gt)

        out = [city]
        out.extend([best_t, precip_prob, precip_intensity, precip_type, temp_high,
                    temp_low, humidity, wind_speed, wind_gust, uv_index])

    weather.append(out)
# -

weather = pd.DataFrame(weather)
weather.columns = ['team', 'gametime_unix', 'precip_prob', 'precip_intensity', 'precip_type',
                   'temp_high', 'temp_low', 'humidity', 'wind_speed', 'wind_gust', 'uv_index']
weather['year'] = set_year
weather['week'] = set_week

weather

#%%
dm.delete_from_db('Pre_TeamData', 'Game_Weather', f"week={set_week} and year={set_year}")
dm.write_to_db(weather, 'Pre_TeamData', 'Game_Weather', if_exist='append')

#%%
# # PFR Position Matchups

rb_match = 'https://www.pro-football-reference.com/fantasy/RB-fantasy-matchups.htm'
wr_match = 'https://www.pro-football-reference.com/fantasy/WR-fantasy-matchups.htm'
te_match = 'https://www.pro-football-reference.com/fantasy/TE-fantasy-matchups.htm'
qb_match = 'https://www.pro-football-reference.com/fantasy/QB-fantasy-matchups.htm'

# +
rb_cols = ['player', 'team', 'games', 'games_started', 'snap_pct', 'rush_att_per_game', 'rush_yds_per_game', 
           'rush_td_per_game', 'tgt_per_game', 'rec_per_game', 'rec_yds_per_game', 'rec_td_per_game',
           'fp_per_game', 'dk_pts_per_game', 'fd_pts_per_game', 'opponent', 'opp_rank', 'opp_fp_per_game', 
           'opp_dk_pt_per_game', 'opp_fd_pt_per_game', 'proj_fp_rank', 'proj_dk_rank', 'proj_fd_rank']

wr_cols = ['player', 'team', 'games', 'games_started', 'snap_pct',  'tgt_per_game', 'rec_per_game', 
           'rec_yds_per_game', 'rec_td_per_game', 'fp_per_game', 'dk_pts_per_game', 'fd_pts_per_game', 
           'opponent', 'opp_rank', 'opp_fp_per_game', 'opp_dk_pt_per_game', 'opp_fd_pt_per_game', 'proj_fp_rank', 
           'proj_dk_rank', 'proj_fd_rank']

qb_cols = ['player', 'team', 'games', 'games_started', 'snap_pct', 'pass_complete_per_game', 'pass_att_per_game', 
           'pass_yds_per_game', 'pass_td_per_game', 'int_per_game', 'sack_per_game', 'rush_att_per_game',
           'rush_yds_per_game', 'rush_td_per_game', 'fp_per_game', 'dk_pts_per_game', 'fd_pts_per_game', 
           'opponent', 'opp_rank', 'opp_fp_per_game', 'opp_dk_pt_per_game', 'opp_fd_pt_per_game', 
           'proj_fp_rank', 'proj_dk_rank', 'proj_fd_rank']

# +
rb = pfr_matchup_cleanup(pd.read_html(rb_match)[0], rb_cols)
wr = pfr_matchup_cleanup(pd.read_html(wr_match)[0], wr_cols)
te = pfr_matchup_cleanup(pd.read_html(te_match)[0], wr_cols)
qb = pfr_matchup_cleanup(pd.read_html(qb_match)[0], qb_cols)

for df in [rb, wr, te, qb]:
    df['week'] = set_week
    df['year'] = set_year
# -

for t, d in zip(['RB', 'WR', 'TE', 'QB'], [rb, wr, te, qb]):
    print('Appending ', t)
    d.player = d.player.apply(dc.name_clean)
    dm.delete_from_db('Pre_PlayerData', f'{t}_PFR_Matchups', f"week={set_week} and year={set_year}")
    dm.write_to_db(d, 'Pre_PlayerData', f'{t}_PFR_Matchups', if_exist='append')

#%%
# ## PFF Matchups

def pff_matchups(label):
    
    try:
        os.replace(f"/Users/mborysia/Downloads/{label}_matchup_chart.csv", 
            f'{root_path}/CSVs/pff_matchups/pff_{label}/{set_year}/{label}_week{set_week}.csv')
    except:
        df = pd.read_csv(f'{root_path}/CSVs/pff_matchups/pff_{label}/{set_year}/{label}_week{set_week}.csv')
    
    if label != 'te':
        df.offTeam = df.offTeam.map(team_map)
        df.defTeam = df.defTeam.map(team_map)

    df['week'] = set_week
    df['year'] = set_year
    
    return df

# wrangle the WR-CB matchups into a table
wr_cb = pff_matchups('wr_cb')
wr_cb = wr_cb.rename(columns={'advantage': 'adv'}).drop('expectedSnaps', axis=1)
wr_cb = wr_cb[wr_cb.defPlayer=='All Defenders'].reset_index(drop=True)
wr_cb.offPlayer = wr_cb.offPlayer.apply(dc.name_clean)

# wrangle the TE matchups into a table
te = pff_matchups('te')
te.offPlayer = te.offPlayer.apply(dc.name_clean)
te['offTeam'] = None
te['defTeam'] = None
te['defPosition'] = None
te_cols = dm.read("SELECT * FROM PFF_TE_Matchups", "Pre_PlayerData").columns
te = te[te_cols]

# wrangle the oline-dline matchups into a table
ol_dl = pff_matchups('oline_dline')

for t, d in zip(['WR_CB', 'TE', 'Oline_Dline'], [wr_cb, te, ol_dl]):

    dm.delete_from_db('Pre_PlayerData', f'PFF_{t}_Matchups', f"week={set_week} and year={set_year}")
    dm.write_to_db(d, 'Pre_PlayerData', f'PFF_{t}_Matchups', if_exist='append')

#%%
# # PFF Rankings + Projections

def pff_proj(label_pre, label_post, folder, rep=True):
    
    try:
        os.replace(f"/Users/mborysia/Downloads/{label_pre}.csv", 
                   f'{root_path}/CSVs/{folder}/{set_year}/{label_post}_week{set_week}.csv')
        df = pd.read_csv(f'{root_path}/CSVs/{folder}/{set_year}/{label_post}_week{set_week}.csv')

    except:
        df = pd.read_csv(f'{root_path}/CSVs/{folder}/{set_year}/{label_post}_week{set_week}.csv')

    df = df.rename(columns={'teamName': 'offTeam', 'games': 'defTeam'})
    df.defTeam = df.defTeam.apply(lambda x: x.replace('@', ''))
    df.offTeam = df.offTeam.map(team_map)
    df.defTeam = df.defTeam.map(team_map)

    df['week'] = set_week
    df['year'] = set_year
    
    players = df[df.position!='dst']
    teams = df[df.position=='dst']
    
    return players, teams


def pff_rank(label_pre, label_post, folder):
    
    try:
        os.replace(f"/Users/mborysia/Downloads/{label_pre}.csv", 
                   f'{root_path}CSVs/{folder}/{set_year}/{label_post}_week{set_week}.csv')
        df = pd.read_csv(f'{root_path}/CSVs/{folder}/{set_year}/{label_post}_week{set_week}.csv')

    except:
        df = pd.read_csv(f'{root_path}/CSVs/{folder}/{set_year}/{label_post}_week{set_week}.csv')

    df = df.rename(columns={'Team': 'offTeam', 'Opponent': 'defTeam'})
    df.defTeam = df.defTeam.apply(lambda x: x.replace('@', ''))
    df.offTeam = df.offTeam.map(team_map)
    df.defTeam = df.defTeam.map(team_map)

    df['week'] = set_week
    df['year'] = set_year

    players = df[df.Position!='DST']
    teams = df[df.Position=='DST']
    
    return players, teams


proj_pl, proj_tm = pff_proj('projections', 'projections', 'pff_proj')
rank1_pl, rank1_tm = pff_rank('week-rankings-export', 'expert_ranks', 'pff_rank')
# rank2_pl, rank2_tm = pff_rank('week-rankings-export-2', 'vor_ranks', 'pff_rank', True)

rank1_pl = rank1_pl.drop(f'w{set_week}', axis=1)
rank1_tm = rank1_tm.drop(f'w{set_week}', axis=1)

# rank2_pl = rank2_pl.drop(f'w{set_week}', axis=1)
# rank2_tm = rank2_tm.drop(f'w{set_week}', axis=1)

proj_pl.playerName = proj_pl.playerName.apply(dc.name_clean)
rank1_pl.Name = rank1_pl.Name.apply(dc.name_clean)
# rank2_pl.Name = rank2_pl.Name.apply(dc.name_clean)

proj_pl = proj_pl.rename(columns={'playerName': 'player'})
rank1_pl = rank1_pl.rename(columns={'Name': 'player'})

#%%

tables = ['Proj', 'Proj', 'Expert', 'Expert']#, 'VOR']
dfs = [proj_pl, proj_tm, rank1_pl, rank1_tm]#, rank2_pl]
dbs = ['Pre_PlayerData', 'Pre_TeamData', 'Pre_PlayerData', 'Pre_TeamData']

for t, d, db in zip(tables, dfs, dbs):
    dm.delete_from_db(db, f'PFF_{t}_Ranks', f"week={set_week} AND year={set_year}")
    dm.write_to_db(d, db, f'PFF_{t}_Ranks', 'append')

#%%

#--------------
# Pull in DK Salaries & Id's
#--------------

try:
    os.replace(f"/Users/mborysia/Downloads/DKSalaries.csv", 
            f'{root_path}/Data/OtherData/DK_Salaries/{set_year}/DKSalaries_week{set_week}.csv')
except:
    salary_id = pd.read_csv(f'{root_path}/Data/OtherData/DK_Salaries/{set_year}/DKSalaries_week{set_week}.csv',
                            skiprows=7).dropna(axis=1)

salary_id = salary_id.rename(columns={'Name': 'player', 'Salary': 'salary', 'ID': 'player_id'})
salary_id.player = salary_id.player.apply(dc.name_clean)

defense = salary_id.loc[salary_id.Position=='DST', ['TeamAbbrev', 'salary', 'player_id']]
salary_id = salary_id.loc[salary_id.Position != 'DST']
salary_id = salary_id[['player', 'salary', 'player_id']]
salary_id = pd.concat([salary_id, defense.rename(columns={'TeamAbbrev': 'player'})], axis=0)

salary = salary_id[['player', 'salary']]
salary = salary.assign(year=set_year).assign(league=set_week)

ids = salary_id[['player', 'player_id']]
ids = ids.assign(year=set_year).assign(league=set_week)

dm.delete_from_db('Simulation', 'Salaries', f"league={set_week} AND year={set_year}")
dm.write_to_db(salary, 'Simulation', 'Salaries', 'append')

dm.delete_from_db('Simulation', 'Player_Ids', f"league={set_week} AND year={set_year}")
dm.write_to_db(ids, 'Simulation', 'Player_Ids', 'append')

# update the salaries with actual DK salaries from the website
other_sal = dm.read(f'''SELECT * 
                        FROM Daily_Salaries
                        WHERE week={set_week} AND year={set_year}''', 'Pre_PlayerData')
other_sal = pd.merge(other_sal, salary[['player', 'salary']], on=['player'])
other_sal['dk_salary'] = other_sal.salary
other_sal = other_sal.drop('salary', axis=1)
dm.delete_from_db('Pre_PlayerData', 'Daily_Salaries', f"week={set_week} AND year={set_year}")
dm.write_to_db(other_sal, 'Pre_PlayerData', 'Daily_Salaries', 'append')

#%%
# # #============
# # #    Post Game       
# # #============

# # Pro Football Reference

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

append_to_db(rush_rz_df, db_name='Post_PlayerData.sqlite3', table_name='PFR_Redzone_Rush', 
             if_exist='append', set_week=set_week, set_year=set_year)
append_to_db(rec_rz_df, db_name='Post_PlayerData.sqlite3', table_name='PFR_Redzone_Rec', 
             if_exist='append', set_week=set_week, set_year=set_year)
append_to_db(pass_rz_df, db_name='Post_PlayerData.sqlite3', table_name='PFR_Redzone_Pass', 
             if_exist='append', set_week=set_week, set_year=set_year)

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

def_vs_rb_df.team = def_vs_rb_df.team.map(def_vs_map)
def_vs_wr_df.team = def_vs_wr_df.team.map(def_vs_map)
def_vs_te_df.team = def_vs_te_df.team.map(def_vs_map)
def_vs_qb_df.team = def_vs_qb_df.team.map(def_vs_map)

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
    append_to_db(d, db_name='Post_PlayerData.sqlite3', table_name=f'Def_Allowed_{t}', 
                 if_exist='append', set_week=set_week, set_year=set_year)

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


#%%
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

# +
#------------
# Save Adv RB/WR stats as CSV in Downloads
#------------

for pos in ['rb', 'wr']:
    os.replace(f"/Users/Mark/Downloads/{pos}_week{set_week}.csv", 
               f'{root_path}CSVs/pfr_adv_stats/{set_year}/{pos}_week{set_week}.csv')

rb = pd.read_csv(f'/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/CSVs/pfr_adv_stats/{set_year}/rb_week{set_week}.csv')
rec = pd.read_csv(f'/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/CSVs/pfr_adv_stats/{set_year}/wr_week{set_week}.csv')

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
# -

for df, t in zip([qb, qb_ay, qb_acc, qb_pres, qb_pt, rb, rec],
                 ['QB', 'QB_AirYards', 'QB_Accuracy', 'QB_Pressure', 'QB_PlayType', 'RB', 'WR']):
    df['week'] = set_week
    df['year'] = set_year
    df.team = df.team.map(pfr_fp_map)
    append_to_db(df, db_name='Post_PlayerData.sqlite3', table_name=f'PFR_Advanced_{t}', 
                 if_exist='append', set_week=set_week, set_year=set_year)

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

snaps.snap_pct = snaps.snap_pct.apply(lambda x: x.replace('%', ''))
snaps.avg_snap_pct = snaps.avg_snap_pct.apply(lambda x: x.replace('%', ''))
snaps = convert_float(snaps)
# -

append_to_db(snaps, db_name='Post_PlayerData.sqlite3', table_name='Snap_Counts', 
                 if_exist='append', set_week=set_week, set_year=set_year)
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

append_to_db(qbr, db_name='Post_PlayerData.sqlite3', table_name='ESPN_QBR', 
                 if_exist='append', set_week=set_week, set_year=set_year)

# # Backup Database

copy_db('Post_PlayerData', root_path, set_week, set_year)


#%%
# # Rework Air Yards

# # +
# df_air_json = requests.get(f'http://api.airyards.com/{set_year}/weeks')

# df_air_json

# # +
# import requests

# df_air_json = requests.get(f'http://api.airyards.com/{set_year}/weeks')
# df_air = pd.DataFrame(df_air_json.json())

# df_air = df_air[df_air.position.isin(['WR', 'RB', 'TE'])].reset_index(drop=True)
# df_air = df_air[['full_name', 'position', 'team', 'week', 'air_yards', 'tar', 'rec', 'rec_yards', 'tm_att', 
#                    'team_air', 'aypt', 'racr', 'ms_air_yards', 'target_share', 'wopr', 'yac']]
# df_air['year'] = set_year
# df_air = df_air.rename(columns={'full_name': 'player'})

# col_agg = {'air_yards': 'sum',
#            'rec_yards': 'sum',
#            'rec': 'sum',
#            'tar': 'sum',
#            'tm_att': 'sum',
#            'team_air': 'sum',
#            'yac': 'sum',
#            'games': 'count'}

# df_air_player = df_air.groupby(['player', 'team', 'year', 'position']).agg(col_agg).reset_index()

# dup_players = df_air_player.groupby('player').agg({'tar': 'max'}).reset_index()
# dup_teams = pd.merge(df_air_player[['player', 'team', 'tar']], dup_players, on=['player', 'tar']).drop('tar', axis=1)

# col_agg['games'] = 'sum'
# df_air_player = df_air_player.groupby(['player', 'year', 'position']).agg(col_agg).reset_index()
# df_air_player = pd.merge(df_air_player, dup_teams, on='player')

# df_air_player['ay_per_game'] = df_air_player.air_yards / df_air_player.games
# df_air_player['yac_per_game'] = df_air_player.yac / df_air_player.games
# df_air_player['racr'] = df_air_player.rec_yards / (df_air_player.air_yards + 1.5)
# df_air_player['ay_per_tar'] = df_air_player.air_yards / (df_air_player.tar + 1.5)
# df_air_player['ay_per_rec'] = df_air_player.air_yards / (df_air_player.rec + 1.5)
# df_air_player['tgt_mkt_share'] = df_air_player.tar / df_air_player.tm_att
# df_air_player['ay_converted'] = (df_air_player.rec_yards - df_air_player.yac) / (df_air_player.air_yards+1.5)
# df_air_player['yac_per_ay'] = df_air_player.yac / (df_air_player.air_yards+1.5)
# df_air_player['air_yd_mkt_share'] = df_air_player.air_yards / df_air_player.team_air
# df_air_player['wopr'] = 1.5*df_air_player.tgt_mkt_share + 0.7*df_air_player.air_yd_mkt_share
# df_air_player['rec_yds_per_ay'] = df_air_player.rec_yards / (df_air_player.air_yards + 1)
# df_air_player['yac_plus_ay'] = df_air_player.yac + df_air_player.air_yards

# team_agg = {'tm_att': 'max',
#             'team_air': 'max',
#             'rec_yards': 'sum',
#             'yac': 'sum'}

# df_air_team = df_air.groupby(['team', 'games']).agg(team_agg)
# df_air_team = df_air_team.groupby('team').agg({'tm_att': 'sum', 'team_air': 'sum', 'rec_yards': 'sum', 'yac': 'sum'})
# df_air_team = df_air_team.rename(columns={'yac': 'team_yac'})
# df_air_team['tm_air_per_att'] = df_air_team.team_air / df_air_team.tm_att
# df_air_team['tm_ay_converted'] = (df_air_team.rec_yards - df_air_team.team_yac) / df_air_team.team_air
# df_air_team['tm_rec_yds_per_ay'] = df_air_team.rec_yards / df_air_team.team_air
# df_air_team['tm_yac_per_ay'] = df_air_team.team_yac / df_air_team.team_air
# df_air_team = df_air_team.drop('rec_yards', axis=1)

# df_air_player = df_air_player.drop(['tm_att', 'team_air', 'rec_yards', 'rec'], axis=1)
# df_air_player = pd.merge(df_air_player, df_air_team, on=['team'])
# df_air_player['tm_ay_per_game'] = df_air_player.team_air / df_air_player.games
# df_air_player['total_tgt_mkt_share'] = df_air_player.tar / df_air_player.tm_att
# df_air_player['yac_mkt_share'] = df_air_player.yac / df_air_player.team_yac
# df_air_player['yac_wopr'] =  1.5*df_air_player.tgt_mkt_share + 0.7*df_air_player.air_yd_mkt_share + df_air_player.yac_mkt_share
# df_air_player = df_air_player.drop(['tar', 'games', 'team'], axis=1)
# -


