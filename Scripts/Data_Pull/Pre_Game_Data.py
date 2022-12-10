#%%
import pandas as pd
import os
import numpy as np
import requests 
import datetime as dt
import sqlite3
from DataPull_Helper import *
pd.set_option('display.max_columns', 999)
import shutil as su

# +
set_year = 2022
set_week = 13

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

    if set_pos == 'WR':
        df = df[df.player != 'Cordarrelle Patterson']

    df.loc[df.projected_points=='-', 'projected_points'] = 0

    df = create_adj_ranks(df, 'fp_rank', f"WHERE pos='{set_pos}'", 'FantasyPros', dm)
    
    dm.delete_from_db('Pre_PlayerData', 'FantasyPros', f"week={set_week} and year={set_year} and pos='{set_pos}'")
    dm.write_to_db(df, 'Pre_PlayerData', 'FantasyPros', if_exist='append')


#%%
# # PFF Rankings + Projections

def pff_proj(label_pre, label_post, folder, rep=True):
    
    try:
        os.replace(f"/Users/mborysia/Downloads/{label_pre}.csv", 
                   f'{root_path}/Data/OtherData/{folder}/{set_year}/{label_post}_week{set_week}.csv')
        df = pd.read_csv(f'{root_path}/Data/OtherData/{folder}/{set_year}/{label_post}_week{set_week}.csv')

    except:
        df = pd.read_csv(f'{root_path}/Data/OtherData/{folder}/{set_year}/{label_post}_week{set_week}.csv')

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
                   f'{root_path}/Data/OtherData/{folder}/{set_year}/{label_post}_week{set_week}.csv')
        df = pd.read_csv(f'{root_path}/Data/OtherData/{folder}/{set_year}/{label_post}_week{set_week}.csv')

    except:
        df = pd.read_csv(f'{root_path}/Data/OtherData/{folder}/{set_year}/{label_post}_week{set_week}.csv')

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

all_df = pd.DataFrame()
for p in ['QB', 'RB', 'WR', 'TE', 'K']:
    base_df = rank1_pl[rank1_pl.Position==p].copy()

    for c in ['expertKevinCole', 'expertConsensus', 'expertNathanJahnke']:
        cur_df = create_adj_ranks(base_df, c, f"WHERE Position='{p}'", 'PFF_Expert_Ranks', dm)
        base_df = pd.merge(base_df, cur_df[['player', 'offTeam', f'rankadj_{c}', f'playeradj_{c}']], on=['player', 'offTeam'], how='left')
    all_df = pd.concat([all_df, base_df], axis=0)

rank1_pl = all_df.copy()
# if set_week==7 and set_year==2022:
#     old = dm.read("SELECT * FROM PFF_Proj_Ranks", 'Pre_PlayerData')
#     old = old[(old.player=='Sterling Shepard') & (old.week==3) & (old.year==2022)]
#     old = old.assign(player="Wan'Dale Robinson", defTeam='JAC', week=7, year=2022, salary=4500, byeWeek=9)
#     proj_pl = pd.concat([proj_pl, old], axis=0)

    # rank1_pl.loc[rank1_pl.player=='Clyde Edwards Helaire', 'player'] = 'placehold'
    # rank1_pl.loc[rank1_pl.player=='Jerick Mckinnon', 'player'] = 'Clyde Edwards Helaire'
    # rank1_pl.loc[rank1_pl.player=='Isiah Pacheco', 'player'] = 'Jerick Mckinnon'
    # rank1_pl.loc[rank1_pl.player=='placehold', 'player'] = 'Isiah Pacheco'

    # proj_pl.loc[proj_pl.player=='Clyde Edwards Helaire', 'player'] = 'placehold'
    # proj_pl.loc[proj_pl.player=='Jerick Mckinnon', 'player'] = 'Clyde Edwards Helaire'
    # proj_pl.loc[proj_pl.player=='Isiah Pacheco', 'player'] = 'Jerick Mckinnon'
    # proj_pl.loc[proj_pl.player=='placehold', 'player'] = 'Isiah Pacheco'

#%%

tables = ['Proj', 'Proj', 'Expert', 'Expert']#, 'VOR']
dfs = [proj_pl, proj_tm, rank1_pl, rank1_tm]#, rank2_pl]
dbs = ['Pre_PlayerData', 'Pre_TeamData', 'Pre_PlayerData', 'Pre_TeamData']

for t, d, db in zip(tables, dfs, dbs):
    dm.delete_from_db(db, f'PFF_{t}_Ranks', f"week={set_week} AND year={set_year}")
    dm.write_to_db(d, db, f'PFF_{t}_Ranks', 'append')


#%%
# ## PFF Matchups

def pff_matchups(label):
    
    try:
        os.replace(f"/Users/mborysia/Downloads/{label}_matchup_chart.csv", 
            f'{root_path}/Data/OtherData/pff_matchups/pff_{label}/{set_year}/{label}_week{set_week}.csv')
    except:
        print('No file to pull')
    df = pd.read_csv(f'{root_path}/Data/OtherData/pff_matchups/pff_{label}/{set_year}/{label}_week{set_week}.csv')
    
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

teams = dm.read(f'''SELECT player, team, week, year
                    FROM (
                    SELECT CASE WHEN pos!='DST' THEN player ELSE team END player, 
                                team,
                                week, 
                                year,
                                rank() OVER (PARTITION BY player, year, week
                                                   ORDER BY  projected_points DESC) rn 
                    FROM FantasyPros
                    ) WHERE rn=1''', 'Pre_PlayerData').drop_duplicates()

dm.write_to_db(teams, 'Simulation', 'Player_Teams', 'replace')


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
salaries = pd.merge(dk, fd, how='left', 
                    left_on=['player','position', 'team'],
                    right_on=['player', 'position', 'team'])

salaries = pd.merge(salaries, yahoo, how='left',
                    left_on=['player','position', 'team'],
                    right_on=['player', 'position', 'team'])


salaries['week'] = set_week
salaries['year'] = set_year
player_salary = salaries[salaries.position!='DST']
player_salary.player = player_salary.player.apply(dc.name_clean)
player_salary.team = player_salary.team.map(team_map)

team_salary = salaries[salaries.position=='DST']
team_salary.team = team_salary.team.map(team_map)

player_salary.sort_values(by='dk_salary', ascending=False).iloc[:20]

#%%
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
    
    if 'at ' in row[1]: 
        
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

data.away_team = data.away_team.map(name_map).map(team_map)
data.home_team = data.home_team.map(name_map).map(team_map)
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

print('Missing Teams:', set([t for t in team_map.values() if t not in teams]))

#%%

dm.delete_from_db('Pre_TeamData', 'Gambling_Lines', f"week={set_week} and year={set_year}")
dm.write_to_db(data, 'Pre_TeamData', 'Gambling_Lines', if_exist='append')

glines = dm.read("SELECT * FROM Gambling_Lines", 'Pre_TeamData')
dm.write_to_db(glines, 'Simulation', 'Gambling_Lines', 'replace')

#%%
# # Weather

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

#--------------
# Pull in DK Salaries & Id's
#--------------

try:
    su.copyfile(f'//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/MBorysiak/DK/Lineups/{set_year}/week{set_week}/DKSalaries.csv', 
                f'{root_path}/Data/OtherData/DK_Salaries/{set_year}/DKSalaries_week{set_week}.csv')   
except:
    print('No file to move')

ids = pd.read_csv(f'{root_path}/Data/OtherData/DK_Salaries/{set_year}/DKSalaries_week{set_week}.csv',
                  skiprows=7).dropna(axis=1)
ids = ids[['Name', 'ID']].rename(columns={'ID': 'GoodId'})

salary_id = pd.read_csv(f'{root_path}/Data/OtherData/DK_Salaries/{set_year}/DKSalaries_week{set_week}.csv',
                        skiprows=7).dropna(axis=1)
salary_id = pd.merge(salary_id, ids, on='Name', how='left')
salary_id.loc[~salary_id.GoodId.isnull(), 'ID'] = salary_id.loc[~salary_id.GoodId.isnull(), 'GoodId']

salary_id = salary_id.rename(columns={'Name': 'player', 'Salary': 'salary', 'ID': 'player_id'})

defense = salary_id.loc[salary_id['Roster Position']=='DST', ['TeamAbbrev', 'salary', 'player_id']]
defense.TeamAbbrev = defense.TeamAbbrev.apply(dc.name_clean).apply(lambda x: x.upper())

salary_id = salary_id.loc[salary_id['Roster Position'] != 'DST']
salary_id = salary_id[['player', 'salary', 'player_id']]
salary_id.player = salary_id.player.apply(dc.name_clean)

salary_id = pd.concat([salary_id, defense.rename(columns={'TeamAbbrev': 'player'})], axis=0)

salary = salary_id[['player', 'salary']]
salary = salary.assign(year=set_year).assign(league=set_week)

ids = salary_id[['player', 'player_id']]
ids = ids.assign(year=set_year).assign(league=set_week)


dm.delete_from_db('Simulation', 'Salaries', f"league={set_week} AND year={set_year}")
dm.write_to_db(salary, 'Simulation', 'Salaries', 'append')

dm.delete_from_db('Simulation', 'Player_Ids', f"league={set_week} AND year={set_year}")
dm.write_to_db(ids, 'Simulation', 'Player_Ids', 'append')


# %%

# https://www.nfl.com/injuries/league/2022/reg12

df = pd.read_csv(f'{root_path}/Data/OtherData/Injury_Status/{set_year}/week{set_week}.csv', 
                 encoding='latin', skip_blank_lines=True, error_bad_lines=False)
df.columns = ['player', 'pos', 'injuries', 'practice_status', 'game_status']
df = df[df.player!='Player'].dropna(axis=0, thresh=3).reset_index(drop=True)

df.loc[(df['practice_status'] == 'Did Not Participate In Practice') & \
        (df.game_status.isnull()), 'game_status'] = 'Questionable'

df['week'] = set_week
df['year'] = set_year
df.player = df.player.apply(dc.name_clean)

dm.delete_from_db('Pre_PlayerData', 'PlayerInjuries', f"week={set_week} AND year={set_year}")
dm.write_to_db(df, 'Pre_PlayerData', 'PlayerInjuries', 'append')


# %%

try:
    os.replace(f"/Users/mborysia/Downloads/dk-ownership.csv", 
                f'{root_path}/Data/OtherData/projected_ownership/{set_year}/week{set_week}.csv')
except:
    pass

dk = pd.read_csv(f'{root_path}/Data/OtherData/projected_ownership/{set_year}/week{set_week}.csv')

dk.player = dk.player.apply(dc.name_clean)
dk.loc[dk.position=='D', 'player'] = dk.loc[dk.position=='D', 'team']
dk.loc[dk.position=='D', 'position'] = 'DST'

dk = dk[['player', 'position', 'ownership']]
dk['week'] = set_week
dk['year'] = set_year

dm.delete_from_db('Simulation', 'Projected_Ownership', f"week={set_week} AND year={set_year}")
dm.write_to_db(dk, 'Simulation', 'Projected_Ownership', 'replace')

#%%

try:
    su.copyfile(f'//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/MBorysiak/DK/Lineups/{set_year}/week{set_week}/DKSalariesShowdown.csv',
                f'{root_path}/Data/OtherData/DK_Salaries/{set_year}/DKSalariesShowdown_week{set_week}.csv')   
except:
    print('No file to move')

ids = pd.read_csv(f'{root_path}/Data/OtherData/DK_Salaries/{set_year}/DKSalariesShowdown_week{set_week}.csv',
                  skiprows=7).dropna(axis=1)
ids = ids[['Name', 'ID']].rename(columns={'ID': 'GoodId'})

salary_id = pd.read_csv(f'{root_path}/Data/OtherData/DK_Salaries/{set_year}/DKSalariesShowdown_week{set_week}.csv',
                        skiprows=7).dropna(axis=1)
salary_id = pd.merge(salary_id, ids, on='Name', how='left')
salary_id.loc[~salary_id.GoodId.isnull(), 'ID'] = salary_id.loc[~salary_id.GoodId.isnull(), 'GoodId']

salary_id = salary_id.rename(columns={'Name': 'player', 'Salary': 'salary', 'ID': 'player_id', 'Roster Position': 'pos'})
salary_id.player = salary_id.player.apply(dc.name_clean)

defense = salary_id.loc[salary_id['Position']=='DST', ['TeamAbbrev', 'salary', 'player_id']]
defense = defense.rename(columns={'TeamAbbrev': 'player'})
max_d = defense.groupby('player').agg({'salary': 'max'}).reset_index()
max_d['IsCPT'] = 1
defense = pd.merge(defense, max_d, how='left', on=['player', 'salary'])
defense['pos'] = np.where(defense.IsCPT==1, 'CPT', 'FLEX')
defense = defense[['player', 'pos', 'salary', 'player_id']]

salary_id = salary_id.loc[salary_id['Position'] != 'DST']
salary_id = salary_id[['player', 'pos', 'salary', 'player_id']]
salary_id = pd.concat([salary_id, defense.rename(columns={'TeamAbbrev': 'player'})], axis=0)

salary = salary_id[['player', 'pos', 'salary']]
salary = salary.assign(year=set_year).assign(league=set_week)
salary = salary.drop_duplicates().reset_index(drop=True)

salary.loc[salary.player=='Eli Mitchell', 'player'] = 'Elijah Mitchell'

ids = salary_id[['player', 'player_id']]
ids = ids.assign(year=set_year).assign(league=set_week)
ids.loc[ids.player=='Eli Mitchell', 'player'] = 'Elijah Mitchell'

dm.delete_from_db('Simulation', 'Showdown_Salaries', f"league={set_week} AND year={set_year}")
dm.write_to_db(salary, 'Simulation', 'Showdown_Salaries', 'append')

dm.delete_from_db('Simulation', 'Showdown_Player_Ids', f"league={set_week} AND year={set_year}")
dm.write_to_db(ids, 'Simulation', 'Showdown_Player_Ids', 'append')

k_points = pd.read_csv(f'{root_path}/Data/OtherData/DK_Salaries/{set_year}/DKSalariesShowdown_week{set_week}.csv',
                        skiprows=7).dropna(axis=1)

k_points = k_points.loc[k_points.Position=='K', ['Name', 'AvgPointsPerGame']].drop_duplicates()
k_points.columns = ['player', 'pred_fp_per_game']
k_points.player = k_points.player.apply(dc.name_clean)

k_points['std_dev'] = k_points.pred_fp_per_game / 1.5
k_points['max_score'] = k_points.pred_fp_per_game * 3
k_points['min_score'] = 0
k_points['week'] = set_week
k_points['year'] = set_year
k_points['pos'] = 'K'

k_points['model_type'] = 'full_model'
vers = dm.read("SELECT version FROM Model_Predictions WHERE version IS NOT NULL", 'Simulation').iloc[-1]['version']
k_points['version'] = vers
dm.write_to_db(k_points, 'Simulation', 'Model_Predictions', 'append')

# %%

# def clean_alt_salary(df, sal):
#     cols = ['player', 'position', 'year', 'week', 'salary', 'score', 'factor', 'rank']
#     df.columns = cols
#     df.salary = df.salary.apply(lambda x: int(x.replace('$', '')))

#     if sal == 'fd': df.loc[df.position=='D', 'position'] = 'DST'
#     df_pl = df[df.position != 'DST'].reset_index(drop=True)
#     df_team = df[df.position == 'DST'].reset_index(drop=True)

#     df_pl.player = df_pl.player.apply(lambda x: x.split(',')[1].lstrip() + ' ' + x.split(',')[0])
#     df_pl.player = df_pl.player.apply(dc.name_clean)

#     if sal == 'dk':
#         df_team.player = df_team.player.apply(lambda x: x.replace(',', ''))
#         df_team.player = df_team.player.map(full_team_map)
#     else:    
#         alt_team_map = {' '.join(k.split(' ')[:-1]): v for k, v in full_team_map.items()}
#         df_team.player = df_team.player.map(alt_team_map)

#     df_team['team'] = df_team.player
#     df_team = df_team[['player', 'team', 'position', 'salary']].rename(columns={'salary': f'{sal}_salary'})

#     teams = dm.read(f'''SELECT player, team
#                         FROM (
#                         SELECT CASE WHEN pos!='DST' THEN player ELSE team END player, 
#                             team,
#                             row_number() OVER (PARTITION BY player ORDER BY year, week, projected_points DESC) rn 
#                         FROM FantasyPros
#                         ) WHERE rn=1''', 'Pre_PlayerData')

#     df_pl = pd.merge(df_pl, teams, on=['player'])

#     df_pl = df_pl[['player', 'team', 'position', 'salary']].rename(columns={'salary': f'{sal}_salary'})


#     return df_pl, df_team

# dk = pd.read_html('https://www.footballdiehards.com/fantasyfootball/dailygames/Draftkings-Salary-data.cfm')[0]
# dk_pl, dk_team = clean_alt_salary(dk, 'dk')

# fd = pd.read_html('https://www.footballdiehards.com/fantasyfootball/dailygames/Fanduel-Salary-data.cfm')[0]
# fd_pl, fd_team = clean_alt_salary(fd, 'fd')


# yahoo_pl = dm.read('''SELECT player, team, position, yahoo_salary
#                           FROM Daily_Salaries
#                           WHERE week=11 AND year=2021''', 'Pre_PlayerData')
# yahoo_team = dm.read('''SELECT team, position, yahoo_salary
#                         FROM Daily_Salaries
#                         WHERE week=9 AND year=2021''', 'Pre_TeamData')
# df_pl = pd.merge(dk_pl, fd_pl, on=['player', 'team', 'position'], how='left')
# df_pl = pd.merge(df_pl, yahoo_pl, on=['player', 'team', 'position'], how='left')
# df_pl = df_pl.assign(week=set_week).assign(year=set_year)

# df_team = pd.merge(dk_team, fd_team, on=['player', 'team', 'position'], how='left')
# df_team = pd.merge(df_team, yahoo_team, on=['team', 'position'], how='left')
# df_team = df_team.assign(week=set_week).assign(year=set_year)

# dm.delete_from_db('Pre_PlayerData', 'Daily_Salaries', f"week={set_week} AND year={set_year}")
# dm.write_to_db(df_pl, 'Pre_PlayerData', 'Daily_Salaries', 'append')

# dm.delete_from_db('Pre_TeamData', 'Daily_Salaries', f"week={set_week} AND year={set_year}")
# dm.write_to_db(df_team, 'Pre_TeamData', 'Daily_Salaries', 'append')


# %%
to_upload = pd.DataFrame()
rank_c = 'expertKevinCole'
for pos in ['QB', 'RB', 'WR', 'TE', 'K']:
    fp = dm.read(f'''SELECT * 
                    FROM PFF_Expert_Ranks 
                    WHERE Position='{pos}' 
                            ''', 'Pre_PlayerData')
    fp.player = fp.player.apply(dc.name_clean)
    # if pos == 'DST': fp = fp.drop('player', axis=1).rename(columns={'team': 'player'})

    import datetime as dt
    def year_week_to_date(x):
        return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))

    fp = fp.sort_values(by=['year', 'week', rank_c]).reset_index(drop=True)
    fp['game_date'] = fp[['year', 'week']].apply(year_week_to_date, axis=1)


    def create_adj_ranks(df, rank_col):
        
        # get the mean rankings for past 6 weeks
        past_6_weeks = df.game_date.unique()[-6]
        mean_rank = df[df.game_date >= past_6_weeks].reset_index(drop=True)
        mean_rank = mean_rank.groupby('player').agg({rank_col: 'mean'}).reset_index().sort_values(by=rank_col)
        mean_rank[rank_col] = range(1, len(mean_rank)+1)

        # join in the current rankings and find missing players
        cur_rank = df[df.game_date == df.game_date.unique()[-1]].reset_index(drop=True)
        mean_rank = pd.merge(mean_rank, cur_rank[['player', 'game_date']], on=['player'], how='outer')
        mean_rank['to_add'] = np.where(mean_rank.game_date.isnull(), 1, 0)
        mean_rank['to_add'] = mean_rank.to_add.cumsum()
        
        # join current back to the mean and adjust the rankings for missing players
        cur_rank = pd.merge(cur_rank, mean_rank[[rank_col, 'to_add']], on=rank_col)
        cur_rank['rankadj_' + rank_col] = cur_rank[rank_col] + cur_rank.to_add
        cur_rank = cur_rank.drop(['game_date', 'to_add'], axis=1)

        # join current back to the mean and adjust the rankings for missing players
        cur_rank = pd.merge(cur_rank, mean_rank[['player', 'to_add']], on='player', how='left')
        cur_rank['playeradj_' + rank_col] = cur_rank[rank_col] + cur_rank.to_add
        cur_rank = cur_rank.drop(['to_add'], axis=1)

        return cur_rank

    with_adj = fp[fp.game_date <= fp.game_date.unique()[5]].copy().drop('game_date', axis=1)
    for i in fp.game_date.unique()[6:]:
        cur_df = create_adj_ranks(fp[fp.game_date <= i], rank_c)
        with_adj = pd.concat([with_adj, cur_df], axis=0)

    with_adj.loc[with_adj['rankadj_' + rank_c].isnull(), 'rankadj_' + rank_c] = with_adj.loc[with_adj['rankadj_' + rank_c].isnull(), rank_c]
    with_adj.loc[with_adj['playeradj_' + rank_c].isnull(), 'playeradj_' + rank_c] = with_adj.loc[with_adj['playeradj_'+ rank_c].isnull(), rank_c]

    with_adj = pd.merge(fp, with_adj[['player', 'offTeam', 'week', 'year', 
                             'rankadj_' + rank_c, 'playeradj_' + rank_c]],
                             on=['player', 'offTeam', 'week', 'year'], how='left')
    to_upload = pd.concat([to_upload, with_adj], axis=0)

#%%
xx = to_upload[['player', 'offTeam', 'week', 'year', 'rankadj_expertConsensus', 'playeradj_expertConsensus']].copy()

#%%
yy = to_upload[['player', 'offTeam', 'week', 'year', 'rankadj_expertNathanJahnke', 'playeradj_expertNathanJahnke']].copy()

#%%

to_upload = pd.merge(to_upload, xx,on=['player', 'offTeam', 'week', 'year'], how='left')
to_upload = pd.merge(to_upload, yy,on=['player', 'offTeam', 'week', 'year'], how='left')

# %%

dm.write_to_db(to_upload.drop_duplicates(), 'Pre_PlayerData', 'PFF_Expert_Ranks', 'replace', True)

# %%
all_data = dm.read("SELECT * FROM PFF_Expert_Ranks", 'Pre_PlayerData')
all_data.week= all_data.week.astype('str')
all_data.year = all_data.year.astype('str')
all_data['pkey'] = all_data.player + all_data.week + all_data.year

to_upload.week= to_upload.week.astype('str')
to_upload.year = to_upload.year.astype('str')
to_upload['pkey'] = to_upload.player + to_upload.week + to_upload.year
all_data[(~all_data.pkey.isin(to_upload.pkey)) ]#.groupby('player').agg({'Id': 'count'}).sort_values(by='Id', ascending=False)
# %%
