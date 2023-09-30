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
set_year = 2023
set_week = 4

from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)



def move_download_to_folder(root_path, folder, fname, week=''):
    try:
        os.replace(f"/Users/borys/Downloads/{fname}", 
                    f'{root_path}/Data/OtherData/{folder}/{set_year}/{week}{fname}')
    except:
        pass

    df = pd.read_csv(f'{root_path}/Data/OtherData/{folder}/{set_year}/{week}{fname}')
    
    return df


def format_ffa(df, table_name, set_week, set_year):
    df = df.dropna(subset=['player']).drop(['Unnamed: 0'], axis=1)
    df.player = df.player.apply(dc.name_clean)
    df.team = df.team.map(team_map)
    df.loc[df.position=='DST', 'player'] = df.loc[df.position=='DST', 'team']

    if table_name=='Projections': new_cols = ['player', 'position', 'team']
    elif table_name=='RawStats': new_cols = ['player', 'team', 'position', 'week']

    new_cols.extend(['ffa_' + c for c in df.columns if c not in ('player', 'position', 'team', 'week')])
    df.columns = new_cols

    df['week'] = set_week
    df['year'] = set_year
    return df

def format_fantasy_cruncher(df, set_week, set_year):
    new_cols = []
    col_prefix = ''
    for i in range(df.shape[1]):
        if 'Unnamed' not in df.columns[i]:
            col_prefix = df.columns[i].replace('.', '').replace(' ', '_').lower() + '_'
        col_suffix = str(df.iloc[0, i]).replace('.', '').replace('+', ' ').replace('%', 'pct').replace(' ', '_').lower()
        new_cols.append(col_prefix + col_suffix)

    df.columns = new_cols
    df = df.rename(columns={f'game_and_vegas_{set_year}_avg': 'game_and_vegas_this_year_avg',
                            f'game_and_vegas_{set_year-1}_avg': 'game_and_vegas_last_year_avg'})
    for c in ['position_rankings_rdm_pct', 'position_rankings_exp',	'position_rankings_used', 
              'projected_values_rdm_pct', 'projected_values_exp', 'projected_values_used', 'projected_values_con',
              'projected_values_value', 'position_rankings_salary', 'position_rankings_fc_proj', 'position_rankings_my_proj',
              'overall_rankings_salary', 'overall_rankings_fc_proj', 'overall_rankings_my_proj']:
        try: df = df.drop(c, axis=1)
        except: pass

    df = df.iloc[1:]
    df = df.drop(['likes', 'inj'], axis=1)
    new_cols = ['player', 'pos']
    new_cols.extend(['fc_' + c for c in df.columns if c not in ('player', 'pos')])
    df.columns = new_cols
    df = df.rename(columns={'fc_game_and_vegas_stdv': 'fc_game_and_vegas_stddev'})
    df['week'] = set_week
    df['year'] = set_year

    df.player = df.player.apply(dc.name_clean)
    df.loc[df.pos=='DST', 'player'] = df.loc[df.pos=='DST', 'player'].map(name_map)

    return df


def pull_fantasy_data(set_week):

    # move fantasydata projections
    df = move_download_to_folder(root_path, 'FantasyData', 'fantasy-football-weekly-projections.csv', week=set_week)
    df = df.drop('Week', axis=1).assign(week=set_week, year=set_year)
    

    cols = {
            'Rank': 'fdta_rank',
            'Name': 'player', 
            'Team': 'team', 
            'Position': 'position',
            'Opponent': 'opp',
            'PassingYards': 'fdta_pass_yds',
            'PassingTouchdowns': 'fdta_pass_td',
            'PassingInterceptions': 'fdta_pass_int',
            'RushingYards': 'fdta_rush_yds',
            'RushingTouchdowns': 'fdta_rush_td',
            'Receptions': 'fdta_rec',
            'ReceivingYards': 'fdta_rec_yds',
            'ReceivingTouchdowns': 'fdta_rec_td',
            'Sacks': 'fdta_sack',
            'Interceptions': 'fdta_int',
            'FumblesRecovered': 'fdta_fum_rec',
            'FumblesForced': 'fdta_fum_forced',
            'FantasyPointsPerGameDraftKings': 'fdta_dk_points_per_game',
            'FantasyPointsDraftKings': 'fdta_dk_points',
            }
    df = df[df.Position.isin(['QB', 'RB', 'WR', 'TE', 'DST'])].reset_index(drop=True)
    df = df.rename(columns=cols)

    df.player = df.player.apply(dc.name_clean)
    df.team = df.team.map(team_map)
    df.loc[df.position=='DST', 'player'] = df.loc[df.position=='DST', 'team']
    
    return df


#%%

#=============
# Fantasy Pros
#=============

for set_pos in ['QB', 'RB', 'WR', 'TE', 'DST']:

    try:
        os.replace(f"/Users/borys/Downloads/FantasyPros_{set_year}_Week_{set_week}_{set_pos}_Rankings.csv", 
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

def pff_rank(label_pre, label_post, folder):

    try:
        os.replace(f"/Users/borys/Downloads/{label_pre}.csv", 
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


rank1_pl, rank1_tm = pff_rank('week-rankings-export', 'expert_ranks', 'pff_rank')

rank1_pl = rank1_pl.drop(f'w{set_week}', axis=1)
rank1_tm = rank1_tm.drop(f'w{set_week}', axis=1)

rank1_pl.Name = rank1_pl.Name.apply(dc.name_clean)
rank1_pl = rank1_pl.rename(columns={'Name': 'player'})

all_df = pd.DataFrame()
for p in ['QB', 'RB', 'WR', 'TE']:
    base_df = rank1_pl[rank1_pl.Position==p].copy().reset_index(drop=True)

    for c in ['expertConsensus', 'expertNathanJahnke']:
        cur_df = create_adj_ranks(base_df, c, f"WHERE Position='{p}'", 'PFF_Expert_Ranks', dm)
        base_df = pd.merge(base_df, cur_df[['player', 'offTeam', f'rankadj_{c}', f'playeradj_{c}']], on=['player', 'offTeam'], how='left')
    all_df = pd.concat([all_df, base_df], axis=0)

rank1_pl = all_df.copy()

tables = ['Expert', 'Expert']
dfs = [rank1_pl, rank1_tm]
dbs = ['Pre_PlayerData', 'Pre_TeamData']

for t, d, db in zip(tables, dfs, dbs):
    dm.delete_from_db(db, f'PFF_{t}_Ranks', f"week={set_week} AND year={set_year}")
    dm.write_to_db(d, db, f'PFF_{t}_Ranks', 'append')

#%%

def pff_proj(label_pre, label_post, folder, rep=True):
    
    try:
        os.replace(f"/Users/borys/Downloads/{label_pre}.csv", 
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


proj_pl, proj_tm = pff_proj('projections', 'projections', 'pff_proj')
proj_pl.playerName = proj_pl.playerName.apply(dc.name_clean)
proj_pl = proj_pl.rename(columns={'playerName': 'player'})


tables = ['Proj', 'Proj']
dfs = [proj_pl, proj_tm]
dbs = ['Pre_PlayerData', 'Pre_TeamData']

for t, d, db in zip(tables, dfs, dbs):
    dm.delete_from_db(db, f'PFF_{t}_Ranks', f"week={set_week} AND year={set_year}")
    dm.write_to_db(d, db, f'PFF_{t}_Ranks', 'append')


#%%

df = move_download_to_folder(root_path, 'FFA', f'projections_{set_year}_wk{set_week}.csv')
df = format_ffa(df, 'Projections', set_week, set_year)

dm.delete_from_db('Pre_PlayerData', 'FFA_Projections', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'FFA_Projections', 'append')


df = move_download_to_folder(root_path, 'FFA', f'raw_stats_{set_year}_wk{set_week}.csv')
df = format_ffa(df, 'RawStats', set_week, set_year)

dm.delete_from_db('Pre_PlayerData', 'FFA_RawStats', f"week={set_week} AND year={set_year}", create_backup=False)
cols = dm.read("SELECT * FROM FFA_RawStats", 'Pre_PlayerData').columns
df = df[cols]
dm.write_to_db(df, 'Pre_PlayerData', 'FFA_RawStats', 'append')


#%%
# pull fftoday rankings
output = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:
    df = pull_fftoday(pos, set_week, set_year)
    output = pd.concat([output, df], axis=0, sort=False)
output = output.fillna(0)
output = dc.convert_to_float(output)
output.player = output.player.apply(dc.name_clean)
output.team = output.team.map(team_map)

dm.delete_from_db('Pre_PlayerData', 'FFToday_Projections', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(output, 'Pre_PlayerData', 'FFToday_Projections', 'append')

#%%

df = pull_fantasy_data(set_week)
dm.delete_from_db('Pre_PlayerData', 'FantasyData', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'FantasyData', 'append')

#%%
df = move_download_to_folder(root_path, 'FantasyCruncher', f'draftkings_NFL_{set_year}-week-{set_week}_players.csv')
df = format_fantasy_cruncher(df, set_week, set_year)
col = dm.read("SELECT * FROM FantasyCruncher", 'Pre_PlayerData').columns
df = df[[c for c in df.columns if c in col]]

dm.delete_from_db('Pre_PlayerData', 'FantasyCruncher', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'FantasyCruncher', 'append')


#%%
# ## PFF Matchups

def pff_matchups(label):

    try:

        os.replace(f"/Users/borys/Downloads/{label}_matchup_chart.csv", 
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
    df = df.rename(columns={'This Week': name, 'Last Week': 'last_week'})
    df[name] = df[name].apply(lambda x: x.replace('$', '').replace(',', '')).astype('int')
    df['last_week'] = df['last_week'].apply(lambda x: x.replace('$', '').replace(',', '').replace('-', '0')).astype('int')
    if name == 'fd_salary': 
        df.loc[(df.fd_salary < 100) | (df.fd_salary > 10000), 'fd_salary'] = df.loc[(df.fd_salary < 100) | (df.fd_salary > 10000), 'last_week']
        df.loc[(df.fd_salary < 100) | (df.fd_salary > 10000), 'fd_salary'] = np.nan
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

# stadiums = dm.read("SELECT * FROM Stadium_Locations", 'Pre_TeamData')
# location_keys = {}
# i = 0
# for _, row in stadiums.iterrows():
#     team = row.Team
#     latitude = row.Lat
#     longitude = row.Long


#     url = f'http://dataservice.accuweather.com/locations/v1/cities/geoposition/search?apikey={accu}&q={latitude}%2C%20{longitude}'
#     location_key = requests.get(url).json()
#     location_keys[team] = location_key['Key']

# df = pd.DataFrame(location_keys, index=[0]).T.reset_index()
# df.columns = ['team', 'location_key']
# df.team = df.team.map(full_team_map).fillna('KC')
# dm.write_to_db(df, 'Pre_TeamData', 'AccuWeather_Locations', 'replace')

#%%  


city_data  = dm.read(f'''SELECT a.home_team, a.gametime, a.gametime_unix, b.location_key
                        FROM Gambling_Lines a
                        JOIN (SELECT * FROM Accuweather_Locations) b
                              ON a.home_team = b.team
                        WHERE week={set_week} 
                              AND year={set_year}''', 'Pre_TeamData')

accu = '4B8LywkV5ABreulZ2f0DaXqGuuIhfJXI'
weather_list = [] 

for _, row in city_data.iterrows():
    
    city = row.home_team
    location_key = row.location_key
    gt = row.gametime_unix

    weather_url = f'http://dataservice.accuweather.com/forecasts/v1/daily/5day/{location_key}?apikey={accu}&details=true'
    weather = requests.get(weather_url).json()['DailyForecasts']
    
    min_time_diff = 100000000
    for d in weather:
        if abs(d['EpochDate'] - gt) < min_time_diff:
            if abs(d['Sun']['EpochRise'] - gt) < abs(d['Sun']['EpochSet'] - gt): day_or_night = 'Day'
            else: day_or_night = 'Night'

            best_t = d['EpochDate']
            temp_high = d['Temperature']['Maximum']['Value']
            temp_low = d['Temperature']['Minimum']['Value']
            
            rain_prob = d[day_or_night]['RainProbability']
            snow_prob = d[day_or_night]['SnowProbability']
            ice_prob = d[day_or_night]['IceProbability']
            precip_prob = (rain_prob + snow_prob + ice_prob) / 100
            precip_type = np.select([(rain_prob > snow_prob) & (rain_prob > ice_prob), 
                                    (snow_prob > rain_prob) & (snow_prob > ice_prob),
                                    (ice_prob > rain_prob) & (ice_prob > snow_prob)],
                                    ['rain', 'snow', 'ice'], None
                                    )
            precip_type = np.where(precip_prob < 0.1, None, precip_type)
            wind_speed = d[day_or_night]['Wind']['Speed']['Value']
            wind_gust = d[day_or_night]['WindGust']['Speed']['Value']
            uv_index = d['AirAndPollen'][-1]['Value']

            precip_intensity = d[day_or_night]['TotalLiquid']['Value'] / (d[day_or_night]['HoursOfPrecipitation']+0.1)
            min_time_diff = abs(d['EpochDate'] - gt)

            humidity = (0.5 + ((precip_prob - 0.5)/5))

    weather_list.append([city, best_t, precip_prob, precip_intensity, precip_type, temp_high,
                         temp_low, humidity, wind_speed, wind_gust, uv_index])

weather_df = pd.DataFrame(weather_list)
weather_df.columns = ['team', 'gametime_unix', 'precip_prob', 'precip_intensity', 'precip_type',
                   'temp_high', 'temp_low', 'humidity', 'wind_speed', 'wind_gust', 'uv_index']
weather_df['year'] = set_year
weather_df['week'] = set_week
weather_df

#%%
dm.delete_from_db('Pre_TeamData', 'Game_Weather', f"week={set_week} and year={set_year}")
dm.write_to_db(weather_df, 'Pre_TeamData', 'Game_Weather', if_exist='append')

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
    su.copyfile(f'c:/Users/borys/Downloads/DKSalaries.csv', 
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
salary = salary.assign(year=set_year).assign(week=set_week)

ids = salary_id[['player', 'player_id']]
ids = ids.assign(year=set_year).assign(week=set_week)


dm.delete_from_db('Simulation', 'Salaries', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(salary, 'Simulation', 'Salaries', 'append')

dm.delete_from_db('Simulation', 'Player_Ids', f"week={set_week} AND year={set_year}")
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


#%%
