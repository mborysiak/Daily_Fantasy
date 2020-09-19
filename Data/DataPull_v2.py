# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
set_year = 2020
set_week = 2

root_path = '/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/'
# -

import pandas as pd
import os
import numpy as np
import requests 
import datetime as dt
import sqlite3
from DataPull_Helper import *

# # Fantasy Pros

# ## Rankings

qb_url = 'https://www.fantasypros.com/nfl/rankings/qb.php'
rb_url = 'https://www.fantasypros.com/nfl/rankings/half-point-ppr-rb.php'
wr_url = 'https://www.fantasypros.com/nfl/rankings/half-point-ppr-wr.php'
te_url = 'https://www.fantasypros.com/nfl/rankings/half-point-ppr-te.php'
d_url = 'https://www.fantasypros.com/nfl/rankings/dst.php'
k_url = 'https://www.fantasypros.com/nfl/rankings/k.php'

# +
positions = ['QB', 'RB', 'WR', 'TE', 'K']

data = pd.DataFrame()
for i, _url in enumerate([qb_url, rb_url, wr_url, te_url, k_url]):
    df = pd.read_html(_url)[0]
    df = df.dropna(axis=1, thresh=10).dropna(axis=0, how='all')
    df = df.drop('Proj. Pts', axis=1)
    df.columns = ['Rank', 'Player_Team', 'Opponent', 'Best', 'Worst', 'Avg', 'Std Dev']
    df['player'] = df.Player_Team.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1][:-2])
    df['team'] = df.Player_Team.apply(lambda x: x.split(' ')[-1])
    df['position'] = positions[i]
    df['year'] = set_year
    df['week'] = set_week
    cols = ['player', 'position', 'team', 'year', 'week', 'Rank', 'Opponent', 'Best', 'Worst', 'Avg', 'Std Dev']
    df = df[cols].rename(columns={'Std Dev': 'StdDev'})
    data = pd.concat([data, df], axis=0)
    
df = pd.read_html(d_url)[0]
df = df.dropna(axis=1, thresh=10).dropna(axis=0, how='all')
df = df.drop('Proj. Pts', axis=1)
df.columns = ['Rank', 'Player_Team', 'Opponent', 'Best', 'Worst', 'Avg', 'Std Dev']
df['player'] = df.Player_Team.apply(lambda x: x.split('(')[0].replace(' ', ''))
df['team'] = df.Player_Team.apply(lambda x: x.split(')')[1].split(' ')[0])
df['position'] = 'DST'
df['year'] = set_year
df['week'] = set_week
cols = ['player', 'position', 'team', 'year', 'week', 'Rank', 'Opponent', 'Best', 'Worst', 'Avg', 'Std Dev']
df = df[cols].rename(columns={'Std Dev': 'StdDev'})
data = pd.concat([data, df], axis=0)
    
data = data[~data.player.str.contains('google')]
data = data.dropna(axis=0)

data['HomeTeam'] = data.Opponent.apply(lambda x: np.where(x.split(' ')[0] == 'at', 0, 1))
data['Opponent'] = data.Opponent.apply(lambda x: x.split(' ')[1])

data = data.reset_index(drop=True)
# -

append_to_db(data, table_name='FantasyPros_Rankings', if_exist='append')

# ## DK Salary

dk_sal = pd.read_html('https://www.fantasypros.com/daily-fantasy/nfl/draftkings-salary-changes.php')[0]
fd_sal = pd.read_html('https://www.fantasypros.com/daily-fantasy/nfl/fanduel-salary-changes.php')[0]
yahoo_sal = pd.read_html('https://www.fantasypros.com/daily-fantasy/nfl/yahoo-salary-changes.php')[0]


# +
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
# -

append_to_db(salaries, table_name='Daily_Salaries', if_exist='append')

# # NumberFire

# +
df = pd.read_csv(f'/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/number_fire/{set_year}/nf_week{set_week}.csv', header=None)
data = []
for i in range(0, df.shape[0]-4, 4):
    df_tmp = df.iloc[i:i+4, :].reset_index(drop=True)
    names = list(df_tmp.iloc[0, :2].values)
    names.append(df_tmp.iloc[1, 0])
    names.extend(list(df_tmp.iloc[3].values))
    data.append(names)
    
nf = pd.DataFrame(data)
nf.columns = ['player', 'position', 'team_info', 'nf_fantasy_pts', 'fd_cost', 'nf_value', 'nf_completion_att', 
              'nf_pass_yds', 'nf_pass_tds', 'nf_int', 'nf_rush_att', 'nf_rush_yds', 'nf_rush_tds', 
              'nf_rec', 'nf_rec_yds', 'nf_rec_tds', 'targets']
def get_inj(x):
    try:
        return x.split(' ')[3]
    except:
        return 'NotListed'
nf['injury'] = nf['team_info'].apply(get_inj)
nf['nf_completion'] = nf.nf_completion_att.apply(lambda x: x.split('/')[0]).astype('float')
nf['nf_pass_att'] = nf.nf_completion_att.apply(lambda x: x.split('/')[1]).astype('float')

nf['week'] = set_week
nf['year'] = set_year

nf = nf[['player', 'position', 'year', 'week', 'injury', 'nf_fantasy_pts', 
        'nf_value', 'nf_completion', 'nf_pass_att', 'nf_pass_yds', 'nf_pass_tds', 'nf_int', 'nf_rush_att', 
        'nf_rush_yds', 'nf_rush_tds', 'nf_rec', 'nf_rec_yds', 'nf_rec_tds', 'targets']]
# -

append_to_db(nf, table_name='NumberFire_Rankings', if_exist='append')

# # Betting Lines

# +
source = requests.get("https://www.bovada.lv/services/sports/event/v2/events/A/description/football/nfl").json()

data_list = []

for i in range(0, 15):
    
    away_team = source[0]['events'][i]['description'].split('@')[0].strip(' ')
    home_team = source[0]['events'][i]['description'].split('@')[1].strip(' ')
    try:
        away_moneyline = source[0]['events'][i]['displayGroups'][0]['markets'][2]['outcomes'][0]['price']['american']
        home_moneyline = source[0]['events'][i]['displayGroups'][0]['markets'][2]['outcomes'][1]['price']['american']
    except:
        away_moneyline = None
        home_moneyline = None
        
    try:
        over_under = source[0]['events'][i]['displayGroups'][0]['markets'][1]['outcomes'][0]['price']['handicap']
        over_under_odds = source[0]['events'][i]['displayGroups'][0]['markets'][1]['outcomes'][0]['price']['american']
    except:
        over_under = None
        over_under_odds = None
        
    try:
        away_line = source[0]['events'][i]['displayGroups'][0]['markets'][0]['outcomes'][0]['price']['handicap']
        home_line = source[0]['events'][i]['displayGroups'][0]['markets'][0]['outcomes'][1]['price']['handicap']
    except:
        away_line = None
        home_line = None
        
    gametime = dt.datetime.utcfromtimestamp(int(source[0]['events'][i]['startTime']) / 1000).strftime('%Y-%m-%d %H:%M:%S:%ms')
    gametime_unix = int(source[0]['events'][i]['startTime']) / 1000

    data_exp = [away_team, home_team, away_line, home_line, away_moneyline, home_moneyline, 
                over_under, over_under_odds, gametime, gametime_unix]
    data_list.append(data_exp)
    
    
data = pd.DataFrame(data_list, columns=['away_team', 'home_team', 'away_line', 'home_line',
                                        'away_moneyline', 'home_moneyline',
                                        'over_under', 'over_under_odds', 'gametime', 'gametime_unix'])

def clean_ml(x):
    if x == 'EVEN':
        return -110
    if x is not None:
        return float(x.replace('+', ''))
    else:
        return x
data.away_moneyline = data.away_moneyline.apply(clean_ml)
data.home_moneyline = data.home_moneyline.apply(clean_ml)

data.away_team = data.away_team.apply(lambda x: ' '.join([w.capitalize() for w in x.split(' ')]))
data.home_team = data.home_team.apply(lambda x: ' '.join([w.capitalize() for w in x.split(' ')]))

data.away_team = data.away_team.map(name_map)
data.home_team = data.home_team.map(name_map)
data['year'] = set_year
data['week'] = set_week
# -

data

data = pd.concat([data, pd.DataFrame({'away_team': ['CIN'], 'home_team': ['CLE']})], axis=0, sort=False).reset_index(drop=True)

append_to_db(data, table_name='Gambling_Lines', if_exist='append')

# # Weather

conn = sqlite3.connect('/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/Weekly_Stats.sqlite3')
cities = pd.read_sql_query('select * from city_lat_long', conn)
fp_rank  = pd.read_sql_query('select * from fantasypros_rankings where week={} and year={}'.format(set_week, 
                                                                                                   set_year), conn)
city_week = list(fp_rank[fp_rank.HomeTeam == 1].team.unique())

# +
KEY = '6c500b03257f351161dbe1fea1aa6558'

weather = []
for i, city in enumerate(city_week):
    print(city)
    try:
        gt = data[data.home_team == city].gametime_unix.values[0]
        LATITUDE = cities[cities.team == city].latitude.values[0]
        LONGITUDE = cities[cities.team == city].longitude.values[0]

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
    except:
        print('failed')
# -

weather = pd.DataFrame(weather)
weather.columns = ['team', 'gametime_unix', 'precip_prob', 'precip_intensity', 'precip_type',
                   'temp_high', 'temp_low', 'humidity', 'wind_speed', 'wind_gust', 'uv_index']
weather['year'] = set_year
weather['week'] = set_week

append_to_db(weather, table_name='GameWeather', if_exist='append')

# # PFR Position Matchups

rb_match = 'https://www.pro-football-reference.com/fantasy/RB-fantasy-matchups.htm'
wr_match = 'https://www.pro-football-reference.com/fantasy/WR-fantasy-matchups.htm'
te_match = 'https://www.pro-football-reference.com/fantasy/TE-fantasy-matchups.htm'
qb_match = 'https://www.pro-football-reference.com/fantasy/QB-fantasy-matchups.htm'

# +
rb_cols = ['player', 'team', 'games', 'games_started', 'snap_pct', 'rush_att_per_game', 'rush_yds_per_game', 
        'rush_td_per_game', 'tgt_per_game', 'rec_per_game', 'rec_yds_per_game', 'rec_td_per_game',
        'fp_per_game', 'dk_pts_per_game', 'fd_pts_per_game', 'opponent','opp_rank', 'opp_fp_per_game', 
           'opp_dk_pt_per_game', 'opp_fd_pt_per_game', 'proj_fp_rank', 'proj_dk_rank', 'proj_fd_rank']

wr_cols = ['player', 'team', 'games', 'games_started', 'snap_pct',  'tgt_per_game', 'rec_per_game', 
           'rec_yds_per_game', 'rec_td_per_game', 'fp_per_game', 'dk_pts_per_game', 'fd_pts_per_game', 
           'opponent', 'opp_rank', 'opp_fp_per_game', 'opp_dk_pt_per_game', 'opp_fd_pt_per_game', 'proj_fp_rank', 
           'proj_dk_rank', 'proj_fd_rank']

qb_cols = ['player', 'team', 'games', 'games_started', 'snap_pct', 'pass_complete_per_game', 'pass_att_per_game', 
           'pass_yds_per_game', 'pass_td_per_game', 'int_per_game', 'sack_per_game', 'rush_att_per_game',
           'rush_yds_per_game', 'rush_td_per_game', 'fp_per_game', 'dk_pts_per_game', 'fd_pts_per_game', 
           'opponent', 'opponent_rank', 'opp_fp_per_game', 'opp_dk_pt_per_game', 'opp_fd_pt_per_game', 
           'proj_fp_rank', 'proj_dk_rank', 'proj_fd_rank']
# -

rb = pfr_matchup_cleanup(pd.read_html(rb_match)[0], rb_cols)
wr = pfr_matchup_cleanup(pd.read_html(wr_match)[0], wr_cols)
te = pfr_matchup_cleanup(pd.read_html(te_match)[0], wr_cols)
qb = pfr_matchup_cleanup(pd.read_html(qb_match)[0], qb_cols)

# +
ra = 'append'

append_to_db(rb, table_name='RB_PRF_Matchups', if_exist=ra)
append_to_db(wr, table_name='WR_PRF_Matchups', if_exist=ra)
append_to_db(te, table_name='TE_PRF_Matchups', if_exist=ra)
append_to_db(qb, table_name='QB_PRF_Matchups', if_exist=ra)
# -

# # Fantasy Pros Position Matchups

for p in ['qb', 'rb', 'wr', 'te']:
    
    df = pd.read_html('https://www.fantasypros.com/nfl/matchups/{}.php?show=points'.format(p))[0]
    df = df[['Player', str(set_week)]]
    df = df.dropna(axis=0)
    df.columns = ['player', 'opp_pts_per_game']
    df.player = df.player.apply(lambda x: ' '.join(x.split(' ')[:-1]))
    df.opp_pts_per_game = df.opp_pts_per_game.apply(lambda x: x.split(' ')[4])
    df['week'] = set_week
    df['year'] = set_year
    append_to_db(df, table_name='{}_FantasyPros_Matchups'.format(p.upper()), if_exist='append')

# # PFF Matchups

# ## WR and CB

pff_fp_map = {'ARZ': 'ARI',
             'ATL': 'ATL',
             'BLT': 'BAL',
             'BUF': 'BUF',
             'CAR': 'CAR',
             'CHI': 'CHI',
             'CIN': 'CIN',
             'CLV': 'CLE',
             'DAL': 'DAL',
             'DEN': 'DEN',
             'DET': 'DET',
             'GB': 'GB',
             'HST': 'HOU',
             'IND': 'IND',
             'JAX': 'JAC',
             'KC': 'KC',
             'LA': 'LAR',
             'LAC': 'LAC',
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
             'WAS': 'WAS',
             'WFT': 'WAS'}


def fix_pct(col):
    
    col = float(str(col).replace('%', ''))
    
    if col > 1000:
        return col / 100
    
    else:
        return col


os.replace("/Users/Mark/Downloads/wr_cb_matchup_chart.csv", 
           f'{root_path}/pff_matchups/pff_wr_cb/{set_year}/wr_cb_week{set_week}.csv')

# +
wr_cb = pd.read_csv(f'{root_path}/pff_matchups/pff_wr_cb/{set_year}/wr_cb_week{set_week}.csv')

to_drop = ['offHeightInches', 'offWeight', 'offSpeed', 'defHeightInches', 'defWeight', 'defSpeed']
wr_cb = wr_cb.drop(to_drop, axis=1)

cols = ['team', 'player', 'wr_side', 'routes', 'wr_left_pct', 'wr_slot_pct', 'wr_right_pct', 'tgt_per_route', 
        'fp_per_route', 'catch_pct', 'yds_per_route_run', 'pff_rank_wr', 'pff_advantage', 
        'opponent', 'primary_defender', 'cb_side', 'routes_against', 'cb_left_pct', 'cb_slot_pct',
        'cb_right_pct', 'tgt_per_route_against', 'fp_per_route_against', 'completion_pct_against',
        'yds_per_route_covered', 'pff_rank_cb']
wr_cb.columns = cols

wr_cb.player = wr_cb.player.apply(lambda x: ' '.join([c.capitalize() for c in x.split(' ')]))
wr_cb.player = wr_cb.player.apply(lambda x: ' '.join(x.split()[:2]))
wr_cb.player = wr_cb.player.apply(lambda x: '.'.join([c.capitalize() for c in x.split(' ')[0].split('.')]) + ' ' + x.split(' ')[1].capitalize())

wr_cb.loc[wr_cb.player == 'Marquez Valdes-scantling', 'player'] = 'Marquez Valdes-Scantling'
wr_cb.loc[wr_cb.player == 'Juju Smith-schuster', 'player'] = 'JuJu Smith-Schuster'
wr_cb = wr_cb.fillna(0)

wr_cb.team = wr_cb.team.map(pff_fp_map)
wr_cb['week'] = set_week
wr_cb['year'] = set_year
# -

append_to_db(wr_cb, table_name='PFF_WR_CB_Matchups', if_exist='append')

# # TE PFF

# +
df = pd.read_csv(f'/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/pff_matchups/pff_te/{set_year}/te_week{set_week}.csv',
                 header=None)
idx = []
for i, row in df.iterrows():
    if len(row.values[0]) > 7:
        idx.append(i)
idx.append(idx[-1]+14)

data = []
i = 0
for j in idx[1:]:
    
    val = list(pd.concat([df.iloc[i:j-1].T], axis=1).values[0])
# # Use this if there are null values in the routes run column
#     if val[3][-1] == '%':
#         beg = val[:3]
#         beg.append(None)
#         beg.extend(val[3:])
#         val = beg
        
    data.append(val)
    i = j-1

df = pd.DataFrame(data)
wr = []
cb = []
for i, row in df.iterrows():
    if i % 2 == 0:
        wr.append(list(row.values))
    else:
        cb.append(list(row.values))
        
df = pd.concat([pd.DataFrame(wr), pd.DataFrame(cb)], axis=1)
df = df.dropna(axis=1, thresh=20)

df.columns = range(0, 26)
df = df.drop([2, 3, 18 , 19], axis=1)

cols = ['team', 'player', 'routes', 'block_pct', 'inside_pct', 'slot_pct', 'wide_pct', 'tgt_per_route', 
        'fp_per_route', 'catch_pct', 'yds_per_route_run', 'pff_rank_te', 'pff_advantage', 
        'opponent', 'primary_defender', 'def_pos', 'routes_against', 'tgt_per_route_against', 'fp_per_route_against',
        'catch_pct_allowed', 'yds_per_route_covered', 'pff_rank_cb']
df.columns = cols

df.player = df.player.apply(lambda x: ' '.join([c.capitalize() for c in x.split(' ')]))
df.player = df.player.apply(lambda x: ' '.join(x.split()[:2]))
df.player = df.player.apply(lambda x: '.'.join([c.capitalize() for c in x.split(' ')[0].split('.')]) + ' ' + x.split(' ')[1].capitalize())
df = df.fillna(0)
    
df.team = df.team.map(pff_fp_map)
df.routes = df.routes.apply(lambda x: x.replace('-', '0'))
df.routes = df.routes.apply(fix_pct)
df.routes_against = df.routes_against.apply(lambda x: str(x).replace('-', '0'))
df.routes_against = df.routes_against.apply(fix_pct)
df.pff_rank_cb = df.pff_rank_cb.apply(fix_pct)
df.pff_rank_te = df.pff_rank_te.apply(fix_pct)

for c in df.columns:
    try:
        df[c] = df[c].apply(lambda x: float(str(x).replace('%', '')))
    except:
        pass

df['week'] = set_week
df['year'] = set_year
# -

append_to_db(df, table_name='PFF_TE_Matchups', if_exist='append')

# ## Oline and Dline

# +
df = pd.read_csv(f'/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/pff_matchups/pff_oline_dline/{set_year}/oline_dline_week{set_week}.csv',
                 header=None)

idx = []
for i, row in df.iterrows():
    if len(row.values[0]) > 5:
        idx.append(i)
idx.append(idx[-1]+15)

data = []
i = 0
for j in idx[1:]:
    
    val = list(pd.concat([df.iloc[i:j].T], axis=1).values[0])
    data.append(val)
    i = j
    
df = pd.DataFrame(data)
df.columns = ['gametime', 'team', 'pressure_rate_allow', 'sack_conversion_allow', 'yds_before_contact_generate',
              'runs_inside_5_generate', 'td_pct_inside_5_generate', 'pass_block_advantage', 'run_block_advantage', 
              'opponent', 'pressure_rate_generate', 'sack_conversion_generate', 'yds_before_contact_allow',
              'runs_inside_5_allow', 'td_pct_inside_5_allow']

df.team = df.team.map(pff_fp_map)
df.opponent = df.opponent.map(pff_fp_map)

for c in df.columns:
    try:
        df[c] = df[c].apply(lambda x: x.replace('%', '')).astype('float')
    except:
        pass
    
df['week'] = set_week
df['year'] = set_year
# -

append_to_db(df, table_name='PFF_Oline_Dline_Matchups', if_exist='append')

# # Rotogrinders (deprecated because subscription based 9/12/2020)

pd.read_csv('https://rotogrinders.com/projected-stats/nfl-flex.csv?site=draftkings', header=None)

flex = pd.read_csv('https://rotogrinders.com/projected-stats/nfl-flex.csv?site=draftkings', header=None)
qb = pd.read_csv('https://rotogrinders.com/projected-stats/nfl-qb.csv?site=draftkings', header=None)
defense = pd.read_csv('https://rotogrinders.com/projected-stats/nfl-defense.csv?site=draftkings', header=None)
kicker = pd.read_csv('https://rotogrinders.com/projected-stats/nfl-kicker.csv?site=draftkings', header=None)
kicker_fd = pd.read_csv('https://rotogrinders.com/projected-stats/nfl-kicker.csv?site=fanduel', header=None)

# +
roto_grind = pd.DataFrame()
for d in [qb, flex, defense, kicker]:
    roto_grind = pd.concat([roto_grind, d], axis=0)
    
roto_grind['website_sal'] = 'draftkings'
kicker_fd['website_sal'] = 'fanduel'
roto_grind = pd.concat([roto_grind, kicker_fd], axis=0)

cols = ['player', 'salary', 'team', 'position', 'opponent', 'roto_grind_ceiling', 
           'roto_grind_floor', 'roto_grind_proj', 'website_sal']
roto_grind.columns = cols
roto_grind = roto_grind.reset_index(drop=True)
roto_grind['week'] = set_week
roto_grind['year'] = set_year
# -

append_to_db(roto_grind, table_name='Roto_Grinders_Projections', if_exist='append')

# # Copy Database Backup for Week

copy_db(set_week, set_year, pre_post='pre')

# # #============
# # #    Post Game       
# # #============

# # Air Yards

import requests
df = requests.get('http://api.airyards.com/2020/weeks')

df

# + language="R"
# library(data.table)
# library(jsonlite)
# df_air <- data.table(fromJSON('http://airyards.com/2020/weeks'))
# fwrite(df_air, '/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/air_yards/ay_2019.csv')
# -

ay_map = {'ARI': 'ARI',
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
         'JAC': 'JAC',
         'KC': 'KC',
         'LA': 'LAR',
         'LAC': 'LAC',
         'MIA': 'MIA',
         'MIN': 'MIN',
         'NE': 'NE',
         'NO': 'NO',
         'NYG': 'NYG',
         'NYJ': 'NYJ',
         'OAK': 'OAK',
         'LVR': 'LVR'
         'PHI': 'PHI',
         'PIT': 'PIT',
         'SEA': 'SEA',
         'SF': 'SF',
         'TB': 'TB',
         'TEN': 'TEN',
         'WAS': 'WAS'
         'WFT': 'WAS'}

df = pd.read_csv('/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/air_yards/ay_2019.csv')
df = df.drop(['index', 'player_id'], axis=1)
cols = ['player', 'position', 'team', 'week', 'tar', 'rec_td', 'rush_td', 'rec', 
        'rec_yds', 'rush_yds', 'yac', 'air_yds', 'tm_pass_att', 'team_air_yds', 
        'aypt', 'racr', 'ms_air_yards', 'target_share', 'wopr']
df.columns = cols
df = df[df.position.isin(['WR', 'TE', 'RB'])].reset_index(drop=True)
df = df[df.week == set_week]
df.team = df.team.map(ay_map)
df['year'] = set_year

append_to_db(df, table_name='Air_Yards', if_exist='append')

# # Basic Stats

qb_weekly = pd.read_html('https://www.fantasypros.com/nfl/stats/qb.php?league=1922936&week={}&range=week&scoring=HALF'.format(set_week))[0]
rb_weekly = pd.read_html('https://www.fantasypros.com/nfl/stats/rb.php?league=1922936&week={}&scoring=HALF&range=week'.format(set_week))[0]
wr_weekly = pd.read_html('https://www.fantasypros.com/nfl/stats/wr.php?league=1922936&week={}&scoring=HALF&range=week'.format(set_week))[0]
te_weekly = pd.read_html('https://www.fantasypros.com/nfl/stats/te.php?league=1922936&week={}&scoring=HALF&range=week'.format(set_week))[0]
def_weekly = pd.read_html('https://www.fantasypros.com/nfl/stats/dst.php?league=1922936&week={}&scoring=HALF&range=week'.format(set_week))[0]
k_weekly = pd.read_html('https://www.fantasypros.com/nfl/stats/k.php?league=1922936&week={}&scoring=HALF&range=week'.format(set_week))[0]

# +
rb_cols = ['rank','player', 'rush_att', 'rush_yds', 'yards_per_att', 'long_rush', 'plus_twenty_runs', 'rush_td', 
           'rec', 'tgts', 'rec_yds', 'yards_per_rec', 'rec_td', 'fumbles_lost', 'games', 'fantasy_pts', 
           'fantasy_ppg', 'own_pct']

qb_cols = ['rank','player', 'pass_complete', 'pass_att', 'pct_complete', 'pass_yds', 'pass_yd_per_att', 'pass_td', 'pass_int',
        'sacks', 'rush_att', 'rush_yds', 'rush_td',  'fumbles_lost','games', 'fantasy_pts', 'fantasy_ppg', 'own_pct']

wr_cols = ['rank','player', 'rec', 'tgts', 'rec_yds', 'yards_per_rec', 'long_rec', 'plus_twenty_rec', 'rec_td',
           'rush_att', 'rush_yds', 'rush_td', 'fumbles_lost', 'games', 'fantasy_pts', 'fantasy_ppg',
           'own_pct']

def_cols = ['rank', 'player', 'sacks', 'int', 'fumble_rec', 'forced_fumbles', 'def_td', 'safety', 
            'special_teams_td', 'games', 'fantasy_pts', 'fantasy_ppg', 'own_pct']

k_cols = ['rank', 'player', 'fg_made', 'fg_att', 'pct_made', 'long', 'one_nineteen', 'twenty_twentynine', 
          'thirty_thirtynine', 'forty_fortynine', 'fifty_plus', 'extra_pt_made', 'extra_pt_att',
          'games', 'fantasy_pts', 'fanatsy_ppg', 'own_pct']

qb_weekly.columns = qb_cols
rb_weekly.columns = rb_cols
wr_weekly.columns = wr_cols
te_weekly.columns = wr_cols
def_weekly.columns = def_cols
k_weekly.columns = k_cols

for df in ['qb_weekly', 'rb_weekly', 'wr_weekly', 'te_weekly', 'k_weekly']:
    globals()[df] = globals()[df].drop('rank', axis=1)
    globals()[df]['team'] = globals()[df].player.apply(lambda x: x.split('(')[-1].replace(')', ''))
    globals()[df]['player'] = globals()[df].player.apply(lambda x: ' '.join(x.split('(')[:-1]).rstrip(' '))
    globals()[df]['week'] = set_week
    globals()[df]['year'] = set_year
    globals()[df].own_pct = globals()[df].own_pct.fillna('0').apply(lambda x: float(x.replace('%', '')))
    
def_weekly.columns = def_cols
def_weekly = def_weekly.drop('rank', axis=1)
def_weekly['player'] = def_weekly.player.apply(lambda x: x.split('(')[-1].replace(')', ''))
def_weekly = def_weekly.rename(columns={'player': 'team'})
def_weekly['week'] = set_week
def_weekly['year'] = set_year
def_weekly.own_pct = def_weekly.own_pct.fillna('0').apply(lambda x: float(x.replace('%', '')))
# -

a_or_r = 'append'
append_to_db(qb_weekly, table_name='QB_Basic_Stat_Results', if_exist=a_or_r)
append_to_db(rb_weekly, table_name='RB_Basic_Stat_Results', if_exist=a_or_r)
append_to_db(wr_weekly, table_name='WR_Basic_Stat_Results', if_exist=a_or_r)
append_to_db(te_weekly, table_name='TE_Basic_Stat_Results', if_exist=a_or_r)
append_to_db(def_weekly, table_name='DEF_Basic_Stat_Results', if_exist=a_or_r)
append_to_db(k_weekly, table_name='K_Basic_Stat_Results', if_exist=a_or_r)

# # Red Zone Targets

# ## NFL Savant

rz_tgt = pd.read_html('http://www.nflsavant.com/targets.php?rz=redzone&ddlYear=' + \
                      '{}&week={}&ddlTeam=&ddlPosition='.format(set_year, set_week))[0]
rz_tgt.columns = ['rank', 'player', 'team', 'position', 'rz_complete', 'rz_tgts', 'complete_pct', 'rec_tds']
rz_tgt['week'] = set_week
rz_tgt['year'] = set_year
# rz_tgt['player'] = rz_tgt.player.apply(lambda x: x.split(',')[0] + ' ' + x.split(',')[1])

append_to_db(rz_tgt, table_name='RedZone_Targets', if_exist='append')

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

rush_rz_df.team = rush_rz_df.team.map(pfr_fp_map)
rec_rz_df.team = rec_rz_df.team.map(pfr_fp_map)
pass_rz_df.team = pass_rz_df.team.map(pfr_fp_map)

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
# -

r_or_a = 'append'
append_to_db(rush_rz_df, table_name='PFR_Redzone_Rush', if_exist=r_or_a)
append_to_db(rec_rz_df, table_name='PFR_Redzone_Rec', if_exist=r_or_a)
append_to_db(pass_rz_df, table_name='PFR_Redzone_Pass', if_exist=r_or_a)

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

def_vs_rb = pd.read_html('https://www.pro-football-reference.com/years/{}/fantasy-points-against-RB.htm'.format(set_year))
def_vs_te = pd.read_html('https://www.pro-football-reference.com/years/{}/fantasy-points-against-TE.htm'.format(set_year))
def_vs_qb = pd.read_html('https://www.pro-football-reference.com/years/{}/fantasy-points-against-QB.htm'.format(set_year))
def_vs_wr = pd.read_html('https://www.pro-football-reference.com/years/{}/fantasy-points-against-WR.htm'.format(set_year))

# +
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
# -

r_or_a = 'append'
append_to_db(def_vs_rb_df, table_name='Def_Allowed_RB', if_exist=r_or_a)
append_to_db(def_vs_wr_df, table_name='Def_Allowed_WR', if_exist=r_or_a)
append_to_db(def_vs_te_df, table_name='Def_Allowed_TE', if_exist=r_or_a)
append_to_db(def_vs_qb_df, table_name='Def_Allowed_QB', if_exist=r_or_a)

# # Advanced PFR Stats

# +
#qb_adv = pd.read_csv('/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/pfr_adv_stats/qb_adv_week{}.csv'.format(set_week))

# qb_adv_cols = ['rank', 'player', 'team', 'age', 'position', 'games', 'games_started', 'completions', 'pass_att',
#                'pass_yds', 'first_downs', 'air_yds', 'air_per_complete', 'air_per_att', 'adot', 'yac',
#                'yac_complete', 'drops', 'drop_pct', 'bad_throws', 'bad_throw_pct', 'blitzed', 'hurried',
#                'hits', 'scrambles', 'yds_per_scramble']

# qb_adv.columns = qb_adv_cols
# qb_adv.position = qb_adv.position.fillna('QB')
# qb_adv = qb_adv.fillna(0)
# -

rec.columns

# +
qb = pd.read_csv(f'/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/pfr_adv_stats/{set_year}/qb_week{set_week}.csv')
qb = qb.drop('1D', axis=1)
rb = pd.read_csv(f'/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/pfr_adv_stats/{set_year}/rb_week{set_week}.csv')
rec = pd.read_csv(f'/Users/Mark/Documents/GitHub/Daily_Fantasy/Data/pfr_adv_stats/{set_year}/wr_week{set_week}.csv')

qb_cols = ['rank', 'player', 'team',' age', 'position', 'games', 'games_started', 'qb_rec', 'completions',
           'pass_att', 'complete_pct', 'pass_yds', 'pass_td', 'pass_td_pct', 'ints', 'int_pct', 'long_pass',
           'yds_per_att', 'adj_yd_att', 'yds_per_complete', 'yds_per_game', 'qb_rating', 'qbr', 'sacks',
           'sack_yds', 'net_yds_att', 'adj_net_yds_att', 'sack_pct', 'fourth_qtr_comeback', 'game_winning_drives']
qb.columns = qb_cols
qb.qb_rec = qb.qb_rec.fillna('0-0-0')
qb.position = qb.position.fillna('QB')

qb.qb_rec = qb.qb_rec.apply(lambda x: x.replace('/', '-'))
qb['wins'] = qb.qb_rec.apply(lambda x: float(x.split('-')[0]))
qb['losses'] = qb.qb_rec.apply(lambda x: float(x.split('-')[1]))
qb['ties'] = qb.qb_rec.apply(lambda x: float(x.split('-')[2]))

qb = qb.drop('qb_rec', axis=1)
qb = qb.fillna(0)

rb_cols = ['rank', 'player', 'team', 'age', 'position', 'games', 'games_started', 'rush_att', 'rush_yds',
           'first_downs', 'yds_before_contact', 'yds_before_contact_att', 'yds_after_contact', 'yac_att', 
           'broke_tackles', 'att_broken']
rb.columns = rb_cols
rb.position = rb.position.fillna('RB')
rb = rb.fillna(0)

['Rk', 'Player', 'Tm', 'Age', 'Pos', 'G', 'GS', 'Tgt', 'Rec', 'Yds',
       'TD', '1D', 'YBC', 'YBC/R', 'YAC', 'YAC/R', 'ADOT', 'BrkTkl', 'Rec/Br',
       'Drop', 'Drop%', 'Int', 'Rat']

rec_cols = ['rank', 'player', 'team', 'age', 'position', 'games', 'games_started', 'targets', 'rec', 
            'rec_yds', 'rec_td', 'first_downs', 'yds_before_catch', 'yds_before_catch_per_rec', 'yards_after_catch', 
            'yards_after_catch_per_rec', 'adot', 'broken_tackles', 'rec_per_broken', 'drops', 'drop_pct',
            'targ_int', 'targ_rating']

rec.columns = rec_cols

# drop columns that were added in 2020 (td, adot, int, rating)
rec = rec.drop(['position', 'rec_td', 'adot', 'targ_int', 'targ_rating'], axis=1).fillna(0)


for df in [qb, rb, rec]:#, qb_adv]:
    df['week'] = set_week
    df['year'] = set_year
    df.team = df.team.map(pfr_fp_map)
    
rb = rb.dropna(axis=0)
# -

r_or_a = 'append'
append_to_db(qb, table_name='PFR_QB_Stats', if_exist=r_or_a)
append_to_db(rb, table_name='PFR_RB_Stats', if_exist=r_or_a)
append_to_db(rec, table_name='PRF_Rec_Stats', if_exist=r_or_a)

# # Team Stats

# +
# team_pass = apd.read_html(f'https://www.nfl.com/stats/team-stats/offense/passing/{set_year}/reg/all')
# team_rush = pd.read_html(f'https://www.nfl.com/stats/team-stats/offense/rushing/{set_year}/reg/all')
# -

team_pass = pd.read_html('http://www.nfl.com/stats/categorystats?tabSeq=2&offensiveStatisticCategory=' + \
                         'TEAM_PASSING&conference=ALL&role=TM&season={}&seasonType=REG'.format(set_year))[0]
team_rush = pd.read_html('http://www.nfl.com/stats/categorystats?archive=false&conference=null&role=TM&offensiveStatisticCategory=' + \
                         'RUSHING&defensiveStatisticCategory=null&season={}&seasonType=REG&tabSeq=2&qualified=false&Submit=Go'.format(set_year))[0]
team_oline = pd.read_html('http://www.nfl.com/stats/categorystats?archive=false&conference=null&role=TM&' + \
                          'offensiveStatisticCategory=OFFENSIVE_LINE&defensiveStatisticCategory=null&season={}&seasonType=REG&tabSeq=2&qualified=false&Submit=Go'.format(set_year))[0]
team_ostats = pd.read_html('http://www.nfl.com/stats/categorystats?archive=false&conference=null&role=TM&offensiveStatisticCategory=' + \
                           'TOTAL_YARDS&defensiveStatisticCategory=null&season={}&seasonType=REG&tabSeq=2&qualified=false&Submit=Go'.format(set_year))[0]
team_def_yds = pd.read_html('http://www.nfl.com/stats/categorystats?archive=false&conference=null&role=OPP&offensiveStatisticCategory=null&defensiveStatisticCategory=' + \
                            'TOTAL_YARDS&season={}&seasonType=REG&tabSeq=2&qualified=false&Submit=Go'.format(set_year))[0]
team_def_sacks = pd.read_html('http://www.nfl.com/stats/categorystats?archive=false&conference=null&role=OPP&offensiveStatisticCategory=null&defensiveStatisticCategory=' + \
                         'SACKS&season={}&seasonType=REG&tabSeq=2&qualified=false&Submit=Go'.format(set_year))[0] 

team_ostats

# +
ostats_cols = ['rank', 'team', 'games', 'team_pts_per_game', 'total_pts', 'plays_run', 'team_total_yds_per_game',
               'yds_per_play', 'first_down_per_game', 'third_down_conversions', 'third_down_att',
               'pct_third_down', 'fourth_down_conversions', 'fourth_down_att', 'fourth_down_pct',
               'penalties', 'penalty_yds', 'time_of_pos', 'fumbles', 'fumbles_lost', 'turnover_margin']

team_pass_columns = ['rank', 'team', 'games', 'team_pts_per_game', 'total_pts', 'pass_completions', 
                     'pass_att', 'pct_complete', 'att_per_game', 'team_pass_yds', 'team_avg_pass_yds',
                     'team_pass_yds_per_game', 'team_pass_td', 'team_pass_int', 'team_first_downs',
                     'team_first_down_pct', 'long_pass', 'twenty_plus_pass', 'forty_plus_pass', 'team_sacks',
                     'team_pass_rate']

team_rush_columns = ['rank', 'team', 'games', 'team_pts_per_game', 'total_pts', 'team_rush_att', 
                     'team_rush_att_per_game', 'team_rush_yds', 'team_rush_yds_per_carry', 
                     'team_rush_yds_per_game', 'team_rush_td', 'team_long_rush', 'team_rush_fd',
                     'team_rush_first_down_pct', 'team_twenty_plus_rush', 'team_forty_plus_rush', 
                     'team_fumble']

oline_cols = ['rank', 'team', 'oline_exp', 'rush_att', 'rush_yds', 'rush_ypc', 'rush_tds', 
              'rush_first_downs_left', 'neg_rush_left', 'ten_plus_rush_left', 'pct_short_yd_left',
              'rush_first_downs_center', 'neg_rush_center', 'ten_plus_rush_center', 'pct_short_yd_center',
              'rush_first_downs_right', 'neg_rush_right', 'ten_plus_rush_right', 'pct_short_yd_right',
              'sacks_allowed',' qb_hits_allowed']

team_def_yds_cols = ['rank', 'team', 'games', 'pts_allow_per_game', 'total_pts_allowed', 'plays_run_allowed',
                     'yds_per_game_allowed', 'yds_per_play_allowed', 'first_down_per_game_allowed',
                     'third_down_allowed', 'third_down_att_allowed', 'third_down_pct_allowed', 
                     'fourth_down_allowed',' fourth_down_att_allowed', 'fourth_down_pct_allowed',
                     'penalty_against', 'penalty_yds_against', 'time_of_pos_allowed', 'fumbles', 'fumbles_rec']

team_def_sacks_cols = ['rank', 'team', 'games', 'pts_per_game_allowed', 'total_pts_allowed', 'combined_tackles',
                       'solo_tackles', 'assisted_tackles', 'sacks_given', 'safeties', 'passes_defended',
                       'interceptions_def', 'tds_from_int', 'yds_from_int', 'long_int', 'forced_fumbles',
                       'recovered_fumbles', 'td_from_fumble']

team_pass.columns = team_pass_columns
team_rush.columns = team_rush_columns
team_oline.columns = oline_cols
team_ostats.columns = ostats_cols
team_def_yds.columns = team_def_yds_cols
team_def_sacks.columns = team_def_sacks_cols


def team_stats_clean(df):
    df = df.iloc[:32, :]
    df = convert_float(df)
    df['week'] = set_week
    df['year'] = set_year
    df['team'] = df.team.map(name_map)
    
    return df
    
team_pass = team_stats_clean(team_pass) 
team_rush = team_stats_clean(team_rush)
team_oline = team_stats_clean(team_oline)
team_ostats = team_stats_clean(team_ostats)
team_def_yds = team_stats_clean(team_def_yds)
team_def_sacks = team_stats_clean(team_def_sacks)    

team_ostats.time_of_pos = team_ostats.time_of_pos.apply(lambda x: float(str(x).replace(':', '.')))
team_ostats.turnover_margin = team_ostats.turnover_margin.apply(lambda x: float(str(x).replace('+', '')))
team_def_yds.time_of_pos_allowed = team_def_yds.time_of_pos_allowed.apply(lambda x: float(str(x).replace(':', '.')))
# -

a_or_r = 'append'
append_to_db(team_pass, table_name='Team_Pass_Stats', if_exist=a_or_r)
append_to_db(team_rush, table_name='Team_Rush_Stats', if_exist=a_or_r)
append_to_db(team_oline, table_name='Team_Oline_Stats', if_exist=a_or_r)
append_to_db(team_ostats, table_name='Team_Off_Stats', if_exist=a_or_r)
append_to_db(team_def_yds, table_name='Team_Def_Stats', if_exist=a_or_r)
append_to_db(team_def_sacks, table_name='Team_Def_Sacks', if_exist=a_or_r)

# # Snap Counts

# +
# set_week = 6

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

# +
# set_week = 7
# -

append_to_db(snaps, table_name='Snap_Counts', if_exist='append')

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

qbr.head()

qbr.columns = ['rank', 'player', 'pts_added', 'plays', 'pass_added', 'run_added', 'penalty_added',
               'total_epa', 'qb_plays', 'raw_qbr', 'total_qbr']
qbr = qbr[qbr.player != 'PLAYER'].reset_index(drop=True)
qbr['week'] = set_week
qbr['year'] = set_year
qbr['team'] = qbr.player.apply(lambda x: x.split(' ')[-1].lstrip(' ').rstrip(' '))
qbr['player'] = qbr.player.apply(lambda x: ' '.join(x.split(' ')[:-1]).lstrip(' ').rstrip(' '))
# qbr['team'] = qbr.player.apply(lambda x: x.split(',')[-1].lstrip(' ').rstrip(' '))
# qbr['player'] = qbr.player.apply(lambda x: ' '.join(x.split(',')[:-1]).lstrip(' ').rstrip(' '))
qbr = convert_float(qbr)
qbr.team = qbr.team.map(espn_map)

append_to_db(qbr, table_name='QBR', if_exist='append')

# # Football Outsiders

fo_fp_map = {'ARI': 'ARI',
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
             'OAK': 'OAK',
             'PHI': 'PHI',
             'PIT': 'PIT',
             'SEA': 'SEA',
             'SF': 'SF',
             'TB': 'TB',
             'TEN': 'TEN',
             'WAS': 'WAS'}

# +
team_off_dvoa = pd.read_html('https://www.footballoutsiders.com/stats/teamoff/{}'.format(set_year))[0]
team_def_dvoa = pd.read_html('https://www.footballoutsiders.com/stats/teamdef/{}'.format(set_year))[0]

qb_dvoa_ = pd.read_html('https://www.footballoutsiders.com/stats/qb/{}'.format(set_year))
qb_dvoa = pd.concat([qb_dvoa_[0], qb_dvoa_[1]], axis=0, sort=False)
qb_dvoa_run = qb_dvoa_[2]

rb_dvoa_ = pd.read_html('https://www.footballoutsiders.com/stats/rb/{}'.format(set_year))
rb_dvoa = pd.concat([rb_dvoa_[0], rb_dvoa_[1]], axis=0, sort=False)
rb_dvoa_rec = pd.concat([rb_dvoa_[2], rb_dvoa_[3]], axis=0, sort=False)

wr_dvoa_ = pd.read_html('https://www.footballoutsiders.com/stats/wr/{}'.format(set_year))
wr_dvoa = pd.concat([wr_dvoa_[0], wr_dvoa_[1]], axis=0, sort=False)

te_dvoa_ = pd.read_html('https://www.footballoutsiders.com/stats/te/{}'.format(set_year))
te_dvoa = pd.concat([te_dvoa_[0], te_dvoa_[1]], axis=0, sort=False)

ol_dvoa = pd.read_html('https://www.footballoutsiders.com/stats/ol/{}'.format(set_year))[0]
dl_dvoa = pd.read_html('https://www.footballoutsiders.com/stats/dl/{}'.format(set_year))[0]

# +
team_off_cols = ['rank', 'team', 'off_dvoa', 'off_last_week', 'off_dave', 'off_dave_rank', 'pass_off_dvoa',
                 'pass_off_rank', 'rush_off_dvoa', 'rush_off_dvoa_rank']

team_def_cols = ['rank', 'team', 'deff_dvoa', 'def_last_week', 'def_dave', 'def_dave_rank', 'pass_def_dvoa',
                 'pass_def_rank', 'rush_def_dvoa', 'rush_def_dvoa_rank']

qb_dvoa_cols = ['player', 'team', 'dyar', 'rank', 'dvoa', 'dvoa_rank1', 'qbr', 'qbr_rank',
                'passes', 'pass_yds', 'effective_pass_yds', 'pass_td', 'fk', 'fl', 'ints',
                'complete_pct', 'defensive_pass_interfere', 'alex']

rb_dvoa_cols = ['player', 'team', 'dyar', 'rank', 'dvoa', 'dvoa_rank1', 'runs',
                'rush_yds', 'effective_rush_yds', 'rush_td', 'fumbles', 'success_rate', 'success_rate_rank']

wr_dvoa_cols = ['player', 'team', 'dyar', 'rank', 'dvoa', 'dvoa_rank1', 'rec_dvoa',
                'rec_yds', 'effective_rec_yds', 'rectd', 'catch_rate', 'fumbles', 'dpi']

ol_dvoa_cols = ['rank', 'team_rush', 'adj_line_yds', 'rb_yds', 'power_success', 'power_rank', 'stuffed_pct',
                'stuffed_rank', 'second_level_yds', 'second_level_rank', 'open_field_yds', 'open_field_rank',
                'team_pass', 'pass_rank', 'sacks_allowed', 'adj_sack_rate']

dl_dvoa_cols = ['rank', 'team_rush', 'adj_line_yds', 'rb_yds', 'power_success', 'power_rank', 'stuffed_pct',
                'stuffed_rank', 'second_level_yds', 'second_level_rank', 'open_field_yds', 'open_field_rank',
                'team_pass', 'pass_rank', 'sacks_allowed', 'adj_sack_rate']
rb_rec_dvoa_cols = ['player', 'team', 'dyar', 'rank', 'dvoa', 'dvoa_rank1', 'rec_dvoa',
                    'rec_yds', 'effective_rec_yds', 'rectd', 'catch_rate', 'fumbles']

team_off_dvoa = team_off_dvoa.iloc[:, :len(team_off_cols)]
team_off_dvoa.columns = team_off_cols

team_def_dvoa = team_def_dvoa.iloc[:, :len(team_def_cols)]
team_def_dvoa.columns = team_def_cols

qb_dvoa = qb_dvoa.drop(['YAR', 'Rk.1', 'VOA'], axis=1)
qb_dvoa.columns = qb_dvoa_cols

rb_dvoa = rb_dvoa.drop(['YAR', 'Rk.1', 'VOA'], axis=1)
rb_dvoa.columns = rb_dvoa_cols

rb_dvoa_rec = rb_dvoa_rec.drop(['YAR', 'Rk.1', 'VOA'], axis=1)
rb_dvoa_rec.columns = rb_rec_dvoa_cols

wr_dvoa = wr_dvoa.drop(['YAR', 'Rk.1', 'VOA'], axis=1)
wr_dvoa.columns = wr_dvoa_cols

te_dvoa = te_dvoa.drop(['YAR', 'Rk.1', 'VOA'], axis=1)
te_dvoa.columns = wr_dvoa_cols

ol_dvoa = ol_dvoa.iloc[:, :len(ol_dvoa_cols)]
ol_dvoa.columns = ol_dvoa_cols

dl_dvoa = dl_dvoa.iloc[:, :len(dl_dvoa_cols)]
dl_dvoa.columns = dl_dvoa_cols

try:
    qb_dvoa.qbr_rank = qb_dvoa.qbr_rank.apply(lambda x: float(x.replace('--', '0')))
except:
    pass
qb_dvoa = qb_dvoa.fillna(0)
qb_dvoa['pass_interfere'] = qb_dvoa.defensive_pass_interfere.apply(lambda x: float(x.split('/')[0]))
qb_dvoa['pass_interfere_yds'] = qb_dvoa.defensive_pass_interfere.apply(lambda x: float(x.split('/')[1]))
qb_dvoa = qb_dvoa.drop('defensive_pass_interfere', axis=1)

ol_rush_dvoa = ol_dvoa.loc[:, 'rank':'open_field_rank'].rename(columns={'team_rush': 'team'})
ol_pass_dvoa = ol_dvoa.loc[:, 'team_pass':'adj_sack_rate'].rename(columns={'team_pass': 'team'})
ol_dvoa_df = pd.merge(ol_rush_dvoa, ol_pass_dvoa, how='inner', left_on='team', right_on='team')
ol_dvoa_df = ol_dvoa_df[ol_dvoa_df.team != 'NFL']

dl_rush_dvoa = dl_dvoa.loc[:, 'rank':'open_field_rank'].rename(columns={'team_rush': 'team'})
dl_pass_dvoa = dl_dvoa.loc[:, 'team_pass':'adj_sack_rate'].rename(columns={'team_pass': 'team'})
dl_dvoa_df = pd.merge(dl_rush_dvoa, dl_pass_dvoa, how='inner', left_on='team', right_on='team')
dl_dvoa_df = dl_dvoa_df[dl_dvoa_df.team != 'NFL']

for c in ['rank', 'dvoa_rank1']:
    rb_dvoa[c] = rb_dvoa[c].fillna(np.percentile(rb_dvoa[c].dropna(), 75))
    rb_dvoa_rec[c] = rb_dvoa_rec[c].fillna(np.percentile(rb_dvoa_rec[c].dropna(), 75))
    wr_dvoa[c] = wr_dvoa[c].fillna(np.percentile(wr_dvoa[c].dropna(), 75))
    te_dvoa[c] = te_dvoa[c].fillna(np.percentile(te_dvoa[c].dropna(), 75))

rb_dvoa['success_rate_rank'] = rb_dvoa['success_rate_rank'].fillna(np.percentile(rb_dvoa['success_rate_rank'].dropna(), 75))
rb_dvoa['success_rate'] = rb_dvoa.success_rate.fillna('40%')

for df in [team_off_dvoa, team_def_dvoa, qb_dvoa, rb_dvoa, wr_dvoa, te_dvoa, ol_dvoa_df, dl_dvoa_df, rb_dvoa_rec]:
    for c in df.columns:
        try:
            df[c] = df[c].apply(lambda x: float(x.replace('%', '')))
        except:
            pass
        
        try:
            df[c] = df[c].apply(lambda x: float(x.replace('x', '0')))
        except:
            pass
        
    df.team = df.team.map(fo_fp_map)
    df['week'] = set_week
    df['year'] = set_year
# -

a_or_r = 'append'
append_to_db(team_off_dvoa, table_name='DVOA_Team_Off', if_exist=a_or_r)
append_to_db(team_def_dvoa, table_name='DVOA_Team_Def', if_exist=a_or_r)
append_to_db(qb_dvoa, table_name='DVOA_QB', if_exist=a_or_r)
append_to_db(rb_dvoa, table_name='DVOA_RB', if_exist=a_or_r)
append_to_db(rb_dvoa_rec, table_name='DVOA_RB_Rec', if_exist=a_or_r)
append_to_db(wr_dvoa, table_name='DVOA_WR', if_exist=a_or_r)
append_to_db(te_dvoa, table_name='DVOA_TE', if_exist=a_or_r)
append_to_db(ol_dvoa_df, table_name='DVOA_OL', if_exist=a_or_r)
append_to_db(dl_dvoa_df, table_name='DVOA_DL', if_exist=a_or_r)

# # Backup Database

copy_db(set_week, set_year, 'post')

# +
import pandas as pd 

#Enter desired years of data
YEARS = [2019,2018,2017]

data = pd.DataFrame()

for i in YEARS:  
    #low_memory=False eliminates a warning
    i_data = pd.read_csv('https://github.com/guga31bb/nflfastR-data/blob/master/data/' \
                         'play_by_play_' + str(i) + '.csv.gz?raw=True',
                         compression='gzip', low_memory=False)

    #sort=True eliminates a warning and alphabetically sorts columns
    data = data.append(i_data, sort=True)

#Give each row a unique index
data.reset_index(drop=True, inplace=True)
# -

data.columns[:50]

df = pd.read_html('https://www.pro-football-reference.com/players/K/KamaAl00/gamelog/2019/advanced/')


