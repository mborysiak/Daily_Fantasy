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

import pandas as pd
import numpy as np
import requests 
import datetime as dt
import sqlite3

set_week=1
set_year=2020


def convert_float(df):
    for c in df.columns:
        try:
            df[c] = df[c].astype('float')
        except:
            pass
    
    return df


def copy_db(db_name, root_path, week=set_week, year=set_year):
    path = f'{root_path}/Databases/'
    db = sqlite3.connect(f'{path}/{db_name}.sqlite3')
    path2 = f'{path}/Backups/{db_name}_Week{week}_{year}.sqlite3'
    db2 = sqlite3.connect(path2)
 
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        print(table_name)
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        table.to_sql(table_name, db2, index=False, if_exists = 'replace')
    cursor.close()
    db.close()
    db2.close()


import datetime as dt
def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))

def create_adj_ranks(df, rank_col, where_pos, table_name, dm):
    
    all_data = dm.read(f"SELECT * FROM {table_name} {where_pos} AND {rank_col} < 100", 'Pre_PlayerData')
    all_data = all_data.sort_values(by=['year', 'week']).reset_index(drop=True)
    all_data['game_date'] = all_data[['year', 'week']].apply(year_week_to_date, axis=1)

    # get the mean rankings for past 6 weeks
    past_6_weeks = all_data.game_date.unique()[-6]
    mean_rank = all_data[all_data.game_date >= past_6_weeks].reset_index(drop=True)
    mean_rank = pd.concat([mean_rank, df], axis=0)
    mean_rank = mean_rank.groupby('player').agg({rank_col: 'mean'}).reset_index().sort_values(by=rank_col)
    mean_rank[rank_col] = range(1, len(mean_rank)+1)

    # join in the current rankings and find missing players
    cur_rank = df.copy()
    cur_rank = cur_rank.dropna(subset=[rank_col]).reset_index(drop=True)
    mean_rank = pd.merge(mean_rank, cur_rank[['player', 'week']], on=['player'], how='outer')
    mean_rank['to_add'] = np.where(mean_rank.week.isnull(), 1, 0)
    mean_rank['to_add'] = mean_rank.to_add.cumsum()
    
    # join current back to the mean and adjust the rankings for missing players
    cur_rank = pd.merge(cur_rank, mean_rank[[rank_col, 'to_add']], on=rank_col)
    cur_rank['rankadj_' + rank_col] = cur_rank[rank_col] + cur_rank.to_add
    cur_rank = cur_rank.drop(['to_add'], axis=1)

    # join current back to the mean and adjust the rankings for missing players
    cur_rank = pd.merge(cur_rank, mean_rank[['player', 'to_add']], on='player', how='left')
    cur_rank['playeradj_' + rank_col] = cur_rank[rank_col] + cur_rank.to_add
    cur_rank = cur_rank.drop(['to_add'], axis=1)

    return cur_rank





team_map = {
    'ARI': 'ARI',
    'ARZ': 'ARI',
    'ATL': 'ATL',
    'BAL': 'BAL',
    'BLT': 'BAL',
    'BUF': 'BUF',
    'CAR': 'CAR',
    'CHI': 'CHI',
    'CIN': 'CIN',
    'CLE': 'CLE',
    'CLV': 'CLE',
    'DAL': 'DAL',
    'DEN': 'DEN',
    'DET': 'DET',
    'GB': 'GB',
    'GNB': 'GB',
    'HOU': 'HOU',
    'HST': 'HOU',
    'IND': 'IND',
    'JAC': 'JAC',
    'JAX': 'JAC',
    'KC': 'KC',
    'KAN': 'KC',
    'LA': 'LAR',
    'LAR': 'LAR',
    'LAC': 'LAC',
    'MIA': 'MIA',
    'MIN': 'MIN',
    'NE': 'NE',
    'NWE': 'NE',
    'NO': 'NO',
    'NOR': 'NO',
    'NYG': 'NYG',
    'NYJ': 'NYJ',
    'OAK': 'LVR',
    'LVR': 'LVR',
    'LV': 'LVR',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SEA': 'SEA',
    'SF': 'SF',
    'SFO': 'SF',
    'TB': 'TB',
    'TAM': 'TB',
    'TEN': 'TEN',
    'WAS': 'WAS',
    'WFT': 'WAS'
}



# +
pfr_fp_map = {'ARI': 'ARI',
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
             'GNB': 'GB',
             'HOU': 'HOU',
             'IND': 'IND',
             'JAX': 'JAC',
             'KAN': 'KC',
             'LAC': 'LAR',
             'LAR': 'LAC',
             'MIA': 'MIA',
             'MIN': 'MIN',
             'NWE': 'NE',
             'NOR': 'NO',
             'NYG': 'NYG',
             'NYJ': 'NYJ',
             'OAK': 'OAK',
             'LVR': 'LVR',
             'PHI': 'PHI',
             'PIT': 'PIT',
             'SEA': 'SEA',
             'SFO': 'SF',
             'TAM': 'TB',
             'TEN': 'TEN',
             'WAS': 'WAS'}

def pfr_matchup_cleanup(df, cols):

    status = df.iloc[:, 2]
    status.name = 'inj_status'
    df.columns = [c[1] for c in df.columns]
    if df.FantPt.isnull().sum()[0] > 0:
        df['FantPt']  = df['DKPt']
    df = df.dropna(axis=1, thresh=int(df.shape[0]*0.75))
    df.columns = cols
    df = pd.concat([df, status], axis=1)
    df['inj_status'] = df.inj_status.fillna('NotListed')
    df = df.dropna(axis=0)
    df['total_snaps'] = df.snap_pct.apply(lambda x: x.split('(')[0])
    df['snap_pct'] = df.snap_pct.apply(lambda x: x.split('(')[1].replace('%', '').replace(')', ''))
    df['week'] = set_week
    df['year'] = set_year
    df.team = df.team.map(pfr_fp_map)
    df.opponent = df.opponent.map(pfr_fp_map)
    for c in df.columns:
        try:
            df[c] = df[c].astype('float')
        except:
            pass
        
    df = df[['player', 'team','opponent', 'opp_rank', 'opp_fp_per_game', 
           'opp_dk_pt_per_game', 'opp_fd_pt_per_game', 'proj_fp_rank', 'proj_dk_rank', 'proj_fd_rank']]
        
    return df


# -
# # used for bovada data pull
full_team_map = {'Arizona Cardinals': 'ARI',
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
 'Kansas City Cheifs': 'KC',
 'Los Angeles Chargers': 'LAC',
 'Los Angeles Chargers Chargers': 'LAC',
 'Los Angeles Rams Rams': 'LAR',
 'Los Angeles Rams': 'LAR',
 'Miami Dolphins': 'MIA',
 'Minnesota Vikings': 'MIN',
 'New England Patriots': 'NE',
 'New Orleans Saints': 'NO',
 'New York Giants': 'NYG',
 'New York Jets': 'NYJ',
 'New York Jets Jets': 'NYJ',
 'New York Giants Giants': 'NYG',
 'Oakland Raiders': 'LVR',
 'Las Vegas Raiders': 'LVR',
 'Philadelphia Eagles': 'PHI',
 'Pittsburgh Steelers': 'PIT',
 'San Francisco 49ers': 'SF',
 'Seattle Seahawks': 'SEA',
 'Tampa Bay Buccaneers': 'TB',
 'Tennessee Titans': 'TEN',
 'Washington Redskins': 'WAS',
 'Washington Football Team': 'WAS',
 'Washington Commanders': 'WAS'}

name_map = {'Cardinals': 'ARI',
 'Falcons': 'ATL',
 'Ravens': 'BAL',
 'Bills': 'BUF',
 'Panthers': 'CAR',
 'Bears': 'CHI',
 'Bengals': 'CIN',
 'Browns': 'CLE',
 'Cowboys': 'DAL',
 'Broncos': 'DEN',
 'Lions': 'DET',
 'Packers': 'GB',
 'Texans': 'HOU',
 'Colts': 'IND',
 'Jaguars': 'JAC',
 'Chiefs': 'KC',
 'Chargers': 'LAC',
 'Rams': 'LAR',
 'Dolphins': 'MIA',
 'Vikings': 'MIN',
 'Patriots': 'NE',
 'Saints': 'NO',
 'Giants': 'NYG',
 'Jets': 'NYJ',
 'Raiders': 'LVR',
 'Eagles': 'PHI',
 'Steelers': 'PIT',
 '49ers': 'SF',
 '49Ers': 'SF',
 'Seahawks': 'SEA',
 'Buccaneers': 'TB',
 'Titans': 'TEN',
 'Redskins': 'WAS',
 'Football Team': 'WAS',
 'Commanders': 'WAS'}

city_map = {'Arizona': 'ARI',
 'Atlanta': 'ATL',
 'Baltimore': 'BAL',
 'Buffalo': 'BUF',
 'Carolina': 'CAR',
 'Chicago': 'CHI',
 'Cincinnati': 'CIN',
 'Cleveland': 'CLE',
 'Dallas': 'DAL',
 'Denver': 'DEN',
 'Detroit': 'DET',
 'Green Bay': 'GB',
 'Houston': 'HOU',
 'Indianapolis': 'IND',
 'Jacksonville': 'JAC',
 'Kansas City': 'KC',
 'Los Angeles': 'LAR',
 'Los Angeles': 'LAC',
 'Miami': 'MIA',
 'Minnesota': 'MIN',
 'New England': 'NE',
 'New Orleans': 'NO',
 'New York': 'NYJ',
 'New York': 'NYG',
 'Oakland': 'OAK',
 'Las Vegas': 'LVR',
 'Philadelphia': 'PHI',
 'Pittsburgh': 'PIT',
 'San Francisco': 'SF',
 'Seattle': 'SEA',
 'Tampa Bay': 'TB',
 'Tennessee': 'TEN',
 'Washington': 'WAS'}

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
              'LV': 'LVR',
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
         'LVR': 'LVR',
         'PHI': 'PHI',
         'PIT': 'PIT',
         'SEA': 'SEA',
         'SF': 'SF',
         'TB': 'TB',
         'TEN': 'TEN',
         'WAS': 'WAS',
         'WFT': 'WAS'}

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
             'Washington Football Team': 'WAS',
             'Washington Commanders': 'WAS'}

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

#%%

import yaml
import pytz
from ff import general as ffgeneral

root_path = ffgeneral.get_main_path('Daily_Fantasy')

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Assuming the config file is in the same directory as the Python script
config_file = f'{root_path}/Scripts/config.yaml'
config = read_config(config_file)

api_key = config['odds_api_key']
sport = 'americanfootball_nfl' # use the sport_key from the /sports endpoint below, or use 'upcoming' to see the next 8 games across all sports
region = 'us' # uk | us | eu | au. Multiple can be specified if comma delimited
odds_format = 'decimal' # decimal | american
date_format = 'iso' # iso | unix

class OddsAPIPull:

    def __init__(self, week, year, api_key, base_url, sport, region, odds_format, date_format, historical=False):
        
        self.week = week
        self.year = year
        self.api_key = api_key
        self.sport = sport
        self.region = region
        self.odds_format = odds_format
        self.date_format = date_format
        self.historical = historical

        if self.historical: 
            self.base_url = f'{base_url}/historical/sports/'
        else: 
            self.base_url = f'{base_url}/sports/'

    def get_response(self, r_pull):
        if r_pull.status_code != 200:
            print(f'Failed to get odds: status_code {r_pull.status_code}, response body {r_pull.text}')
        else:
            r_json = r_pull.json()
            print('Number of events:', len(r_json))
            print('Remaining requests', r_pull.headers['x-requests-remaining'])
            print('Used requests', r_pull.headers['x-requests-used'])

        return r_json

    @staticmethod
    def convert_utc_to_est(est_dt):
        
        # Define the EST timezone
        est = pytz.timezone('US/Eastern')

        # Localize the datetime object to EST
        local_time_est = est.localize(est_dt)

        # Convert the localized datetime to UTC
        utc_time = local_time_est.astimezone(pytz.utc)
        
        return utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def get_weekday_name(date_string):
        dt_utc = dt.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
        dt_utc = dt_utc.replace(tzinfo=pytz.UTC)

        # Convert to EST
        est_tz = pytz.timezone('America/New_York')
        dt_est = dt_utc.astimezone(est_tz)

        # Get the day name in EST
        weekday_name = dt_est.strftime("%A")
        return weekday_name

    def pull_events(self, start_time, end_time):
        
        if start_time is not None: start_time = self.convert_utc_to_est(start_time)
        if end_time is not None: end_time = self.convert_utc_to_est(end_time)

        get_params = {
                'api_key': self.api_key,
                'regions': self.region,
                'oddsFormat': self.odds_format,
                'dateFormat': self.date_format,
                'commenceTimeFrom': start_time,
                'commenceTimeTo': end_time
            }
        
        if self.historical:
            get_params['date'] = start_time
            self.start_time = start_time

        events = requests.get(
            f'{self.base_url}/{self.sport}/events',
            params=get_params
            )
        
        events_json = self.get_response(events)
        if self.historical: events_json = events_json['data']

        events_df = pd.DataFrame()
        for e in events_json:
            events_df = pd.concat([events_df, pd.DataFrame(e, index=[0])], axis=0)
        
        events_df['week'] = self.week
        events_df['year'] = self.year
        events_df['day_of_week'] = events_df.commence_time.apply(self.get_weekday_name)
        events_df = events_df.rename(columns={'id': 'event_id'})

        return events_df.reset_index(drop=True)
    

    def pull_lines(self, markets, event_id):

        get_params={
                    'api_key': self.api_key,
                    'regions': self.region,
                    'markets': markets,
                    'oddsFormat': self.odds_format,
                    'dateFormat': self.date_format,
                }
       
        if self.historical:
            get_params['date'] = self.start_time

        odds = requests.get(
                f'{self.base_url}/{self.sport}/events/{event_id}/odds',
                params = get_params
            )
        
        odds_json = self.get_response(odds)
        if self.historical: odds_json = odds_json['data']

        props = pd.DataFrame()
        for odds in odds_json['bookmakers']:
            bookmaker = odds['key']
            market_props = odds['markets']
            for cur_prop in market_props:
                p = pd.DataFrame(cur_prop['outcomes'])
                p['bookmaker'] = bookmaker
                p['prop_type'] = cur_prop['key']
                p['event_id'] = event_id

                if cur_prop['key'] in ('spreads', 'h2h'):
                    p = p.rename(columns={'name': 'description'})

                props = pd.concat([props, p], axis=0)
                

        props = props.reset_index(drop=True)
        props['week'] = self.week
        props['year'] = self.year

        return props

    def all_market_odds(self, markets, events_df):

        props = pd.DataFrame()
        for event_id in events_df.event_id.values:
            try:
                print(event_id)
                cur_props = self.pull_lines(markets, event_id)
                props = pd.concat([props, cur_props], axis=0)
            except:
                print(f'Failed to get data for event_id {event_id}')

        return props


#%%
import pandas as pd 
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
import warnings
from scipy.stats import poisson, truncnorm
from typing import Dict

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

pd.set_option('display.max_columns', 999)

team_d_map = {
 'Arizona Cardinals': 'ARI','Arizona': 'ARI','Ari': 'ARI',
 'Atlanta Falcons': 'ATL','Atlanta': 'ATL','Atl': 'ATL', 'Atl Falcons': 'ATL',
 'Baltimore Ravens': 'BAL','Baltimore': 'BAL','Bal': 'BAL',
 'Buffalo Bills': 'BUF','Buffalo': 'BUF','Buf': 'BUF',
 'Carolina Panthers': 'CAR','Carolina': 'CAR','Car': 'CAR',
 'Chicago Bears': 'CHI','Chicago': 'CHI','Chi': 'CHI',
 'Cincinnati Bengals': 'CIN','Cincinnati': 'CIN', 'Cin': 'CIN', 'Cin Bengals': 'CIN',
 'Cleveland Browns': 'CLE','Cleveland': 'CLE','Cle': 'CLE','Cle Browns': 'CLE',
 'Dallas Cowboys': 'DAL','Dallas': 'DAL', 'Dal': 'DAL',
 'Denver Broncos': 'DEN','Denver': 'DEN','Den': 'DEN',
 'Detroit Lions': 'DET','Detroit': 'DET', 'Det': 'DET',
 'Green Bay Packers': 'GB','Green Bay': 'GB','Gb': 'GB',
 'Houston Texans': 'HOU','Houston': 'HOU','Hou': 'HOU',
 'Indianapolis Colts': 'IND','Indianapolis': 'IND','Ind': 'IND',
 'Jacksonville Jaguars': 'JAC','Jacksonville': 'JAC','Jac': 'JAC',
 'Kansas City Chiefs': 'KC','Kansas City': 'KC','Kc': 'KC','Kansas City Cheifs': 'KC',
 'Los Angeles Chargers': 'LAC','La Chargers': 'LAC','Lac': 'LAC','Los Angeles Chargers Chargers': 'LAC',
 'Los Angeles Rams Rams': 'LAR','Los Angeles Rams': 'LAR','La Rams': 'LAR','Lar': 'LAR',
 'Miami Dolphins': 'MIA','Miami': 'MIA','Mia': 'MIA',
 'Minnesota Vikings': 'MIN','Minnesota': 'MIN','Min': 'MIN',
 'New England Patriots': 'NE','New England': 'NE','Ne': 'NE',
 'New Orleans Saints': 'NO','New Orleans': 'NO','No': 'NO', 'Saints': 'NO',
 'New York Giants': 'NYG','Ny Giants': 'NYG','Nyg': 'NYG',
 'New York Jets': 'NYJ','Ny Jets': 'NYJ','Nyj': 'NYJ','New York Jets Jets': 'NYJ','New York Giants Giants': 'NYG',
 'Oakland Raiders': 'LVR','Las Vegas Raiders': 'LVR','Las Vegas': 'LVR','Lvr': 'LVR',
 'Philadelphia Eagles': 'PHI','Philadelphia': 'PHI','Phi': 'PHI',
 'Pittsburgh Steelers': 'PIT','Pittsburgh': 'PIT','Pit': 'PIT',
 'San Francisco 49ers': 'SF','San Francisco': 'SF','Sf': 'SF','San Francisco 49Ers': 'SF',
 'Seattle Seahawks': 'SEA','Seattle': 'SEA', 'Sea': 'SEA',
 'Tampa Bay Buccaneers': 'TB','Tampa Bay': 'TB','Tb': 'TB',
 'Tennessee Titans': 'TEN','Tennessee': 'TEN','Ten': 'TEN',
 'Washington Redskins': 'WAS','Washington': 'WAS','Was': 'WAS',
 'Washington Football Team': 'WAS','Washington Commanders': 'WAS'}


def get_stat_constraints(past_data, pos, stat_cols):

    if pos in ('RB', 'WR', 'TE'): filter_str = '(rec_pass_attempt_sum + rush_rush_attempt_sum) > 5'
    elif pos == 'QB': filter_str = '(pass_pass_attempt_sum + rush_rush_attempt_sum) > 15'
    else: filter_str = 'week > 0'

    player_data = dm.read(f'''SELECT * 
                              FROM {pos}_Stats 
                              WHERE week < 17
                                        AND season >= 2020
                                        AND {filter_str}
                                    ''', 'FastR').rename(columns={'season': 'year'})
    
    if pos == 'QB': player_data['rec_pass_touchdown_sum'] = 0
    if pos == 'Defense': player_data = player_data.rename(columns={'defteam': 'player'})
    
    player_data = pd.merge(past_data, player_data, on=['player', 'week', 'year'])
    stat_mean = player_data[stat_cols].sum(axis=1).mean()
    stat_std = player_data[stat_cols].sum(axis=1).std()
    stat_min = player_data[stat_cols].sum(axis=1).min()
    stat_max = player_data[stat_cols].sum(axis=1).max()

    stat_cv = stat_std / stat_mean

    return stat_mean, stat_cv, stat_min, stat_max


def trunc_normal(mean_val, sdev, min_sc, max_sc, num_samples=500):

    import scipy.stats as stats

    # create truncated distribution
    lower_bound = (min_sc - mean_val) / sdev, 
    upper_bound = (max_sc - mean_val) / sdev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc=mean_val, scale=sdev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


class PlayerPropsCalculator:
    """
    Vectorized calculator for player props expected values using 
    Poisson and Truncated Normal distributions.
    """
    
    def __init__(self, 
                 max_iterations: int = 20, 
                 max_lambda: float = 50.0):
        """
        Initialize calculator with configuration parameters.
        
        Args:
            max_iterations: Maximum iterations for binary search
            max_lambda: Maximum lambda value for Poisson

        """
        self.max_iterations = max_iterations
        self.max_lambda = max_lambda
        
   
    
    def calculate_discrete_probabilities_vectorized(self, lambda_params: np.ndarray, 
                                                  thresholds: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized calculation of discrete probabilities for Poisson.
        """
        floor_thresholds = np.floor(thresholds).astype(int)
        ceil_thresholds = np.ceil(thresholds).astype(int)
        
        floor_probs = poisson.cdf(floor_thresholds, lambda_params)
        ceil_probs = poisson.cdf(ceil_thresholds, lambda_params)
        weights = ceil_thresholds - thresholds
        
        return {
            'floor_probs': floor_probs,
            'ceil_probs': ceil_probs,
            'weights': weights
        }
    
    def find_poisson_parameters_vectorized(self, points: np.ndarray, 
                                         target_probs: np.ndarray) -> np.ndarray:
        """
        Vectorized binary search to find lambda parameters.
        """
        low = np.zeros_like(points)
        high = np.full_like(points, self.max_lambda)
        
        for _ in range(self.max_iterations):
            mid = (low + high) / 2
            probs = self.calculate_discrete_probabilities_vectorized(mid, points)
            interpolated_probs = (
                probs['floor_probs'] * probs['weights'] + 
                probs['ceil_probs'] * (1 - probs['weights'])
            )
            over_probs = 1 - interpolated_probs
            high = np.where(over_probs > target_probs, mid, high)
            low = np.where(over_probs <= target_probs, mid, low)
        
        return (low + high) / 2

    def calculate_truncnorm_ev_vectorized(self, means: np.ndarray, 
                                        stds: np.ndarray) -> np.ndarray:
        """
        Calculate expected values for truncated normal with configured bounds.
        """
        min_value = self.truncnorm_config['min_value']
        max_value = self.truncnorm_config['max_value']
        
        # Calculate normalized bounds
        a = (min_value - means) / stds
        b = (max_value - means) / stds
        
        # Calculate expected value using truncated normal formula
        expected_values = truncnorm.mean(a, b, loc=means, scale=stds)
        
        return expected_values
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire DataFrame of player props using vectorized operations.
        """
        required_columns = {'point', 'price', 'prop_type', 'week', 'year'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = df.copy()
        result_df['point'] = pd.to_numeric(result_df['point'], errors='coerce')
        result_df['price'] = pd.to_numeric(result_df['price'], errors='coerce')
        
        valid_mask = (
            ~result_df['point'].isna() & 
            ~result_df['price'].isna() & 
            (result_df['price'] > 1) & 
            (result_df['point'] > 0)
        )
        
        # Initialize columns
        result_df['ev_poisson'] = np.nan
        
        if valid_mask.any():
            points = result_df.loc[valid_mask, 'point'].values
            prices = result_df.loc[valid_mask, 'price'].values
            implied_probs = 1 / prices
            
            # Poisson calculations
            lambda_params = self.find_poisson_parameters_vectorized(points, implied_probs)
            poisson_evs = lambda_params
            
            # Assign results
            result_df.loc[valid_mask, 'ev_poisson'] = poisson_evs
        
        return result_df
    
def cleanup_vegas_names(df):
    df['is_defense'] = 0
    df.loc[df.description.str.contains('Defense|D/ST'), 'is_defense'] = 1
    df_def = df[df.is_defense==1].copy()
    df = df[df.is_defense==0].copy()

    df.description = df.description.apply(dc.name_clean)
    df_def.description = df_def.description.apply(dc.name_clean).apply(lambda x: x.split('Defense')[0].split('D/St')[0]).str.strip()
    print('Missing D mappings:', [c for c in df_def.description.unique() if c not in team_d_map.keys()])
    df_def.description = df_def.description.map(team_d_map)

    df = pd.concat([df, df_def], axis=0)
    df = df.drop(['is_defense'], axis=1)
    df = df.rename(columns={'description': 'player'})

    return df

def pull_vegas_stats(prop_types, pos):
    
    df = dm.read(f'''SELECT *
                     FROM Game_Odds
                     WHERE prop_type IN {prop_types}
                           AND name IN ('Over', 'Yes')
                ''', 'Pre_PlayerData')
    
    df.loc[df.prop_type == 'player_anytime_td', 'point'] = 0.5
    df = cleanup_vegas_names(df)

    if pos == 'Defense': 
        pos = 'DST'
        player_col = 'team'
    else: 
        player_col = 'player'
    fp = dm.read(f'''
                    SELECT {player_col} player, week, year
                    FROM FantasyPros
                    WHERE pos='{pos}'
                 ''', 'Pre_PlayerData')
    
    df = pd.merge(df, fp, on=['player', 'week', 'year'], how='inner')

    return df

def get_all_vegas_stats(pos, week, year):

    pos_stats = {
    
    'QB': [
            [('player_pass_tds', 0), ['pass_pass_touchdown_sum']],
            [('player_pass_yds', 0), ['pass_yards_gained_sum']],
            [('player_pass_attempts', 0), ['pass_pass_attempt_sum']],
            [('player_pass_completions', 0), ['pass_complete_pass_sum']],
            [('player_pass_interceptions', 0), ['pass_interception_sum']],

            [('player_anytime_td', 'player_tds_over'), ['rush_rush_touchdown_sum', 'rec_pass_touchdown_sum']],
            [('player_rush_yds', 0),['rush_yards_gained_sum']],
            [('player_rush_attempts', 0), ['rush_rush_attempt_sum']]
    ],
    'RB': [
            [('player_anytime_td', 'player_tds_over'), ['rush_rush_touchdown_sum', 'rec_pass_touchdown_sum']],
            [('player_rush_yds', 0), ['rush_yards_gained_sum']],
            [('player_reception_yds', 0), ['rec_yards_gained_sum']],
            [('player_rush_attempts', 0), ['rush_rush_attempt_sum']],
            [('player_receptions', 0), ['rec_complete_pass_sum']]
        ],
    'WR': [
            [('player_anytime_td', 'player_tds_over'), ['rush_rush_touchdown_sum', 'rec_pass_touchdown_sum']],
            [('player_reception_yds', 0), ['rec_yards_gained_sum']],
            [('player_rush_attempts', 0), ['rush_rush_attempt_sum']],
            [('player_receptions', 0), ['rec_complete_pass_sum']],
            [('player_rush_yds', 0), ['rush_yards_gained_sum']]
        ],
    'TE': [
            [('player_anytime_td', 'player_tds_over'), ['rush_rush_touchdown_sum', 'rec_pass_touchdown_sum']],
            [('player_reception_yds', 0), ['rec_yards_gained_sum']],
            [('player_receptions', 0), ['rec_complete_pass_sum']],
            [('player_rush_yds', 0), ['rush_yards_gained_sum']]
        ],

    'Defense': [
        [('player_anytime_td', 'player_tds_over'), ['return_touchdown', 'def_td']]
        ]
    }

    if week == 1:
        week = 16
        year -= 1
    else:
        week -= 1

    if pos == 'Defense':
        past_data = dm.read(f'''SELECT player, week, year
                                FROM Defense_Data_Week{week}
                        ''', f'Model_Features_{year}')
    else:
        past_data = dm.read(f'''SELECT player, week, year
                                FROM Backfill_{pos}_Week{week}
                            ''', f'Model_Features_{year}')
    i = 0
    for prop_types, stat_cols in pos_stats[pos]:
        odds = pull_vegas_stats(prop_types, pos)
        _, stat_cv, stat_min, stat_max = get_stat_constraints(past_data, pos, stat_cols)
        odds['ev_trunc_norm'] = np.nan
        for point in odds.point.unique():
            td_dist = trunc_normal(point, point*stat_cv, stat_min, stat_max, num_samples=10000)
            odds.loc[odds.point==point, 'ev_trunc_norm'] = np.percentile(td_dist, 100/odds.loc[odds.point==point, 'price'])

        calculator = PlayerPropsCalculator(max_lambda=odds.point.max())
        odds = calculator.process_dataframe(odds)
        odds = odds.pivot_table(index=['player', 'year', 'week'], columns='prop_type', values=['ev_trunc_norm', 'ev_poisson']).reset_index()
        odds.columns = [f"{c[1]}_{c[0]}" if c[1]!='' else c[0] for c in odds.columns]
        if i == 0: player_vegas_stats = odds.copy()
        else: player_vegas_stats = pd.merge(player_vegas_stats, odds, on=['player', 'week', 'year'], how='outer')
        i += 1

    return player_vegas_stats
