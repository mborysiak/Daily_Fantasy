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


# function to append data to sqlite
def append_to_db(df, db_name, table_name, set_week, set_year, if_exist):

    import sqlite3
    import os
    
    #--------
    # Append pandas df to database in Github
    #--------

    os.chdir('/Users/Mark/Documents/Github/Daily_Fantasy/Data/Databases/')

    conn = sqlite3.connect(db_name)
    
    if if_exist == 'append':
        q = "DELETE FROM {} where week={} and year={}".format(table_name, set_week, set_year)
        conn.cursor().execute(q)

    df.to_sql(
    name=table_name,
    con=conn,
    if_exists=if_exist,
    index=False
    )

    #--------
    # Append pandas df to database in Dropbox
    #--------

    os.chdir('/Users/Mark/OneDrive/FF/Daily/')

    conn = sqlite3.connect(db_name)
    
    if if_exist == 'append':
        q = "DELETE FROM {} where week={} and year={}".format(table_name, set_week, set_year)
        conn.cursor().execute(q)
    
    df.to_sql(
    name=table_name,
    con=conn,
    if_exists=if_exist,
    index=False
    )


# +
import sqlite3
import pandas as pd


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
# name_map = {'Arizona Cardinals': 'ARI',
#  'Atlanta Falcons': 'ATL',
#  'Baltimore Ravens': 'BAL',
#  'Buffalo Bills': 'BUF',
#  'Carolina Panthers': 'CAR',
#  'Chicago Bears': 'CHI',
#  'Cincinnati Bengals': 'CIN',
#  'Cleveland Browns': 'CLE',
#  'Dallas Cowboys': 'DAL',
#  'Denver Broncos': 'DEN',
#  'Detroit Lions': 'DET',
#  'Green Bay Packers': 'GB',
#  'Houston Texans': 'HOU',
#  'Indianapolis Colts': 'IND',
#  'Jacksonville Jaguars': 'JAC',
#  'Kansas City Chiefs': 'KC',
#  'Los Angeles Chargers': 'LAC',
#  'Los Angeles Rams': 'LAR',
#  'Miami Dolphins': 'MIA',
#  'Minnesota Vikings': 'MIN',
#  'New England Patriots': 'NE',
#  'New Orleans Saints': 'NO',
#  'New York Giants': 'NYG',
#  'New York Jets': 'NYJ',
#  'Oakland Raiders': 'LVR',
#  'Las Vegas Raiders': 'LVR',
#  'Philadelphia Eagles': 'PHI',
#  'Pittsburgh Steelers': 'PIT',
#  'San Francisco 49ers': 'SF',
#  'Seattle Seahawks': 'SEA',
#  'Tampa Bay Buccaneers': 'TB',
#  'Tennessee Titans': 'TEN',
#  'Washington Redskins': 'WAS',
#  'Washington Football Team': 'WAS'}

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
 'Seahawks': 'SEA',
 'Buccaneers': 'TB',
 'Titans': 'TEN',
 'Redskins': 'WAS',
 'Football Team': 'WAS'}

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
             'Washington Football Team': 'WAS'}

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
