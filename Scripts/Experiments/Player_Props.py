#%%

import pandas as pd
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

urls = {
    'rush_yds': 'https://sportsbook.draftkings.com/leagues/football/nfl?category=rush/rec-props&subcategory=rush-yds',
  #  'rush_att': 'https://sportsbook.draftkings.com/leagues/football/nfl?category=rush/rec-props&subcategory=rush-attempts',
    'rec_yds': 'https://sportsbook.draftkings.com/leagues/football/nfl?category=rush/rec-props&subcategory=rec-yds',
    'rec': 'https://sportsbook.draftkings.com/leagues/football/nfl?category=rush/rec-props&subcategory=receptions',
    'pass_td': 'https://sportsbook.draftkings.com/leagues/football/nfl?category=passing-props',
    'pass_yds': 'https://sportsbook.draftkings.com/leagues/football/nfl?category=passing-props&subcategory=pass-yds',
    'pass_comp': 'https://sportsbook.draftkings.com/leagues/football/nfl?category=passing-props&subcategory=pass-completions',
    'pass_att': 'https://sportsbook.draftkings.com/leagues/football/nfl?category=passing-props&subcategory=pass-attempts',
    'pass_int': 'https://sportsbook.draftkings.com/leagues/football/nfl?category=passing-props&subcategory=interceptions'
}
# %%

def cleanup_data(data, stat_type):
    df = pd.DataFrame()
    for i in range(len(data)):
        if data[i].columns[0]=='PLAYER':
            df = pd.concat([df, data[i]], axis=0)

    df['OVER'] = df.OVER.apply(lambda x: x.split('O\xa0')[-1])
    df['UNDER'] = df.UNDER.apply(lambda x: x.split('U\xa0')[-1])

    df['over_odds'] = df.OVER.apply(lambda x: x[-4:])
    df['under_odds'] = df.UNDER.apply(lambda x: x[-4:])
    df['over_under_value'] = df.OVER.apply(lambda x: x[:-4])
    df = df.drop(['OVER', 'UNDER'], axis=1).rename(columns={'PLAYER': 'player'})
    df['stat_type'] = stat_type
    return df

#%%

import pandas as pd
import numpy as np

import requests
import re
import time

class NFLScraper():
    def nfl_games_dk(self, half=0):
        """
        Scrapes current NFL game lines.
        Returns
        -------
        games : dictionary containing teams, spreads, totals, moneylines, home/away, and opponent.
        """
        if half == 0:
            dk_api = requests.get("https://sportsbook.draftkings.com//sites/US-NJ-SB/api/v5/eventgroups/88808?format=json").json()
            dk_markets = dk_api['eventGroup']['offerCategories'][0]['offerSubcategoryDescriptors'][0]['offerSubcategory']['offers']
        elif half == 1:
            dk_api = requests.get("https://sportsbook.draftkings.com//sites/US-NJ-SB/api/v5/eventgroups/88808/categories/526?format=json").json()
            for i in dk_api['eventGroup']['offerCategories']:
                if i['name'] == 'Halves':
                        dk_markets = i['offerSubcategoryDescriptors'][0]['offerSubcategory']['offers']
        
        games = {}
        for i in dk_markets:
            if i[0]['outcomes'][0]['oddsDecimal'] == 0: # Skip this if there is no spread
                continue
            away_team = i[0]['outcomes'][0]['label']
            home_team = i[0]['outcomes'][1]['label']
            
            if away_team not in games: 
                # Gotta be a better way then a bunch of try excepts
                games[away_team] = {'location':0}
                try:
                    games[away_team]['moneyline'] = i[2]['outcomes'][0]['oddsDecimal']
                except:
                    pass
                try:
                    games[away_team]['spread'] = [i[0]['outcomes'][0]['line'],
                                                   i[0]['outcomes'][0]['oddsDecimal']]
                except:
                    pass
                try:
                    games[away_team]['over'] = [i[1]['outcomes'][0]['line'],
                                                i[1]['outcomes'][0]['oddsDecimal']]
                except:
                    pass
                try:
                    games[away_team]['under'] = [i[1]['outcomes'][1]['line'],
                                                 i[1]['outcomes'][1]['oddsDecimal']]
                except:
                    pass
                games[away_team]['opponent'] = home_team
            
            if home_team not in games:
                games[home_team] = {'location':1}
                try:
                    games[home_team]['moneyline'] = i[2]['outcomes'][1]['oddsDecimal']
                except:
                    pass
                try:
                    games[home_team]['spread'] = [i[0]['outcomes'][1]['line'],
                                                  i[0]['outcomes'][1]['oddsDecimal']]
                except:
                    pass
                try:
                    games[home_team]['over'] = [i[1]['outcomes'][0]['line'],
                                                i[1]['outcomes'][0]['oddsDecimal']]
                except:
                    pass
                try:
                    games[home_team]['under'] = [i[1]['outcomes'][1]['line'],
                                                 i[1]['outcomes'][1]['oddsDecimal']]
                except:
                    pass     
                games[home_team]['opponent'] = away_team
                
        return games

    def nfl_props_dk(self):
        games = {}
        for cat in range(1000, 1003):
            dk_api = requests.get(f"https://sportsbook.draftkings.com//sites/US-NJ-SB/api/v4/eventgroups/88670561/categories/{cat}?format=json").json()
            for i in dk_api['eventGroup']['offerCategories']:
                if 'offerSubcategoryDescriptors' in i:
                    dk_markets = i['offerSubcategoryDescriptors']
            
            subcategoryIds = []# Need subcategoryIds first
            for i in dk_markets:
                subcategoryIds.append(i['subcategoryId'])
                        
            for ids in subcategoryIds:
                dk_api = requests.get(f"https://sportsbook.draftkings.com//sites/US-NJ-SB/api/v4/eventgroups/88670561/categories/{cat}/subcategories/{ids}?format=json").json()
                for i in dk_api['eventGroup']['offerCategories']:
                    if 'offerSubcategoryDescriptors' in i:
                        dk_markets = i['offerSubcategoryDescriptors']
                
                for i in dk_markets:
                    if 'offerSubcategory' in i:
                        market = i['name']
                        for j in i['offerSubcategory']['offers']:
                            for k in j:
                                if 'participant' in k['outcomes'][0]:
                                    player = k['outcomes'][0]['participant']
                                else:
                                    continue
                                
                                if player not in games:
                                    games[player] = {}
                                    
                                try:
                                    games[player][market] = {'over':[k['outcomes'][0]['line'],
                                                                     k['outcomes'][0]['oddsDecimal']],
                                                             'under':[k['outcomes'][1]['line'],
                                                                     k['outcomes'][1]['oddsDecimal']]}
                                except:
                                    print(player, market)
                                    pass
                
        return games

nfl_scrape = NFLScraper()
df = nfl_scrape.nfl_props_dk()
df['Brock Purdy']['Pass Yds']


# %%
import pandas as pd 
import pyarrow.parquet as pq
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

DATA_PATH = f'{root_path}/Data/OtherData/PlayerProps/'
FNAME = 'player_prop_test.parquet'
data = pq.read_table(f'{DATA_PATH}/{FNAME}').to_pandas()

data.prop = data.prop.map({'Rush Yds': 'rush_yds',
                           'Rec Yds': 'rec_yds',
                           'Rush Attempts': 'rush_att',
                           'Receptions': 'rec',
                           'Rush + Rec Yds': 'rush_rec_yds',
                           'Pass Yds': 'pass_yds',
                           'Pass TDs': 'pass_tds',
                           'Pass Completions': 'pass_comp',
                           'Pass Attempts': 'pass_att',
                           'Interceptions': 'pass_int',
                           'Pass + Rush Yds': 'pass_rush_yds'})
data.player = data.player.apply(dc.name_clean)

week = 19
year = 2022

df = dm.read(f"SELECT * FROM PFF_Proj_Ranks WHERE year={year} AND week={week}", 'Pre_PlayerData')
df = df[['player', 'passYds', 'passTd', 'passInt', 'passComp', 'passAtt', 'rushAtt', 'rushYds', 'rushTd',
         'recvReceptions', 'recvYds', 'recvTd']]
df = pd.melt(df, id_vars='player')
df = df.rename(columns={'variable': 'prop', 'value': 'proj'})

df.prop = df.prop.map({'passYds': 'pass_yds', 
                       'passTd': 'pass_td', 
                       'passInt': 'pass_int', 
                       'passComp': 'pass_comp', 
                       'passAtt': 'pass_att', 
                       'rushAtt': 'rush_att', 
                       'rushYds': 'rush_yds', 
                       'rushTd': 'rush_td',
                       'recvReceptions': 'rec', 
                       'recvYds': 'rec_yds', 
                       'recvTd': 'rec_td'})

df = pd.merge(df, data, on=['player', 'prop'])


df['pct_diff'] = (df.proj - df.line) / df.line
df.sort_values(by='pct_diff').iloc[:50]
# %%

import pandas as pd

df = pd.read_html('https://www.numberfire.com/nba/daily-fantasy/daily-basketball-projections')[3]
df.columns = [c[1] for c in df.columns]
df.Player = df.Player.apply(lambda x: ' '.join(x.split(' ')[2:4]))

# %%

import requests

requests.get(f"https://sportsbook.draftkings.com//sites/US-NJ-SB/api/v4/eventgroups?format=json").json()

# %%
