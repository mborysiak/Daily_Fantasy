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

output = pd.DataFrame()
for stat_type, url in urls.items():
    data = pd.read_html(url)
    cur_df = cleanup_data(data, stat_type)
    output = pd.concat([output, cur_df], axis=0)
    time.sleep(3)
# %%
pd.read_html('https://sportsbook.draftkings.com/leagues/football/nfl?category=rush/rec-props&subcategory=rush-yds', 
             match= 'sportsbook-table')

# %%
import requests
from bs4 import BeautifulSoup

url = 'https://sportsbook.draftkings.com/leagues/football/nfl?category=rush/rec-props&subcategory=rush-yds'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
dfs = pd.read_html(page.text)
# %%
soup.find_all('table')
# %%
pd.read_html('https://www.scoresandodds.com/nfl/props')
# %%
import requests

url = 'https://www.actionnetwork.com/nfl/props'

header = {
  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
  "X-Requested-With": "XMLHttpRequest"
}

r = requests.get(url, headers=header)

dfs = pd.read_html(r.text)
# %%
