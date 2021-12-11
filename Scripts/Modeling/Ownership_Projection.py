#%%
from numpy.core.numeric import full
import pandas as pd
import os
import zipfile
import numpy as np
download_path = '//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/Mborysiak/DK/'
extract_path = download_path + 'Results/'

set_year = 2021

csv_files = [f for f in os.listdir(extract_path)]
df = pd.DataFrame()
for f in csv_files:
    cur_df = pd.read_csv(extract_path+f, low_memory=False)
    cur_df['week'] = int(f.replace('.csv', '').replace('week', ''))
    cur_df['year'] = set_year
    df = pd.concat([df, cur_df], axis=0)

# %%

full_entries = df[['Rank', 'Points', 'Lineup', 'week', 'year']].dropna().reset_index(drop=True)
player_ownership = df[['Player', 'Roster Position','%Drafted', 'FPTS', 'week', 'year']].dropna().reset_index(drop=True)
player_ownership.columns = ['player', 'player_position', 'pct_drafted', 'points', 'week', 'year']
# %%

def extract_players(lineup):
    positions = ['QB', 'RB', 'WR', 'TE', 'DST', 'FLEX']
    for p in positions:
        lineup = lineup.replace(p, ',')
    lineup = lineup.split(',')[1:]
    lineup = [p.rstrip().lstrip() for p in lineup]

    return lineup

def extract_positions(lineup):
    positions = ('QB', 'RB', 'WR', 'TE', 'DST', 'FLEX')
    lineup = lineup.split(' ')
    lineup = [p for p in lineup if p in positions]
    return lineup

def clean_lineup_df(players, positions, row):
    
    clean_lineup = pd.DataFrame([positions, players]).T
    clean_lineup.columns = ['lineup_position', 'player']
    clean_lineup = clean_lineup.assign(place=row.Rank, points=row.Points, week=row.week, year=row.year)
    return clean_lineup

N = 50
best_lineups = full_entries[full_entries.Rank <= N].copy().reset_index(drop=True)

best_results = pd.DataFrame()
for _, row in best_lineups.iterrows():
    players = extract_players(full_entries.Lineup[0])
    positions = extract_positions(full_entries.Lineup[0])
    clean_lineup = clean_lineup_df(players, positions, row)
    best_results = pd.concat([best_results, clean_lineup], axis=0)
# %%


# %%
