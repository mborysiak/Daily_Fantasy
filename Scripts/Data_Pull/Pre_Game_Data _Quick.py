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
import lxml
import yaml
import requests
from io import StringIO

set_year = 2024
set_week = 18

from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Assuming the config file is in the same directory as the Python script
config_file = f'{root_path}/Scripts/config.yaml'
config = read_config(config_file)


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
    try: df = df.drop('week', axis=1)
    except: pass
    df.player = df.player.apply(dc.name_clean)
    df.team = df.team.map(team_map)
    df.loc[df.position=='DST', 'player'] = df.loc[df.position=='DST', 'team']

    if table_name=='Projections': new_cols = ['player', 'position', 'team']
    elif table_name=='RawStats': new_cols = ['player', 'team', 'position']
    
    new_cols.extend([c for c in df.columns if c not in ('player', 'position', 'team')])
    df = df[new_cols]
    df = df.rename(columns={c: 'ffa_' + c for c in df.columns if c not in ('player', 'position', 'team')})

    df['week'] = set_week
    df['year'] = set_year
    return df


def pull_fftoday(pos, set_week, set_year):

    pos_ids = {
        'QB': 10,
        'RB': 20,
        'WR': 30,
        'TE': 40
    }

    num_pages = {
        'QB': [0],
        'RB': [0, 1],
        'WR': [0, 1, 2],
        'TE': [0]
        }

    cols = {
            'QB': ['player', 'team', 'opp', 'fft_pass_comp', 'fft_pass_att', 'fft_pass_yds', 'fft_pass_td',
                'fft_pass_int', 'fft_rush_att', 'fft_rush_yds', 'fft_rush_td', 'fft_proj_pts'],
            'WR': ['player', 'team', 'opp',  'fft_rush_att', 'fft_rush_yds', 'fft_rush_td', 
                   'fft_rec', 'fft_rec_yds', 'fft_rec_td', 'fft_proj_pts'],
            'RB': ['player', 'team', 'opp', 'fft_rush_att', 'fft_rush_yds', 'fft_rush_td', 
                'fft_rec', 'fft_rec_yds', 'fft_rec_td', 'fft_proj_pts'],
            'TE': ['player', 'team', 'opp', 'fft_rec','fft_rush_att', 'fft_rush_yds', 'fft_rush_td', 
                    'fft_rec_yds', 'fft_rec_td', 'fft_proj_pts']
        }

    df = pd.DataFrame()
    for page_num in num_pages[pos]:
        try:
            fft_url = f"https://www.fftoday.com/rankings/playerwkproj.php?Season={set_year}&GameWeek={set_week}&PosID={pos_ids[pos]}&LeagueID=107644&order_by=FFPts&sort_order=DESC&cur_page={page_num}"

            df_cur = pd.read_html(fft_url)[7]
            df_cur = df_cur.iloc[2:, 1:]
            df_cur.columns = cols[pos]

            df_cur = df_cur.assign(pos=pos, week=set_week, year=set_year)

            col_arr = ['player', 'pos', 'team', 'week', 'year']
            col_arr.extend([c for c in df_cur.columns if 'fft' in c])
            df_cur = df_cur[col_arr]
            
            df = pd.concat([df, df_cur], axis=0)
            
        except:
            print(pos, set_week, set_year, 'failed')
    try:
        df.player = df.player.apply(dc.name_clean)
        df.team = df.team.map(team_map)
    except:
        pass

    return df

def pull_fantasy_data(set_week, is_def=False):

    if is_def: pos = 'DEF'
    else: pos = ''

    try:
        fdta_file = [f for f in os.listdir('c:/Users/borys/Downloads') if 'fantasy-football-weekly-projections' in f][0]
        new_fname = '-'.join(fdta_file.split('-')[:-1])+pos+'.csv'
        os.rename(f'/Users/borys/Downloads/{fdta_file}', f'/Users/borys/Downloads/{new_fname}')
    except: 
        print('No new Fantasy Data file found')

    # move fantasydata projections
    df = move_download_to_folder(root_path, 'FantasyData', f'nfl-fantasy-football-weekly-projections{pos}.csv', week=set_week)
    df = df.assign(week=set_week, year=set_year).drop(['id', 'game.week'], axis=1)
    
    if is_def:
        cols = {
            'rank': 'fdta_rank',
            'team': 'player',
            'tkl_loss': 'fdta_tkl_loss',
            'def_sck': 'fdta_sack',
            'qb_hits': 'fdta_qb_hits',
            'def_int': 'fdta_def_int',
            'fum_recovered': 'fdta_fum_rec',
            'safeties': 'fdta_safeties',
            'def_td': 'fdta_def_td',
            'return_td': 'fdta_return_td',
            'opp_pts': 'fdta_opp_pts',
            'fpts_draftkings': 'fdta_proj_points'
        }

        df = df.rename(columns=cols)
        df.player = df.player.map(team_map)

    else:
        cols = {
                'rank': 'fdta_rank',
                'player': 'player', 
                'team': 'team', 
                'pos': 'position',
                'opp': 'opp',
                'pass_yds': 'fdta_pass_yds',
                'pass_td': 'fdta_pass_td',
                'pass_int': 'fdta_pass_int',
                'rush_yds': 'fdta_rush_yds',
                'rush_td': 'fdta_rush_td',
                'rec': 'fdta_rec',
                'rec_yds': 'fdta_rec_yds',
                'rec_td': 'fdta_rec_td',
                'def_sck': 'fdta_sack',
                'def_int': 'fdta_int',
                'fum_recovered': 'fdta_fum_rec',
                'fum_forced': 'fdta_fum_forced',
                'fpts_draftkings': 'fdta_proj_points',
                }
        df = df[df.pos.isin(['QB', 'RB', 'WR', 'TE', 'DST'])].reset_index(drop=True)
        df = df.rename(columns=cols)

        df.player = df.player.apply(dc.name_clean)
        df.team = df.team.map(team_map)
        df.loc[df.position=='DST', 'player'] = df.loc[df.position=='DST', 'team']
  
    return df

def check_new_data(df, table, db, where_statement=""):

    from scipy.stats import ttest_ind
    past_data = dm.read(f'''SELECT * FROM {table} {where_statement}''', db)
    num_cols = past_data.dtypes[past_data.dtypes!='object'].index

    for c in num_cols:
        if c not in ('week', 'year'):
            try:
                test = df.loc[df[c]!=0, c].dropna().values
                population = past_data.loc[past_data[c]!=0, c].dropna().values
                t_result = ttest_ind(test, population, axis=0, equal_var=False)[1]
                if t_result < 0.00001:
                    print(f'{c} Significantly Changed. T-Test Result: {t_result}')
            except:
                print(f'{c} Failed')


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

    check_new_data(df, 'FantasyPros', 'Pre_PlayerData', "WHERE pos='{set_pos}'")
    
    dm.delete_from_db('Pre_PlayerData', 'FantasyPros', f"week={set_week} and year={set_year} and pos='{set_pos}'")
    dm.write_to_db(df, 'Pre_PlayerData', 'FantasyPros', if_exist='append')

#%%

player_pos_team = dm.read(f'''SELECT player, team, pos, fp_rank
                              FROM FantasyPros
                              WHERE week={set_week} AND year={set_year}
                           ''', 'Pre_PlayerData')

extra_d = pd.DataFrame(team_map, index=[0]).T.reset_index()
extra_d.columns = ['player', 'team']
extra_d = extra_d.assign(pos='DST')
player_pos_team = pd.concat([player_pos_team, extra_d], axis=0)

print('Before Dups:', player_pos_team[player_pos_team.duplicated(subset=['player'], keep=False)])

player_pos_team = player_pos_team.sort_values(by=['player', 'team', 'pos', 'fp_rank']).reset_index(drop=True)
player_pos_team = player_pos_team.drop_duplicates(subset=['player'], keep='first').drop('fp_rank', axis=1)

print('\nAfter Dups:', player_pos_team[player_pos_team.duplicated(subset=['player'], keep=False)])

df = dm.read(f'''SELECT * 
                 FROM FantasyPros 
                 WHERE week={set_week} 
                       AND year={set_year}
                 ''', 'Pre_PlayerData')

df = pd.merge(df, player_pos_team, on=['player', 'team', 'pos'])
dm.delete_from_db('Pre_PlayerData', 'FantasyPros', f"week={set_week} and year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'FantasyPros', 'append')

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
    check_new_data(d, f'PFF_{t}_Ranks', db)
    dm.delete_from_db(db, f'PFF_{t}_Ranks', f"week={set_week} AND year={set_year}", create_backup=False)
    dm.write_to_db(d, db, f'PFF_{t}_Ranks', 'append')


#%%

matchups = dm.read(f'''SELECT offTeam, defTeam, count(*) cnts
                       FROM PFF_Proj_Ranks 
                       WHERE week='{set_week}' 
                             AND year='{set_year}'
                       GROUP BY offTeam, defTeam
                    ''', 'Pre_PlayerData')
matchups = matchups.sort_values(by='cnts', ascending=False).reset_index(drop=True).drop('cnts', axis=1)
matchups = matchups.drop_duplicates(subset=['offTeam'], keep='first')

def pff_rank_new(label_pre, label_post, folder):
    try:
        os.replace(f"/Users/borys/Downloads/{label_pre}.csv", 
                    f'{root_path}/Data/OtherData/{folder}/{set_year}/{label_post}_week{set_week}.csv')
        df = pd.read_csv(f'{root_path}/Data/OtherData/{folder}/{set_year}/{label_post}_week{set_week}.csv')

    except:
        df = pd.read_csv(f'{root_path}/Data/OtherData/{folder}/{set_year}/{label_post}_week{set_week}.csv')

    df = df.reset_index()
    df.columns = df.iloc[0,:].values
    df = df.drop(0, axis=0)

    df = df.rename(columns={
        'Full Name': 'player',
        'Team Abbreviation': 'offTeam',
        'Position Rank': 'expertConsensus',
        'Bye Week': 'Bye',
        'Projected Points': 'Proj Pts',
        'Team': 'offTeam'
    })

    for c in ['Overall Rank', 'ADP', 'Auction Value', 'Injury Status']:
        try: df = df.drop(c, axis=1)
        except: pass
    df.expertConsensus = df.expertConsensus.astype('float')
    df['expertNathanJahnke'] = df['expertConsensus']
    df.player = df.player.apply(dc.name_clean)
    df.offTeam = df.offTeam.map(team_map)
    df = df.assign(week=set_week, year=set_year)
    return df

label_pre_players = f'Week-{set_week}-rankings-export'
label_pre_teams = f'Week-{set_week}-rankings-export (1)'
label_post_players = 'expert_ranks'
label_post_teams = 'expert_ranks_teams'
folder = 'pff_rank'

rank1_pl = pff_rank_new(label_pre_players, label_post_players, folder)
rank1_tm = pff_rank_new(label_pre_teams, label_post_teams, folder)
rank1_tm.Position = 'DST'
rank1_tm = rank1_tm.rename(columns={'player': 'Name'})

all_df = pd.DataFrame()
for p in ['QB', 'RB', 'WR', 'TE']:
    base_df = rank1_pl[rank1_pl.Position==p].copy().reset_index(drop=True)

    for c in ['expertConsensus', 'expertNathanJahnke']:
        cur_df = create_adj_ranks(base_df, c, f"WHERE Position='{p}'", 'PFF_Expert_Ranks', dm)
        base_df = pd.merge(base_df, cur_df[['player', 'offTeam', f'rankadj_{c}', f'playeradj_{c}']], on=['player', 'offTeam'], how='left')
    all_df = pd.concat([all_df, base_df], axis=0)

rank1_pl = all_df.copy()

rank1_pl = pd.merge(rank1_pl, proj_pl[['player', 'fantasyPoints']], on='player', how='left')
rank1_pl = rank1_pl.rename(columns={'fantasyPoints': 'Proj Pts'})

rank1_tm = pd.merge(rank1_tm, proj_tm[['offTeam', 'fantasyPoints']], on='offTeam', how='left')
rank1_tm = rank1_tm.rename(columns={'fantasyPoints': 'Proj Pts'})

tables = ['Expert', 'Expert']
dfs = [rank1_pl, rank1_tm]
dbs = ['Pre_PlayerData', 'Pre_TeamData']

for t, d, db in zip(tables, dfs, dbs):
    d = pd.merge(d, matchups, on='offTeam', how='left')
    check_new_data(d, f'PFF_{t}_Ranks', db)
    dm.delete_from_db(db, f'PFF_{t}_Ranks', f"week={set_week} AND year={set_year}", create_backup=False)
    dm.write_to_db(d, db, f'PFF_{t}_Ranks', 'append')

#%%

df = move_download_to_folder(root_path, 'FFA', f'projections_{set_year}_wk{set_week}.csv')
df = format_ffa(df, 'Projections', set_week, set_year)
df = df[~df.team.isnull()].reset_index(drop=True)

player_pos_team = player_pos_team.rename(columns={'pos': 'position'})
df = pd.merge(df, player_pos_team, on=['player', 'team', 'position'])

check_new_data(df, 'FFA_Projections', 'Pre_PlayerData')
dm.delete_from_db('Pre_PlayerData', 'FFA_Projections', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'FFA_Projections', 'append')


df = move_download_to_folder(root_path, 'FFA', f'raw_stats_{set_year}_wk{set_week}.csv')
df = format_ffa(df, 'RawStats', set_week, set_year)
df = df[~df.team.isnull()].reset_index(drop=True)
df = pd.merge(df, player_pos_team, on=['player', 'team', 'position'])


check_new_data(df, 'FFA_RawStats', 'Pre_PlayerData')
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

check_new_data(output, 'FFToday_Projections', 'Pre_PlayerData')
dm.delete_from_db('Pre_PlayerData', 'FFToday_Projections', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(output, 'Pre_PlayerData', 'FFToday_Projections', 'append')

#%%

df = pull_fantasy_data(set_week)
player_pos_team = player_pos_team.rename(columns={'pos': 'position'})
df = pd.merge(df, player_pos_team, on=['player', 'team', 'position'])

check_new_data(df, 'FantasyData', 'Pre_PlayerData')
dm.delete_from_db('Pre_PlayerData', 'FantasyData', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'FantasyData', 'append')

#%%

df = pull_fantasy_data(set_week, is_def=True)
check_new_data(df, 'FantasyData_Defense', 'Pre_PlayerData')
dm.delete_from_db('Pre_PlayerData', 'FantasyData_Defense', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'FantasyData_Defense', 'append')

#%%

url = 'https://www.numberfire.com/nfl/fantasy/fantasy-football-projections'
response = requests.get(url, verify=False)
nf = pd.read_html(StringIO(response.text))

stats = nf[1]
stats.columns = [f"{c[0].lower()}_{c[1].lower().replace('.', '')}" for c in nf[1].columns]

players = nf[0]
players = players.T.reset_index().T.reset_index(drop=True)
players.columns = ['player']
players.player = players.player.apply(lambda x: x.split(' ')[0] + ' ' + x.split(' ')[1])
players.player = players.player.apply(dc.name_clean)

nf_data = pd.concat([players, stats], axis=1)
for c in ['opp_rank', 'ranks_ovr', 'ranks_pos']:
    nf_data[c] = nf_data[c].apply(lambda x: x.replace('#', '')).astype('int')
nf_data.columns = [f'nf_{c}' if c!='player' else c for c in nf_data.columns]
nf_data['week'] = set_week
nf_data['year'] = set_year
nf_data = nf_data.drop('nf_opp_team', axis=1)

nf_data['nf_proj_points'] = nf_data.nf_passing_yds * 0.04 + nf_data.nf_passing_tds * 4 - nf_data.nf_passing_ints * 1 + \
                          nf_data.nf_rushing_yds * 0.1 + nf_data.nf_rushing_tds * 6 + \
                          nf_data.nf_receiving_yds * 0.1 + nf_data.nf_receiving_tds * 6 + nf_data.nf_receiving_rec * 1

check_new_data(nf_data, 'NumberFire', 'Pre_PlayerData')
dm.delete_from_db('Pre_PlayerData', 'NumberFire', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(nf_data, 'Pre_PlayerData', 'NumberFire', 'append')

#%%

fname = 'Weekly DraftKings Projections'

try:
    os.replace(f"/Users/borys/Downloads/{fname}.csv", 
                f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')
    df = pd.read_csv(f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')

except:
    df = pd.read_csv(f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')

df = df.rename(columns={
    'Player': 'player', 
    'Team': 'team', 
    'Opponent': 'opp', 
    'DK Position': 'pos', 
    'DK Salary': 'dk_salary',
    'DK Projection': 'etr_proj_points',
    'DK Value': 'etr_proj_value', 
    'DK Large Ownership': 'etr_proj_large_own', 
    'DK Small Ownership': 'etr_proj_small_own',
    'DKSlateID': 'dk_slate_id', 
    'DK Floor': 'etr_proj_floor', 
    'DK Ceiling': 'etr_proj_ceiling'
})

df.player = df.player.apply(dc.name_clean)
df.team = df.team.apply(lambda x: x.replace('@', '')).map(team_map)
df.opp = df.opp.apply(lambda x: x.replace('@', '')).map(team_map)

df['week'] = set_week
df['year'] = set_year

check_new_data(df, 'ETR_Projections_DK', 'Pre_PlayerData')
dm.delete_from_db('Pre_PlayerData', 'ETR_Projections_DK', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'ETR_Projections_DK', 'append')

#%%

fname = 'Weekly Fantasy Projections'

try:
    os.replace(f"/Users/borys/Downloads/{fname}.csv", 
                f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')
    df = pd.read_csv(f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')

except:
    df = pd.read_csv(f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')

df = df.rename(columns={
    'Player': 'player', 
    'Team': 'team', 
    'Position': 'pos', 
    'DK': 'etr_proj_points',
    'FD': 'etr_proj_points_fd', 
    'Std': 'etr_proj_points_std', 
    'Half': 'etr_proj_points_half',
    'Full': 'dk_slate_id', 
    'DK Ceiling': 'etr_proj_ceiling',
    'FD Ceiling': 'etr_proj_ceiling_fd'
})

df
df.player = df.player.apply(dc.name_clean)
df.team = df.team.map(team_map)

df['week'] = set_week
df['year'] = set_year

check_new_data(df, 'ETR_Projections', 'Pre_PlayerData')
dm.delete_from_db('Pre_PlayerData', 'ETR_Projections', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'ETR_Projections', 'append')

#%%

fname = 'DFS Main Slate Ownership - DraftKings and FanDuel'

try:
    os.replace(f"/Users/borys/Downloads/{fname}.csv", 
                f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')
    df = pd.read_csv(f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')

except:
    df = pd.read_csv(f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')


df = df.rename(columns={
    'Player': 'player', 
    'Team': 'team', 
    'Position': 'pos', 
    'Opponent': 'opp',
    'DK Salary': 'dk_salary', 
    'DK Small': 'etr_small_own',
    'DK Medium': 'etr_medium_own', 
    'DK Large': 'etr_large_own', 
    'FD Salary': 'fd_salary', 
    'FD Large': 'etr_large_own_fd',
}).fillna('0%')

for c in ['etr_small_own', 'etr_medium_own', 'etr_large_own', 'etr_large_own_fd']:
    df[c] = df[c].apply(lambda x: x.replace('%', '')).astype('float')

df.player = df.player.apply(dc.name_clean)
df.team = df.team.map(team_map)

df['week'] = set_week
df['year'] = set_year

dm.delete_from_db('Pre_PlayerData', 'ETR_Ownership', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'ETR_Ownership', 'append')

#%%

fname = 'Weekly Projections Detail'

try:
    os.replace(f"/Users/borys/Downloads/{fname}.csv", 
                f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')
    df = pd.read_csv(f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')

except:
    df = pd.read_csv(f'{root_path}/Data/OtherData/ETR/{set_year}/{fname}_week{set_week}.csv')

df = df.rename(columns={
    'Player': 'player', 
    'Team': 'team', 
    'Position': 'pos', 
    'Opponent': 'opp',
    'Completions': 'etr_pass_cmp', 
    'Attempts': 'etr_pass_att',
    'Pass Yards': 'etr_pass_yds', 
    'Pass TDs': 'etr_pass_tds', 
    'Pass INTs': 'etr_pass_int', 
    'Carries': 'etr_rush_att', 
    'Rush Yards': 'etr_rush_yds',
    'Rush TDs': 'etr_rush_tds', 
    'Receptions': 'etr_rec', 
    'Receiving Yards': 'etr_rec_yds', 
    'Receiving TDs': 'etr_rec_td',
}).fillna(0)

df.player = df.player.apply(dc.name_clean)
df.team = df.team.map(team_map)

df['week'] = set_week
df['year'] = set_year

check_new_data(df, 'ETR_Projections_Detail', 'Pre_PlayerData')
dm.delete_from_db('Pre_PlayerData', 'ETR_Projections_Detail', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'ETR_Projections_Detail', 'append')


#%%


fpts_fname = '2024 NFL Fantasy Football Rankings  Projections QB, RB, WR, TE  Fantasy Points'
try:
    os.replace(f"/Users/borys/Downloads/{fpts_fname}.csv", 
                f'{root_path}/Data/OtherData/FantasyPoints/{set_year}/{fpts_fname}_week{set_week}.csv')
    df = pd.read_csv(f'{root_path}/Data/OtherData/FantasyPoints/{set_year}/{fpts_fname}_week{set_week}.csv')

except:
    df = pd.read_csv(f'{root_path}/Data/OtherData/FantasyPoints/{set_year}/{fpts_fname}_week{set_week}.csv')

df = df.rename(columns={
    'RK': 'fpts_overall_rank', 
    'Name': 'player', 
    'POS': 'pos', 
    'Team': 'team', 
    'Opp':  'opp', 
    'FPTS': 'fpts_proj_points', 
    'ATT': 'fpts_pass_att', 
    'CMP': 'fpts_pass_cmp', 
    'YDS': 'fpts_pass_yds', 
    'TD': 'fpts_pass_td',
    'INT': 'fpts_pass_int', 
    'ATT.1': 'fpts_rush_att', 
    'YDS.1': 'fpts_rush_yds', 
    'TD.1': 'fpts_rush_td', 
    'REC': 'fpts_rec', 
    'YDS.2': 'fpts_rec_yds', 
    'TD.2': 'fpts_rec_td', 
    'UP': 'fpts_up', 
    'DOWN': 'fpts_down',
    'MOVE': 'fpts_move', 
    'WW': 'fpts_ww', 
    'INJ': 'fpts_inj',
})

df.player = df.player.apply(dc.name_clean)
df.team = df.team.map(team_map)
df.opp = df.opp.map(team_map)
df = df.assign(week=set_week, year=set_year)

for c in df.columns:
    try: df[c] = df[c].apply(lambda x: x.replace('-', '0')).astype('float')
    except: pass

df['fpts_pass_yds_per_att'] = df.fpts_pass_yds / df.fpts_pass_att
df['fpts_pass_yds_per_cmp'] = df.fpts_pass_yds / df.fpts_pass_cmp
df['fpts_rush_yds_per_att'] = df.fpts_rush_yds / df.fpts_rush_att
df['fpts_rec_yds_per_rec'] = df.fpts_rec_yds / df.fpts_rec

dm.delete_from_db('Pre_PlayerData', 'FantasyPoints', f"week={set_week} AND year={set_year}", create_backup=False)
dm.write_to_db(df, 'Pre_PlayerData', 'FantasyPoints', 'append')

#%%


fpts_fname = '2024 NFL Fantasy Football Rankings  Projections QB, RB, WR, TE  Fantasy Points (1)'
try:
    os.replace(f"/Users/borys/Downloads/{fpts_fname}.csv", 
                f'{root_path}/Data/OtherData/FantasyPoints/{set_year}/Def_{fpts_fname}_week{set_week}.csv')
    df = pd.read_csv(f'{root_path}/Data/OtherData/FantasyPoints/{set_year}/Def_{fpts_fname}_week{set_week}.csv')

except:
    df = pd.read_csv(f'{root_path}/Data/OtherData/FantasyPoints/{set_year}/Def_{fpts_fname}_week{set_week}.csv')


df = df.rename(columns={
    'RK': 'fpts_overall_rank', 
    'Name': 'player', 
    'POS': 'pos', 
    'Team': 'team', 
    'Opp':  'opp', 
    'FPTS': 'fpts_proj_points', 
    'SACK': 'fpts_sack', 
    'INT': 'fpts_int', 
    'FR': 'fpts_fum_rec', 
    'DTD': 'fpts_def_td',
    'STD': 'fpts_special_td',
    'UP': 'fpts_up', 
    'DOWN': 'fpts_down',
    'MOVE': 'fpts_move', 
    'WW': 'fpts_ww', 
    'INJ': 'fpts_inj',
})

df.player = df.team
df.player = df.player.map(team_map)
df.team = df.team.map(team_map)
df.opp = df.opp.map(team_map)
df = df.assign(week=set_week, year=set_year)

for c in df.columns:
    try: df[c] = df[c].apply(lambda x: x.replace('-', '0')).astype('float')
    except: pass

dm.delete_from_db('Pre_PlayerData', 'FantasyPoints_Defense', f"week={set_week} AND year={set_year}")
dm.write_to_db(df, 'Pre_PlayerData', 'FantasyPoints_Defense', 'append')

#%%

pull_historical = True
base_url = 'https://api.the-odds-api.com/v4/'

month = 12
day = 15

odds_api = OddsAPIPull(set_week, set_year, api_key, base_url, sport, region, odds_format, date_format, historical=pull_historical)

start_time = dt.datetime(set_year, month, day, 5, 0, 0)
end_time = (start_time + dt.timedelta(hours=3*24))

# events_df = odds_api.pull_events(start_time=start_time, end_time=end_time)
# event_ids = tuple(events_df.event_id.unique()) + (0,)
# print(events_df)

# stats = [
#     'player_pass_attempts', 'player_pass_completions', 'player_pass_interceptions', 'player_pass_longest_completion', 
#         'player_pass_rush_reception_tds', 'player_pass_rush_reception_yds', 'player_pass_tds', 'player_pass_yds', 'player_receptions',
#         'player_receptions', 'player_reception_longest', 'player_reception_yds', 'player_rush_attempts', 'player_rush_longest',
#         'player_rush_reception_tds', 'player_rush_reception_yds', 'player_rush_yds','player_1st_td', 'player_anytime_td', 
#         'player_tds_over', 'player_sacks'
#         ]

# markets = ','.join(stats)
# player_props = odds_api.all_market_odds(markets, events_df)
# print(player_props.head())
# event_ids = tuple(player_props.event_id.unique()) + (0,)

# dm.delete_from_db('Pre_PlayerData', 'Game_Odds', f'''week={set_week} and year={set_year} and event_id IN {event_ids}  
#                                                         and prop_type IN {tuple(stats)}''', create_backup=False)
# dm.write_to_db(player_props, 'Pre_PlayerData', 'Game_Odds', 'append')

#%%

pull_historical = False

base_url = 'https://api.the-odds-api.com/v4/'
odds_api = OddsAPIPull(set_week, set_year, api_key, base_url, sport, region, odds_format, date_format, historical=pull_historical)

start_time = dt.datetime.now()
end_time = (start_time + dt.timedelta(hours=5*24))

events_df = odds_api.pull_events(start_time=start_time, end_time=end_time)
event_ids = tuple(events_df.event_id.unique()) + (0,)
events_df

#%%

dm.delete_from_db('Pre_TeamData', 'Game_Events', f"week={set_week} and year={set_year} and event_id IN {event_ids}", create_backup=False)
dm.write_to_db(events_df, 'Pre_TeamData', 'Game_Events', 'append')

#%%

stats = [
        'player_pass_attempts', 'player_pass_completions', 'player_pass_interceptions', 'player_pass_longest_completion', 
         'player_pass_rush_reception_tds', 'player_pass_rush_reception_yds', 'player_pass_tds', 'player_pass_yds', 'player_receptions',
         'player_receptions', 'player_reception_longest', 'player_reception_yds', 'player_rush_attempts', 'player_rush_longest',
         'player_rush_reception_tds', 'player_rush_reception_yds', 'player_rush_yds','player_1st_td', 'player_anytime_td', 
         'player_tds_over', 'player_sacks'
         ]

markets = ','.join(stats)
player_props = odds_api.all_market_odds(markets, events_df)
print(player_props)

stats = ['spreads', 'h2h', 'totals']
markets = ','.join(stats)
team_props = odds_api.all_market_odds(markets, events_df)
print(team_props)

#%%
event_ids = tuple(team_props.event_id.unique()) + (0,)
dm.delete_from_db('Pre_TeamData', 'Game_Odds', f"week={set_week} and year={set_year} and event_id IN {event_ids}", create_backup=False)
dm.write_to_db(team_props, 'Pre_TeamData', 'Game_Odds', 'append')

event_ids = tuple(player_props.event_id.unique()) + (0,)
dm.delete_from_db('Pre_PlayerData', 'Game_Odds', f"week={set_week} and year={set_year} and event_id IN {event_ids}", create_backup=False)
dm.write_to_db(player_props, 'Pre_PlayerData', 'Game_Odds', 'append')

for pos in ['QB', 'RB', 'WR', 'TE']:
    df = get_all_vegas_stats(pos, set_week, set_year)
    df = df.dropna(thresh=df.shape[1]*0.5).reset_index(drop=True)
    dm.write_to_db(df, 'Pre_PlayerData', f'Vegas_Clean_{pos}', 'replace')

#%%
def home_away_format(home_away):

    spread = dm.read(f'''
                        SELECT event_id, description, ROUND(AVG(point),1) as {home_away}_line
                        FROM Game_Odds 
                        WHERE prop_type='spreads' 
                                AND week={set_week}
                                AND year={set_year}  
                        GROUP BY event_id, description
                        ''', 'Pre_TeamData')
    
    moneyline = dm.read(f'''
                        SELECT event_id, description, ROUND(AVG(price),2) as {home_away}_moneyline
                        FROM Game_Odds 
                        WHERE prop_type='h2h' 
                                AND week={set_week}
                                AND year={set_year}  
                        GROUP BY event_id, description
                        ''', 'Pre_TeamData')
    
    over_under = dm.read(f'''
                        SELECT event_id, ROUND(AVG(point),1) as over_under
                        FROM Game_odds 
                        WHERE prop_type='totals' 
                                AND week={set_week}
                                AND year={set_year}  
                                AND name='Over'
                        GROUP BY event_id
                        ''', 'Pre_TeamData')

    events = dm.read(f'''
                        SELECT event_id, commence_time, {home_away}_team description
                        FROM Game_Events
                        WHERE week={set_week}
                              AND year={set_year} 
                        ''', 'Pre_TeamData')

    spread = pd.merge(spread, events, on=['event_id', 'description'])
    spread = pd.merge(spread, moneyline, on=['event_id', 'description'])
    spread = pd.merge(spread, over_under, on=['event_id'])

    spread = spread.rename(columns={'description': f'{home_away}_team'})

    return spread


home_spread = home_away_format('home')
away_spread = home_away_format('away')

team_lines = pd.merge(home_spread, away_spread.drop(['commence_time', 'over_under'], axis=1), on='event_id')
team_lines = team_lines[['away_team', 'home_team', 'away_line', 'home_line', 'away_moneyline', 'home_moneyline', 'over_under', 'commence_time']]
team_lines['gametime'] = (
    pd.to_datetime(team_lines['commence_time'], utc=True)   # parse as UTC
      .dt.tz_convert('US/Eastern')              # convert to US/Eastern
      .dt.tz_localize(None)                     # drop the tz info
)
team_lines['gametime_unix'] = team_lines['gametime'].view(np.int64) // 10**9
team_lines = team_lines.drop('commence_time', axis=1).sort_values(by='gametime').reset_index(drop=True)
team_lines = team_lines.assign(year=set_year, week=set_week)
team_lines.away_team = team_lines.away_team.map(full_team_map)
team_lines.home_team = team_lines.home_team.map(full_team_map)

teams = list(team_lines.away_team)
teams.extend(list(team_lines.home_team))
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
dm.write_to_db(team_lines, 'Pre_TeamData', 'Gambling_Lines', if_exist='append')

glines = dm.read("SELECT * FROM Gambling_Lines", 'Pre_TeamData')
dm.write_to_db(glines, 'Simulation', 'Gambling_Lines', 'replace')


#%%


dk_sal = pd.read_html(StringIO(requests.get('https://www.fantasypros.com/daily-fantasy/nfl/draftkings-salary-changes.php', verify=False).text))[0]
fd_sal = pd.read_html(StringIO(requests.get('https://www.fantasypros.com/daily-fantasy/nfl/fanduel-salary-changes.php', verify=False).text))[0]
yahoo_sal = pd.read_html(StringIO(requests.get('https://www.fantasypros.com/daily-fantasy/nfl/yahoo-salary-changes.php', verify=False).text))[0]

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

accu = config['accu_api_key']
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
