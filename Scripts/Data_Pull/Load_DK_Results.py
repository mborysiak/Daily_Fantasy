#%%
import pandas as pd
import os
from ff import data_clean as dc

from ff.db_operations import DataManage   
import ff.general as ffgeneral 

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

set_year = 2022
set_week = 2

download_path = '//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/Mborysiak/DK/'
extract_path = download_path + f'Results/{set_year}/'


def read_in_csv(extract_path, contest, set_week, set_year):

    df = pd.read_csv(f'{extract_path}/{contest}/week{set_week}.csv', low_memory=False)
    df['week'] = set_week
    df['year'] = set_year

    return df


def entries_ownership(df):

    df.Player = df.Player.fillna('Missing').reset_index(drop=True)
    df.Player = df.Player.apply(dc.name_clean)

    full_entries = df[['Rank', 'Points', 'Lineup', 'week', 'year']].dropna().reset_index(drop=True)
    full_entries = full_entries.sort_values(by=['year', 'week', 'Points'], 
                                            ascending=[True, True, False]).reset_index(drop=True)

    player_ownership = df[['Player', 'Roster Position','%Drafted', 'FPTS', 'week', 'year']].dropna().reset_index(drop=True)
    player_ownership.columns = ['player', 'player_position', 'pct_drafted', 'player_points', 'week', 'year']
    player_ownership.pct_drafted = player_ownership.pct_drafted.apply(lambda x: float(x.replace('%', '')))
    player_ownership.player = player_ownership.player.apply(dc.name_clean)
    player_ownership = player_ownership.drop('player_position', axis=1)
    # player_ownership = player_ownership.sort_values(by=['year', 'week', 'player_points'],
    #                                                 ascending=[True, True, False]).reset_index(drop=True)
    

    return full_entries, player_ownership


def add_pct_rank(full_entries):
    
    full_entries['Rank'] = full_entries.groupby(['year', 'week']).cumcount()+1
    total_entries = full_entries.groupby(['year','week']).agg(TotalLineups=('Lineup', 'count')).reset_index()
    full_entries = pd.merge(full_entries, total_entries, on=['year', 'week'])
    full_entries['PctRank'] = full_entries.Rank / full_entries.TotalLineups
    
    return full_entries

def get_prizes(contest):

    prizes_choice = {
        'Million': pd.DataFrame([[1,1000000],
                                 [1,125000],
                                [1,75000],
                                [1,50000],
                                [1,30000],
                                [1,20000],
                                [2,15000],
                                [2,10000],
                                [5,7000],
                                [5,5000],
                                [10,3500],
                                [10,2500],
                                [20,1500],
                                [30,1000],
                                [35,750],
                                [50,600],
                                [50,500],
                                [75,400],
                                [100,300],
                                [150,250],
                                [200,200],
                                [300,150],
                                [450,125],
                                [750,100],
                                [1250,80],
                                [3500,60],
                                [9000,40],
                                [30000,30]]),
    
        'ScreenPass': pd.DataFrame([[1,5000],
                                    [1,2500],
                                    [1,1500],
                                    [1,1000],
                                    [1,750],
                                    [2,600],
                                    [3,500],
                                    [3,400],
                                    [3,300],
                                    [4,250],
                                    [4,200],
                                    [5,150],
                                    [10,100],
                                    [20,75],
                                    [40,60],
                                    [75,50],
                                    [150,40],
                                    [580,30],
                                ]),

         'ThreePointStance': pd.DataFrame([[1,25000],
                                           [1,15000],
                                           [1,10000],
                                           [1,7500],
                                           [1,5000],
                                           [1,3500],
                                           [2,2500],
                                           [2,1500],
                                           [5,1000],
                                           [5,750],
                                           [10,500],
                                           [10,400],
                                           [10,300],
                                           [25,250],
                                           [25,200],
                                           [50,150],
                                           [100,125],
                                           [150,100],
                                           [300, 80],
                                           [500,65],
                                           [1050,50]
                                ])
    } 

    num_entries = {
        'Million': 206000, 
        'ScreenPass': 3921, 
        'ThreePointStance': 8834
        }
    
    prizes = prizes_choice[contest]                       
    prizes.columns = ['num_finishes', 'prize']
    prizes['Rank'] = prizes.num_finishes.cumsum()
    prizes['PctRank'] = prizes.Rank / num_entries[contest]
    return prizes, num_entries[contest]

#%%

for contest in ['Million']:#, 'ThreePointStance', 'ScreenPass']:
    dk_data = read_in_csv(extract_path, contest, set_week, set_year)
    full_entries, player_ownership = entries_ownership(dk_data)
    prizes, num_entries = get_prizes(contest)
    
    if num_entries < full_entries.shape[0]:
        full_entries = full_entries.sample(num_entries).sort_values(by='Points', ascending=False).reset_index(drop=True)

    full_entries = add_pct_rank(full_entries)

    full_entries['prize'] = 0
    for i in range(len(prizes)):
        if i == 0: 
            full_entries.loc[full_entries.Rank==1, 'prize'] = prizes.loc[prizes.Rank==1, 'prize']
        else:
            score_i = prizes.iloc[i]
            score_i_minus = prizes.iloc[i-1]
            if prizes.loc[i, 'Rank'] < 10:
                full_entries.loc[(full_entries.Rank > score_i_minus.Rank) & \
                                (full_entries.Rank <= score_i.Rank), 'prize'] = score_i.prize
            else:
                full_entries.loc[(full_entries.PctRank > score_i_minus.PctRank) & \
                                (full_entries.PctRank <= score_i.PctRank), 'prize'] = score_i.prize

    full_entries['Contest'] = contest
    player_ownership['Contest'] = contest

    dm.delete_from_db('DK_Results', 'Contest_Results', f"week={set_week} AND year={set_year} AND Contest='{contest}'")
    dm.delete_from_db('DK_Results', 'Contest_Ownership', f"week={set_week} AND year={set_year} AND Contest='{contest}'")

    dm.write_to_db(full_entries, 'DK_Results', 'Contest_Results', 'append')
    dm.write_to_db(player_ownership, 'DK_Results', 'Contest_Ownership', 'append')

# full_entries

#%%

df = dm.read("SELECT * FROM Million_Results", 'DK_Results')
df['Contest'] = 'Million'
dm.write_to_db(df, 'DK_Results', 'Contest_Results', 'replace')
# %%
df = dm.read("SELECT * FROM Million_Ownership", 'DK_Results')
df['Contest'] = 'Million'
dm.write_to_db(df, 'DK_Results', 'Contest_Ownership', 'replace')
# %%
