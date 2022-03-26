#%%
from numpy.core.numeric import full
import pandas as pd
import os
import zipfile
import numpy as np
from ff import data_clean as dc
import matplotlib.pyplot as plt
import datetime as dt

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
from skmodel import SciKitModel

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

download_path = '//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/Mborysiak/DK/'
extract_path = download_path + 'Results/'

set_year = 2021
set_week = 18

def read_in_csv(extract_path):

    csv_files = [f for f in os.listdir(extract_path)]
    df = pd.DataFrame()
    for f in csv_files:
        cur_df = pd.read_csv(extract_path+f, low_memory=False)
        cur_df['week'] = int(f.replace('.csv', '').replace('week', ''))
        cur_df['year'] = set_year
        df = pd.concat([df, cur_df], axis=0)

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

def get_prizes():
    prizes = pd.DataFrame([[1,1000000],
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
                            [30000,30]])
                            
    prizes.columns = ['num_finishes', 'prize']
    prizes['Rank'] = prizes.num_finishes.cumsum()
    prizes['PctRank'] = prizes.Rank / 206000
    return prizes

#%%
all_data = read_in_csv(extract_path)
full_entries, player_ownership = entries_ownership(all_data)
full_entries = add_pct_rank(full_entries)
prizes = get_prizes()

full_entries['prize'] = 0
for i in range(len(prizes)):
    if i == 0: 
        full_entries.loc[full_entries.Rank==1, 'prize'] = 1000000
    else:
        score_i = prizes.iloc[i]
        score_i_minus = prizes.iloc[i-1]
        if prizes.loc[i, 'Rank'] < 10:
            full_entries.loc[(full_entries.Rank > score_i_minus.Rank) & \
                            (full_entries.Rank <= score_i.Rank), 'prize'] = score_i.prize
        else:
            full_entries.loc[(full_entries.PctRank > score_i_minus.PctRank) & \
                            (full_entries.PctRank <= score_i.PctRank), 'prize'] = score_i.prize

#%%
dm.write_to_db(full_entries, 'DK_Results', 'Million_Results', 'replace')
dm.write_to_db(player_ownership, 'DK_Results', 'Million_Ownership', 'replace')

# %%

def get_best_lineups(full_entries, min_place, max_place):

    best_lineups = full_entries[(full_entries.Rank >= min_place) & (full_entries.Rank <= max_place)].copy().reset_index(drop=True)
    best_lineups = best_lineups.sort_values(by=['year', 'week', 'Points'], ascending=[True, True, False]).reset_index(drop=True)
    best_lineups['Rank'] = best_lineups.groupby(['year', 'week']).cumcount()

    return best_lineups


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



def format_lineups(min_place, max_place):

    best_lineups = get_best_lineups(full_entries, min_place=min_place, max_place=max_place)

    players = [extract_players(l) for l in best_lineups.Lineup.values]
    positions = [extract_positions(l) for l in best_lineups.Lineup.values]

    players = pd.DataFrame(players)
    positions = pd.DataFrame(positions)

    df = pd.concat([players, positions], axis=1)
    df = pd.concat([df, best_lineups.drop('Lineup', axis=1)], axis=1)

    final_df = pd.DataFrame()
    for i in range(9):
        tmp_df = df[[i, 'Rank', 'Points', 'week', 'year']]
        tmp_df.columns = ['player', 'lineup_position', 'place', 'team_points', 'week', 'year']
        final_df = pd.concat([final_df, tmp_df], axis=0)
  
    return final_df




#%%

def evaluate_metrics(best_results, set_pos, metric, base_place):

    if set_pos is not None:
        total_drafted = best_results[best_results.lineup_position==set_pos]
    else:
        total_drafted = best_results.copy()

    total_drafted = total_drafted.groupby(['place', 'week', 'year']).agg(SumPred=('MeanPred', 'sum'),
                                                                        AvgPred=('MeanPred', 'mean'),
                                                                        MaxPred=('MeanPred', 'max'),
                                                                        MinPred=('MeanPred', 'min'),
                                                                        lineup_cnts=('MeanPred', 'count')).reset_index()
    # total_drafted = total_drafted[total_drafted.lineup_cnts == 8 ]
    total_drafted = total_drafted.rename(columns={metric:base_place})
    total_drafted[base_place].plot.hist(alpha=0.5, legend=True)

    print(f'''
    16th percentile: {round(np.percentile(total_drafted[base_place], 16), 1)},
    Mean: {round(np.mean(total_drafted[base_place]), 1)},
    84th percentile: {round(np.percentile(total_drafted[base_place], 84), 1)},
    ''')


# %%

for base_place, places in zip([0, 50000, 100000, 150000], [25, 1000, 1000, 1000]):
    print(base_place)

    player_pos = dm.read('''SELECT DISTINCT player, pos FROM FantasyPros''', 'Pre_PlayerData')
    df_lineups = format_lineups(min_place=base_place, max_place=base_place+places)
    df_lineups.player = df_lineups.player.apply(dc.name_clean)

    # df_lineups = pd.merge(df_lineups, player_ownership, on=['player', 'week', 'year'])
    # df = pd.merge(df_lineups, val_df, on=['player', 'week', 'year'])
    # evaluate_metrics(df, 'WR', 'SumPred', base_place)

    df_lineups = pd.merge(df_lineups, player_pos, on='player')
    print(df_lineups.loc[df_lineups.lineup_position=='FLEX', 'pos'].value_counts() / \
        df_lineups.loc[df_lineups.lineup_position=='FLEX', 'pos'].shape[0])
    

# %%

full_entries, player_ownership = entries_ownership(all_data)

#%%
df = format_lineups(min_place=0, max_place=10)
df.player = df.player.apply(dc.name_clean)
teams = dm.read('''SELECT * FROM Player_Teams''', 'Simulation')
player_pos = dm.read('''SELECT DISTINCT player, pos FROM FantasyPros''', 'Pre_PlayerData')

df = pd.merge(df, teams, on='player')
df = pd.merge(df, player_pos, on=['player'])

team_cnts = df.groupby(['place', 'team', 'week', 'year']).agg(player_cnts=('player', 'count')).reset_index()
max_cnts = team_cnts.groupby(['place', 'week', 'year']).agg(player_cnts=('player_cnts', 'max')).reset_index()
team_cnts = pd.merge(team_cnts, max_cnts, on=['place', 'week','year', 'player_cnts'])
team_cnts.player_cnts.value_counts() / team_cnts.shape[0]
# %%

pos_cnts = pd.merge(df, team_cnts, on=['place', 'week', 'year', 'team'])
pos_cnts.pos.value_counts() / pos_cnts.shape[0]

# %%
#%%

# calculate ownership projections
def add_proj(df):

    # pull in the salary and actual results data
    proj = dm.read('''SELECT player, position pos, offTeam team, week, year, dk_salary, projected_points, fantasyPoints, ProjPts
                    FROM PFF_Proj_Ranks
                    JOIN (SELECT player, team offTeam, week, year, projected_points 
                            FROM FantasyPros)
                            USING (player, offTeam, week, year)
                    JOIN (SELECT player, offTeam, week, year, `Proj Pts` ProjPts 
                            FROM PFF_Expert_Ranks)
                            USING (player, offTeam, week, year)
                    JOIN (SELECT player, team offTeam, week, year, dk_salary 
                            FROM Daily_Salaries) USING (player, offTeam, week, year)
                ''', 'Pre_PlayerData')

    proj.pos = proj.pos.apply(lambda x: x.upper())
    df = pd.merge(proj, player_ownership.drop(['player_points'], axis=1), on=['player', 'week', 'year'])

    return df

def drop_player_weeks(df, drop_list):
    df['to_drop'] = df.player + df.week.astype('str') + df.year.astype('str')
    df = df[~df.to_drop.isin(drop_list)].reset_index(drop=True).drop('to_drop', axis=1)
    return df

def add_injuries(df):

    inj = dm.read('''SELECT player, pos, week, year, 
                            practice_status, game_status, 
                            practice_status||game_status practice_game
                    FROM PlayerInjuries
                    WHERE game_status != 'None' ''', 'Pre_PlayerData')
    df = pd.merge(df, inj, on=['player', 'pos', 'week', 'year'], how='left')
    df[['practice_status', 'game_status', 'practice_game']] = df[['practice_status', 'game_status', 'practice_game']].fillna('Healthy')

    return df

def feature_engineering(df):

    for c in ['projected_points', 'fantasyPoints', 'ProjPts']:
        df[c+'_over_sal'] = df[c] / df.dk_salary

    team_pts = df.groupby(['team', 'week', 'year']).agg(team_projected_points=('projected_points', 'sum')).reset_index()
    df = pd.merge(df, team_pts, on=['team','week', 'year'])

    return df

def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))

def create_game_date(df, val_year_min, val_week_min, year_week_to_date):
            
    # set up the date column for sorting
    df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
    cv_time_input = int(dt.datetime(val_year_min, 1, val_week_min).strftime('%Y%m%d'))
    train_time_split = int(dt.datetime(set_year, 1, set_week).strftime('%Y%m%d'))

    return df, cv_time_input, train_time_split


def run_model_alpha(val_predict, test_predict, alpha, time_split):

    skm = SciKitModel(df_train, model_obj='quantile')

    # get the model pipe for stacking setup and train it on meta features
    pipe = skm.model_pipe([
                           skm.piece('random_sample'),
                           skm.piece('gbm_q')
                        ])

    params = skm.default_params(pipe)
    params['random_sample__frac'] = np.arange(0.2, 1, 0.05)
    pipe.steps[-1][-1].alpha = alpha
    
    # run the model with parameter search
    best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, 
                                                             params, n_iter=25,
                                                             bayes_rand='custom_rand',
                                                             col_split='game_date',
                                                             time_split=time_split)

    val_predict[f'Perc{int(alpha*100)}'] = oof_data['full_hold']

    predictions = pd.DataFrame()
    for bm in best_models:
        predictions = pd.concat([predictions, pd.Series(bm.predict(X_test))], axis=1)
    
    test_predict[f'Perc{int(alpha*100)}'] = predictions.mean(axis=1)

    return val_predict, test_predict


def run_model_mean(val_predict, test_predict, time_split):
    
    skm = SciKitModel(df_train, model_obj='reg')

    # get the model pipe for stacking setup and train it on meta features
    pipe = skm.model_pipe([
                           skm.piece('k_best'),
                           skm.piece('lgbm')
                        ])

    params = skm.default_params(pipe)
    
    # run the model with parameter search
    best_model, oof_data, _ = skm.time_series_cv(pipe, X, y, 
                                                 params, n_iter=25,
                                                 bayes_rand='custom_rand',
                                                 col_split='game_date',
                                                 time_split=time_split)

    val_predict['MeanPred'] = oof_data['full_hold']
    test_predict['MeanPred'] = best_model.predict(X_test)


    return val_predict, test_predict

#%%
all_data = read_in_csv(extract_path)
full_entries, player_ownership = entries_ownership(all_data)

#%%

val_week_min = 8
val_year_min = 2021

df = add_proj(player_ownership)
drop_list = ['Dalvin Cook32021']
df = drop_player_weeks(df, drop_list)
df = add_injuries(df)
df = feature_engineering(df)
df = df.rename(columns={'pct_drafted': 'y_act'})

for c in ['pos', 'practice_status', 'game_status', 'practice_game']:
    df = pd.concat([df, pd.get_dummies(df[c], drop_first=True)], axis=1).drop(c, axis=1)

df, cv_time_input, train_time_split = create_game_date(df, val_year_min, val_week_min, year_week_to_date)

df_train = df[df.game_date < train_time_split].reset_index(drop=True)
df_test = df[df.game_date >= train_time_split].reset_index(drop=True)

skm = SciKitModel(df_train)
X, y = skm.Xy_split('y_act', to_drop=['player', 'team'])

skm_test = SciKitModel(df_test)
X_test, y_test = skm_test.Xy_split('y_act', to_drop=['player', 'team'])

val_predict = {}
test_predict = {}
for alpha in [0.01, 0.16, 0.84, 0.99]:
    print(f'\n===============Running alpha {int(alpha*100)}\n================')
    val_predict, test_predict = run_model_alpha(val_predict, test_predict, alpha, cv_time_input)

#%%
# val_predict, test_predict = run_model_mean(val_predict, test_predict, time_split)


i = 0
for k, v in val_predict.items():
    if i==0: val_df = v.rename(columns={'pred': k})
    else: val_df = pd.merge(val_df, v.rename(columns={'pred': k}), on=['player', 'team', 'week', 'year', 'y_act'])
    i+=1
val_df[val_df.y_act >= val_df.Perc84]

# test_df = pd.DataFrame(test_predict, index=range(len(test_predict['MeanPred'])))
# test_labels = df_test[['player', 'week', 'year']].reset_index(drop=True)
# test_df = pd.concat([test_labels, test_df], axis=1)

# %%
