#%%
from numpy.core.numeric import full
import pandas as pd
import os
import zipfile
import numpy as np
from ff import data_clean as dc
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_pinball_loss, mean_squared_error
import matplotlib.pyplot as plt

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
set_week = 14

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

    full_entries = df[['Rank', 'Points', 'Lineup', 'week', 'year']].dropna().reset_index(drop=True)

    player_ownership = df[['Player', 'Roster Position','%Drafted', 'FPTS', 'week', 'year']].dropna().reset_index(drop=True)
    player_ownership.columns = ['player', 'player_position', 'pct_drafted', 'player_points', 'week', 'year']
    player_ownership.pct_drafted = player_ownership.pct_drafted.apply(lambda x: float(x.replace('%', '')))
    player_ownership.player = player_ownership.player.apply(dc.name_clean)
    player_ownership = player_ownership.drop('player_position', axis=1)

    return full_entries, player_ownership



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
                        WHERE week < 17''', 'Pre_PlayerData')

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


def run_model_alpha(val_predict, test_predict, alpha, time_split):

    pipe = skm.model_pipe([skm.column_transform(num_pipe=[skm.piece('impute')], 
                                                cat_pipe=[skm.piece('one_hot')]), 
                        ('gbm', GradientBoostingRegressor(loss="quantile", alpha=alpha))
    ])

    params = skm.default_params(pipe)

    pinball_loss_alpha = make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False)
    cv_time = skm.cv_time_splits('week', X, time_split)

    best_model = skm.random_search(pipe, X, y, params, scoring=pinball_loss_alpha, cv=cv_time, n_iter=20)
    print(skm.cv_score(best_model, X, y, scoring=pinball_loss_alpha))

    predictions = skm.cv_predict_time(best_model, X, y, cv_time=cv_time)
    val_predict[f'Perc{int(alpha*100)}'] = predictions
    test_predict[f'Perc{int(alpha*100)}'] = best_model.predict(X_test)

    return val_predict, test_predict


def run_model_mean(val_predict, test_predict, time_split):
    
    pipe = skm.model_pipe([skm.column_transform(num_pipe=[skm.piece('impute')], 
                                                    cat_pipe=[skm.piece('one_hot')]), 
                        skm.piece('gbm')])

    params = skm.default_params(pipe)

    cv_time = skm.cv_time_splits('week', X, time_split)
    best_model = skm.random_search(pipe, X, y, params,  cv=cv_time, n_iter=20)
    print(np.sqrt(-skm.cv_score(best_model, X, y)))
    
    predictions = skm.cv_predict_time(best_model, X, y, cv_time=cv_time)
    val_predict[f'MeanPred'] = predictions
    test_predict[f'MeanPred'] = best_model.predict(X_test)

    return val_predict, test_predict

#%%

all_data = read_in_csv(extract_path)
full_entries, player_ownership = entries_ownership(all_data)

df = add_proj(player_ownership)
drop_list = ['Dalvin Cook32021']
df = drop_player_weeks(df, drop_list)
df = add_injuries(df)
df = feature_engineering(df)

df_train = df[df.week < set_week].reset_index(drop=True)
df_test = df[df.week >= set_week].reset_index(drop=True)

skm = SciKitModel(df_train)
X, y = skm.Xy_split('pct_drafted', to_drop=['player', 'team'])

skm_test = SciKitModel(df_test)
X_test, y_test = skm_test.Xy_split('pct_drafted', to_drop=['player', 'team'])

val_predict = {}
test_predict = {}
time_split = 4
for alpha in [0.01, 0.16, 0.84, 0.99]:
    print(f'Running alpha {int(alpha*100)}')
    val_predict, test_predict = run_model_alpha(val_predict, test_predict, alpha, time_split)

val_predict, test_predict = run_model_mean(val_predict, test_predict, time_split)


val_df = pd.DataFrame(val_predict, index=range(len(val_predict['MeanPred'])))
val_labels = df_train.loc[df_train.week >= time_split, ['player', 'week', 'year', 'pct_drafted']].reset_index(drop=True)
val_df = pd.concat([val_labels, val_df], axis=1)
val_df['PredDiff'] = val_df.pct_drafted - val_df.MeanPred
val_df = val_df.drop('pct_drafted', axis=1)

test_df = pd.DataFrame(test_predict, index=range(len(test_predict['MeanPred'])))
test_labels = df_test[['player', 'week', 'year']].reset_index(drop=True)
test_df = pd.concat([test_labels, test_df], axis=1)


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

    from scipy.stats import gaussian_kde

    # # Generate fake data
    # x = total_drafted.place
    # y = total_drafted[base_place]

    # # Calculate the point density
    # xy = np.vstack([x,y])
    # z = gaussian_kde(xy)(xy)

    # fig, ax = plt.subplots()
    # ax.scatter(x, y, c=z, s=20)
    # plt.show()



# evaluate_metrics(best_results, 'QB', 'SumPred')
# %%

for base_place, places in zip([0, 50000, 100000, 150000], [250, 1000, 1000, 1000]):
    print(base_place)

    player_pos = dm.read('''SELECT DISTINCT player, pos FROM FantasyPros''', 'Pre_PlayerData')
    df_lineups = format_lineups(min_place=base_place, max_place=base_place+places)
    # df_lineups = pd.merge(df_lineups, player_ownership, on=['player', 'week', 'year'])
    # df = pd.merge(df_lineups, val_df, on=['player', 'week', 'year'])
    # evaluate_metrics(df, 'WR', 'SumPred', base_place)

    df_lineups = pd.merge(df_lineups, player_pos, on='player')
    print(df_lineups.loc[df_lineups.lineup_position=='FLEX', 'pos'].value_counts() / \
        df_lineups.loc[df_lineups.lineup_position=='FLEX', 'pos'].shape[0])
    

# %%
