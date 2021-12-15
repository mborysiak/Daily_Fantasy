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

download_path = '//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/Mborysiak/DK/'
extract_path = download_path + 'Results/'

set_year = 2021
set_week = 14

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
player_ownership.columns = ['player', 'player_position', 'pct_drafted', 'player_points', 'week', 'year']
player_ownership.pct_drafted = player_ownership.pct_drafted.apply(lambda x: float(x.replace('%', '')))
player_ownership.player = player_ownership.player.apply(dc.name_clean)

# %%

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
from skmodel import SciKitModel

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

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
    df = pd.merge(proj, player_ownership.drop(['player_position','player_points'], axis=1), on=['player', 'week', 'year'])

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

#%%

test_predict = {}
for alpha in [0.01, 0.16, 0.84, 0.99]:

    print(f'Running alpha {int(alpha*100)}')
    
    pipe = skm.model_pipe([skm.column_transform(num_pipe=[skm.piece('impute')], 
                                                cat_pipe=[skm.piece('one_hot')]), 
                        ('gbm', GradientBoostingRegressor(loss="quantile", alpha=alpha))
    ])

    params = skm.default_params(pipe)

    pinball_loss_alpha = make_scorer(mean_pinball_loss, alpha=alpha, greater_is_better=False)
    best_model = skm.random_search(pipe, X, y, params, scoring=pinball_loss_alpha)
    print(skm.cv_score(best_model, X, y, scoring=pinball_loss_alpha))

    predictions = skm.cv_predict(best_model, X, y, cv=8)
    plt.scatter(predictions, y)

    test_predict[f'Perc{int(alpha*100)}'] = best_model.predict(X_test)


pipe = skm.model_pipe([skm.column_transform(num_pipe=[skm.piece('impute')], 
                                                cat_pipe=[skm.piece('one_hot')]), 
                       skm.piece('gbm')])

params = skm.default_params(pipe)
best_model = skm.random_search(pipe, X, y, params)
print(np.sqrt(-skm.cv_score(best_model, X, y)))
test_predict[f'MeanPred'] = best_model.predict(X_test)
predictions = skm.cv_predict(best_model, X, y, cv=8)
plt.scatter(predictions, y)

train_predict_df = pd.concat([df_train[['player', 'week', 'year']], 
                             pd.Series(predictions, name='MeanPred')], axis=1)
                            
#%%
test_predict_df = pd.DataFrame(test_predict, index=range(len(test_predict['Perc99'])))
test_predict_df = pd.concat([df_test[['player', 'week', 'year']], test_predict_df, pd.Series(y_test, name='actual')], axis=1)

import numpy as np
test_predict_df['above'] = np.where(test_predict_df.actual > test_predict_df.Perc99, 1, 0)
test_predict_df['below'] = np.where(test_predict_df.actual < test_predict_df.Perc1, 1, 0)

test_predict_df[['above', 'below']].sum() / len(test_predict_df)

# %%

test_predict_df.sort_values(by='actual', ascending=False).iloc[:50]

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

def clean_lineup_df(players, positions):
    
    clean_lineup = pd.DataFrame([positions, players]).T
    clean_lineup.columns = ['lineup_position', 'player']
    return clean_lineup

def add_other_info(clean_lineup, rnk, pts, wk, yr):
    return clean_lineup.assign(place=rnk, team_points=pts, week=wk, year=yr)

def create_best_lineups(df_to_join, min_place, max_place):

    best_lineups = get_best_lineups(full_entries, min_place=min_place, max_place=max_place)
    best_results = pd.DataFrame()
    iter_df = best_lineups[['Lineup', 'Rank', 'Points', 'week', 'year']].values
    for lineup, rnk, pts, wk, yr in iter_df:
        players = extract_players(lineup)
        positions = extract_positions(lineup)
        clean_lineup = clean_lineup_df(players, positions)
        clean_lineup = add_other_info(clean_lineup, rnk, pts, wk, yr)
        best_results = pd.concat([best_results, clean_lineup], axis=0)

    best_results.player = best_results.player.apply(dc.name_clean)
    best_results = pd.merge(best_results, df_to_join, on=['player', 'week', 'year'])

    return best_results


def evaluate_metrics(best_results, set_pos, metric):
    total_drafted = best_results[best_results.lineup_position==set_pos]
    total_drafted = total_drafted.groupby(['place', 'week', 'year']).agg(SumPred=('MeanPred', 'sum'),
                                                                        AvgPred=('MeanPred', 'mean'),
                                                                        MaxPred=('MeanPred', 'max'),
                                                                        MinPred=('MeanPred', 'min'),
                                                                        lineup_cnts=('MeanPred', 'count')).reset_index()
    # total_drafted = total_drafted[total_drafted.lineup_cnts == 8 ]
    total_drafted[metric].plot.hist()

    print(f'''
    16th percentile: {round(np.percentile(total_drafted[metric], 16), 1)},
    Mean: {round(np.mean(total_drafted[metric]), 1)},
    84th percentile: {round(np.percentile(total_drafted[metric], 84), 1)},
    ''')

    from scipy.stats import gaussian_kde

    # Generate fake data
    x = total_drafted.place
    y = total_drafted[metric]

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=20)
    plt.show()

#%%
base_place=0
best_results = create_best_lineups(train_predict_df, min_place=base_place, max_place=base_place+200)
evaluate_metrics(best_results, 'QB', 'SumPred')
# %%

base_place=0

player_pos = dm.read('''SELECT DISTINCT player, pos FROM FantasyPros''', 'Pre_PlayerData')
best_results = create_best_lineups(player_ownership, min_place=base_place, max_place=base_place+500)
best_results = pd.merge(best_results, player_pos, on='player')
cnts = best_results.loc[(best_results.lineup_position=='FLEX'), 'pos'].value_counts()
cnts / cnts.sum()

# Top lineups tend to have higher owned WR on sum, max basis
# %%

# %%
