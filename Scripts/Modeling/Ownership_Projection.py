#%%
import pandas as pd
import numpy as np
from ff import data_clean as dc
import datetime as dt

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
from skmodel import SciKitModel

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

set_year = 2021
set_week = 13

#%%

def get_best_lineups(full_entries, min_place, max_place):

    best_lineups = full_entries[(full_entries.Rank >= min_place) & (full_entries.Rank <= max_place)].copy().reset_index(drop=True)
    best_lineups = best_lineups.sort_values(by=['year', 'week', 'Points'], ascending=[True, True, False]).reset_index(drop=True)
    # best_lineups['Rank'] = best_lineups.groupby(['year', 'week']).cumcount()

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



def format_lineups(full_entries, min_place, max_place):

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

    proj['avg_proj_pts'] = proj[['projected_points', 'fantasyPoints', 'ProjPts']].mean(axis=1)

    proj.pos = proj.pos.apply(lambda x: x.upper())
    df = pd.merge(df, proj, on=['player', 'week', 'year'])

    return df

#%%

full_entries = dm.read(f"SELECT * FROM Million_Results WHERE year={set_year}", 'DK_Results')
player_ownership = dm.read(f"SELECT * FROM Million_Ownership WHERE year={set_year}", 'DK_Results')


for base_place, places in zip([1, 25000, 50000, 100000, 150000], [10, 1000, 1000, 1000, 1000]):
    print(f'\nPlaces {base_place}-{places}\n==================')

    df_lineups = format_lineups(full_entries, min_place=base_place, max_place=base_place+places)
    df_lineups.player = df_lineups.player.apply(dc.name_clean)

    player_pos = dm.read('''SELECT DISTINCT player, pos FROM FantasyPros''', 'Pre_PlayerData')
    df_lineups = pd.merge(df_lineups, player_pos, on='player')

    # print the flex numbers by position
    print('Flex Pct by Position\n',
           df_lineups.loc[df_lineups.lineup_position=='FLEX', 'pos'].value_counts() / \
           df_lineups.loc[df_lineups.lineup_position=='FLEX', 'pos'].shape[0])

    # print the number of players from the same team
    teams = dm.read('''SELECT * FROM Player_Teams''', 'Simulation')
    df_lineups = pd.merge(df_lineups, teams, on=['player'])

    team_cnts = df_lineups[df_lineups.pos!='DST'].groupby(['place', 'team', 'week', 'year']).agg(player_cnts=('player', 'count')).reset_index()
    max_cnts = team_cnts.groupby(['place', 'week', 'year']).agg(player_cnts=('player_cnts', 'max')).reset_index()
    team_cnts = pd.merge(team_cnts, max_cnts, on=['place', 'week','year', 'player_cnts'])
    print('Number of players on same team:\n', 
          team_cnts.player_cnts.value_counts() / team_cnts.shape[0])

    df_lineups = pd.merge(df_lineups, player_ownership, on=['player', 'week', 'year'], how='left')
    df_lineups = add_proj(df_lineups)

    drafted_pct = df_lineups.groupby(['place', 'week']).agg({'pct_drafted': 'sum', 'lineup_position': 'count', 'ProjPts': 'sum'})
    drafted_pct['pct_drafted'] = drafted_pct.pct_drafted / (drafted_pct.lineup_position / 9)
    drafted_pct['ProjPts'] = drafted_pct.ProjPts / (drafted_pct.lineup_position / 9)
    print('\nAvg Pct Drafted:', np.mean(drafted_pct.pct_drafted), 
            '\nStd Perc Drafted:', np.std(drafted_pct.pct_drafted),
            '\nAvg Proj Pts:', np.mean(drafted_pct.ProjPts))

    
#%%



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

    df = df.sort_values(by=['team', 'pos', 'week', 'avg_proj_pts']).reset_index(drop=True)
    df['team_pos_week_order'] = df.groupby(['team', 'pos', 'week']).cumcount().values

    df = df.sort_values(by=['pos', 'week', 'avg_proj_pts']).reset_index(drop=True)
    df['pos_week_order'] = df.groupby(['pos', 'week']).cumcount().values

    df = df.sort_values(by=['year', 'week']).reset_index(drop=True)

    return df

def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))

def create_game_date(df, val_year_min, val_week_min, year_week_to_date):
            
    # set up the date column for sorting
    df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
    cv_time_input = int(dt.datetime(val_year_min, 1, val_week_min).strftime('%Y%m%d'))
    train_time_split = int(dt.datetime(set_year, 1, set_week).strftime('%Y%m%d'))

    return df, cv_time_input, train_time_split


def run_model_alpha(val_predict, test_predict, X_test, alpha, time_split):

    skm = SciKitModel(df_train, model_obj='quantile')

    # get the model pipe for stacking setup and train it on meta features
    pipe = skm.model_pipe([
                           skm.piece('k_best'),
                           skm.piece('gbm_q')
                        ])

    params = skm.default_params(pipe)
    pipe.steps[-1][-1].alpha = alpha
    
    # run the model with parameter search
    best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, 
                                                    params, n_iter=25,
                                                    bayes_rand='custom_rand',
                                                    col_split='game_date',
                                                    time_split=time_split)

    val_predict[f'Perc{int(alpha*100)}'] = oof_data['full_hold'].pred

    predictions = pd.DataFrame()
    for bm in best_models:
        predictions = pd.concat([predictions, pd.Series(bm.fit(X,y).predict(X_test))], axis=1)
    
    test_predict[f'Perc{int(alpha*100)}'] = predictions.mean(axis=1)

    return val_predict, test_predict


def run_model_mean(test_predict, X_test, time_split):
    
    skm = SciKitModel(df_train, model_obj='reg')

    # get the model pipe for stacking setup and train it on meta features
    pipe = skm.model_pipe([
                           skm.piece('k_best'),
                           skm.piece('rf')
                        ])

    params = skm.default_params(pipe)
    
    # run the model with parameter search
    best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, 
                                                 params, n_iter=25,
                                                 bayes_rand='custom_rand',
                                                 col_split='game_date',
                                                 time_split=time_split)

    val_predict = oof_data['full_hold'].copy()

    predictions = []
    for bm in best_models:
        predictions.append(bm.fit(X, y).predict(X_test))

    test_predict['pred_ownership'] = pd.DataFrame(predictions).T.mean(axis=1)

    return val_predict, test_predict

#%%
#================
# Predict Ownership Pct
#================

player_ownership = dm.read("SELECT * FROM Million_Ownership", 'DK_Results')

val_week_min = 8
val_year_min = 2021

df = add_proj(player_ownership)
drop_list = ['Dalvin Cook32021', 'Calvin Ridley82021', 'Odell Beckham152021', 'Cooper Kupp152021', 'Van Jefferson152021',
             "D'Andre Swift112021", 'Josh Johnson162021', 'Kyler Murray92021', 'Darren Waller72021',
             'Kyler Murray 132021']
df = drop_player_weeks(df, drop_list)
df = add_injuries(df)
df = feature_engineering(df)
df = df.rename(columns={'pct_drafted': 'y_act'})

for c in ['pos', 'practice_status', 'game_status', 'practice_game']:
    df = pd.concat([df, pd.get_dummies(df[c], drop_first=True)], axis=1).drop(c, axis=1)

df, cv_time_input, train_time_split = create_game_date(df, val_year_min, val_week_min, year_week_to_date)

df_train = df[df.game_date < train_time_split].reset_index(drop=True)
df_test = df[df.game_date == train_time_split].reset_index(drop=True)

skm = SciKitModel(df_train)
X, y = skm.Xy_split('y_act', to_drop=['player', 'team'])

skm_test = SciKitModel(df_test)
X_test, y_test = skm_test.Xy_split('y_act', to_drop=['player', 'team'])

test_predict = df_test[['player', 'team', 'week', 'year']].copy()
val_predict, test_predict = run_model_mean(test_predict, X_test, cv_time_input)
val_predict = val_predict.rename(columns={'pred': 'pred_ownership'})

# for alpha in [0.01, 0.16, 0.84, 0.99]:

#     print(f'\n===============Running alpha {int(alpha*100)}\n================')
#     val_predict, test_predict = run_model_alpha(val_predict, test_predict, X_test, alpha, cv_time_input)


#%%

# from Fix_Standard_Dev import *

sd_spline, max_spline, min_spline = get_std_splines(val_predict, {'pred_ownership': 1}, 
                                                    show_plot=True, k=2, 
                                                    min_grps_den=int(val_predict.shape[0]*0.2), 
                                                    max_grps_den=int(val_predict.shape[0]*0.05))

sc = StandardScaler().fit(val_predict[['pred_ownership']])
val_predict['std_dev'] = sd_spline(sc.transform(val_predict[['pred_ownership']]))
val_predict['max_score'] = max_spline(sc.transform(val_predict[['pred_ownership']]))

for i in [3, 2, 1]:
    val_predict['std_dev_test'] = i*val_predict.std_dev
    val_predict.loc[val_predict.std_dev < 0, 'std_dev_test'] = 1
    val_predict['upper_range'] = val_predict.pred_ownership + val_predict.std_dev_test
    val_predict['lower_range'] = val_predict.pred_ownership - val_predict.std_dev_test

    print(f'Num samples within {i} std dev:',
        val_predict[(val_predict.y_act > val_predict.lower_range) & (val_predict.y_act < val_predict.upper_range)].shape[0] / \
            val_predict.shape[0])

print('Num Samples Greater than Max:',
      val_predict[(val_predict.y_act > val_predict.Perc99)].shape[0] / \
            val_predict.shape[0])

print('Num Samples Less than Min:',
      val_predict[(val_predict.y_act < val_predict.Perc1)].shape[0] / \
            val_predict.shape[0])

test_predict['std_dev'] = sd_spline(sc.transform(test_predict[['pred_ownership']]))
test_predict['max_score'] = max_spline(sc.transform(test_predict[['pred_ownership']]))
test_predict['min_score'] = 0
test_predict

#%%

try:
    test_predict['y_act'] = df_test.y_act
    test_predict.plot.scatter(x='pred_ownership', y='y_act')
    # test_predict = test_predict.drop('y_act', axis=1)
except:
    pass


test_predict.loc[test_predict.min_score < 0, 'min_score'] = 0 
test_predict.loc[test_predict.max_score < test_predict.pred_ownership, 'max_score'] = \
    test_predict.loc[test_predict.max_score < test_predict.pred_ownership, 'pred_ownership'] * 1.5
test_predict.loc[test_predict.std_dev < 0, 'std_dev'] = 1

# test_predict = test_predict[['player', 'team', 'week', 'year', 'pred_ownership', 'std_dev', 'min_score', 'max_score']]

#%%

dm.delete_from_db('Simulation', 'Predicted_Ownership', f'week={set_week} AND year={set_year}')
dm.write_to_db(test_predict, 'Simulation', 'Predicted_Ownership', 'append')

dm.delete_from_db('Simulation', 'Predicted_Ownership_Validation', f'week={set_week} AND year={set_year}')
dm.write_to_db(val_predict, 'Simulation', 'Predicted_Ownership_Validation', 'append')


# %%
from ff.db_operations import DataManage   
import ff.general as ffgeneral 
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


def create_sd_max_metrics(df, metrics):

    sc = StandardScaler()

    cols = list(metrics.keys())
    wts = list(metrics.values())
    sc_metrics = pd.DataFrame(sc.fit_transform(df[cols]), columns=cols) * wts

    df['sd_metric'] = sc_metrics.mean(axis=1)
    df['max_metric'] = sc_metrics.mean(axis=1)
    df['min_metric'] = sc_metrics.mean(axis=1)

    df = df[[c for c in df.columns if c not in sc_metrics]]

    return df


def create_groups(df, num_grps):
    # create equal sizes groups going down the dataframe ordered by each metric
    df_len = len(df)
    repeats = math.ceil(df.shape[0] / num_grps)
    grps = np.repeat([i for i in range(num_grps)], repeats)
    df['grps'] = grps[:df_len]
    return df

def show_spline_fit(splines, met, X, y, X_max, y_max):
    print(met)
    X_pred = list(np.arange(np.min(X.values), np.max(X.values), 0.1))
    plt.scatter(X, y)
    plt.scatter(X_max[met], y_max[met])
    plt.plot(X_pred, splines[met](X_pred), 'g', lw=3)
    plt.show()

def get_std_splines(df, metrics, show_plot=False, k=2, s=2000, min_grps_den=100, max_grps_den=60):
    
    
    all_cols = list(set(list(metrics.keys()) + ['player', 'year', 'y_act']))
    df = df[all_cols].dropna().reset_index(drop=True)

    # calculate sd and max metrics
    df = create_sd_max_metrics(df, metrics)

    # create the groups    
    df = df.dropna()
    min_grps = int(df.shape[0] / min_grps_den)
    max_grps = int(df.shape[0] / max_grps_den)

    splines = {}; X_max = {}; y_max = {}; max_r2 = {}
    for x_val, met in zip(['sd_metric', 'max_metric', 'min_metric'], ['std_dev', 'perc_99', 'perc_1']):
        
        df = df.sort_values(by=x_val).reset_index(drop=True)

        max_r2[met] = 0
        for num_grps in range(min_grps, max_grps, 1):

            # create the groups to aggregate for std dev and max metrics
            df = create_groups(df, num_grps)

            # calculate the standard deviation and max of each group
            Xy = df.groupby('grps').agg({'y_act': [np.std, lambda x: np.percentile(x, 99), lambda x: np.percentile(x, 1)],
                                         'sd_metric': 'max',
                                         'max_metric': 'max',
                                         'min_metric': 'max',
                                         'player': 'count'})
            Xy.columns = ['std_dev', 'perc_99', 'perc_1', 'sd_metric', 'max_metric', 'min_metric', 'player_cnts']
            
            # fit a spline to the group datasets
            X = Xy[x_val]
            y = Xy[met]
            print(X, y)

            spl = UnivariateSpline(X, y, k=k, s=s)

            r2 = r2_score(y, spl(X))
            if r2 > max_r2[met]:
                max_r2[met] = r2
                splines[met] = spl
                X_max[met] = X
                y_max[met] = y

        if show_plot:
            show_spline_fit(splines, met, X, y, X_max, y_max)
            
    return splines['std_dev'], splines['perc_99'], splines['perc_1']

get_std_splines(val_predict, {'pred': 1}, 
                                                    show_plot=True, k=2, 
                                                    min_grps_den=int(val_predict.shape[0]*0.2), 
                                                    max_grps_den=int(val_predict.shape[0]*0.1))
# %%

