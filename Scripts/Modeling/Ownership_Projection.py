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
set_week = 18

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

    df = df.sort_values(by=['team', 'pos', 'year', 'week', 'avg_proj_pts']).reset_index(drop=True)
    df['team_pos_week_order'] = df.groupby(['team', 'pos', 'week', 'year']).cumcount().values

    df = df.sort_values(by=['pos', 'year', 'week', 'avg_proj_pts']).reset_index(drop=True)
    df['pos_week_order'] = df.groupby(['pos', 'year', 'week']).cumcount().values

    avg_per_sal = df.groupby(['pos', 'year', 'week']).agg(avg_pts_per_sal_pos=('ProjPts_over_sal', 'mean'))
    df = pd.merge(df, avg_per_sal, on=['pos', 'year', 'week'])
    df['avg_pts_per_sal_diff'] = df.ProjPts_over_sal - df.avg_pts_per_sal_pos

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


def run_model_mean(m, test_predict, X_test, time_split):
    
    print(f"\n===========Running {m}=============\n")
    skm = SciKitModel(df_train, model_obj='reg')

    # get the model pipe for stacking setup and train it on meta features
    pipe = skm.model_pipe([
                           skm.piece('k_best'),
                           skm.piece(m)
                        ])

    params = skm.default_params(pipe)
    
    # run the model with parameter search
    best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, 
                                                 params, n_iter=25,
                                                 bayes_rand='custom_rand',
                                                 col_split='game_date',
                                                 time_split=time_split)

    val_predict = oof_data['full_hold'].copy()
    val_predict = val_predict.rename(columns={'pred': f'pred_ownership_{m}'})

    predictions = []
    for bm in best_models:
        predictions.append(bm.fit(X, y).predict(X_test))

    test_predict[f'pred_ownership_{m}'] = pd.DataFrame(predictions).T.mean(axis=1)

    return val_predict, test_predict


def drop_teams(data):

    df = dm.read(f'''SELECT week, year, away_team, home_team, gametime
                    FROM Gambling_Lines 
                    WHERE year >= 2021
            ''', 'Pre_TeamData')
    df.gametime = pd.to_datetime(df.gametime)
    df['day_of_week'] = df.gametime.apply(lambda x: x.weekday())
    df['hour_in_day'] = df.gametime.apply(lambda x: x.hour)
    df = df[(df.day_of_week!=6) | (df.hour_in_day > 16)]
    drop_teams = df[['week','year', 'away_team']].rename(columns={'away_team': 'home_team'})
    drop_teams = pd.concat([drop_teams, df[['week','year', 'home_team']]], axis=0)
    drop_teams['to_drop'] = 1
    drop_teams = drop_teams.rename(columns={'home_team': 'team'})

    
    data = pd.merge(data, drop_teams, on=['team', 'week', 'year'], how='left')
    data = data[data.to_drop.isnull()].reset_index(drop=True).drop('to_drop', axis=1)

    return data

#%%
#================
# Predict Ownership Pct
#================

for set_week in [11, 12, 13, 14, 15, 16, 17, 18]:

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
    df_test.y_act = np.log(df_test.y_act/100)

    df_train = df_train[df_train.y_act!=0].reset_index(drop=True)
    df_train.y_act = np.log(df_train.y_act/100)

    skm = SciKitModel(df_train)
    X, y = skm.Xy_split('y_act', to_drop=['player', 'team'])

    skm_test = SciKitModel(df_test)
    X_test, y_test = skm_test.Xy_split('y_act', to_drop=['player', 'team'])

    test_predict = df_test[['player', 'team', 'week', 'year']].copy()
    val_predict_gbm, test_predict = run_model_mean('gbm', test_predict, X_test, cv_time_input)
    val_predict_rf, test_predict = run_model_mean('rf', test_predict, X_test, cv_time_input)

    val_predict = pd.merge(val_predict_gbm, val_predict_rf.drop('y_act', axis=1), on=['player', 'team', 'week', 'year'])
    val_predict['pred_ownership'] = val_predict[[c for c in val_predict.columns if 'pred' in c]].mean(axis=1)
    test_predict['pred_ownership'] = test_predict[[c for c in test_predict.columns if 'pred' in c]].mean(axis=1)

    # test_predict3 = df_test[['player', 'team', 'week', 'year']].copy()
    # val_predict3, test_predict3 = run_model_mean('ada', test_predict3, X_test, cv_time_input)
    # for alpha in [0.01, 0.16, 0.84, 0.99]:

    #     print(f'\n===============Running alpha {int(alpha*100)}\n================')
    #     val_predict, test_predict = run_model_alpha(val_predict, test_predict, X_test, alpha, cv_time_input)


    from Fix_Standard_Dev import *
    sd_m, max_m, min_m = get_std_splines(val_predict, {'pred_ownership': 1}, 
                                                        show_plot=True, k=2, 
                                                        min_grps_den=int(val_predict.shape[0]*0.15), 
                                                        max_grps_den=int(val_predict.shape[0]*0.05),
                                                        iso_spline='spline')

    sc = StandardScaler().fit(val_predict[['pred_ownership']])
    sd_max_val = sc.transform(val_predict[['pred_ownership']])
    val_predict['std_dev'] = sd_m(sd_max_val)
    val_predict['max_score'] = max_m(sd_max_val)
    val_predict['min_score'] = min_m(sd_max_val)

    for i in [3, 2, 1]:
        val_predict['std_dev_test'] = i*val_predict.std_dev
        val_predict.loc[val_predict.std_dev < 0, 'std_dev_test'] = 1
        val_predict['upper_range'] = val_predict.pred_ownership + val_predict.std_dev_test
        val_predict['lower_range'] = val_predict.pred_ownership - val_predict.std_dev_test

        print(f'Num samples within {i} std dev:',
            val_predict[(val_predict.y_act > val_predict.lower_range) & (val_predict.y_act < val_predict.upper_range)].shape[0] / \
                val_predict.shape[0])

    print('Num Samples Greater than Max:',
        val_predict[(val_predict.y_act > val_predict.max_score)].shape[0] / \
                val_predict.shape[0])

    sd_max_test = sc.transform(test_predict[['pred_ownership']])

    test_predict['std_dev'] = sd_m(sd_max_test)
    test_predict['max_score'] = max_m(sd_max_test)
    test_predict['min_score'] = min_m(sd_max_test)

    try:
        test_predict['y_act'] = df_test.y_act
        test_predict.plot.scatter(x='pred_ownership', y='y_act')
        test_predict = test_predict[['player', 'team', 'week', 'year', 'pred_ownership', 'y_act', 'std_dev', 'min_score', 'max_score']]
        display(test_predict.sort_values(by='pred_ownership', ascending=False).iloc[:50])
        test_predict = test_predict.drop('y_act', axis=1)
    except:
        pass
        test_predict = test_predict[['player', 'team', 'week', 'year', 'pred_ownership', 'std_dev', 'min_score', 'max_score']]
        display(test_predict.sort_values(by='pred_ownership', ascending=False).iloc[:50])

    test_predict.loc[test_predict.min_score < 0, 'min_score'] = 0 
    test_predict.loc[test_predict.max_score < test_predict.pred_ownership, 'max_score'] = \
        test_predict.loc[test_predict.max_score < test_predict.pred_ownership, 'pred_ownership'] * 1.5
    test_predict.loc[test_predict.std_dev < 0, 'std_dev'] = 1


    dm.delete_from_db('Simulation', 'Predicted_Ownership', f'week={set_week} AND year={set_year}')
    dm.write_to_db(test_predict, 'Simulation', 'Predicted_Ownership', 'append')

    val_predict = val_predict[['player', 'team', 'week', 'year', 'pred_ownership', 'std_dev', 'min_score', 'max_score']]
    # dm.delete_from_db('Simulation', 'Predicted_Ownership_Validation', f'week={set_week} AND year={set_year}')
    dm.write_to_db(val_predict, 'Simulation', 'Predicted_Ownership_Validation', 'replace')


#%%

full_entries = dm.read(f"SELECT * FROM Million_Results WHERE year={set_year}", 'DK_Results')
player_ownership = dm.read(f"SELECT * FROM Million_Ownership WHERE year={set_year}", 'DK_Results')
pred_player_ownership = dm.read(f'''SELECT player, 
                                           week, 
                                           year, 
                                           AVG(pred_ownership) pred_ownership, 
                                           AVG(std_dev) std_dev 
                                    FROM Predicted_Ownership_Validation 
                                    WHERE year={set_year}
                                    GROUP BY player, week, year
                                    ''', 'Simulation')
pred_player_ownership = add_proj(pred_player_ownership)

#%%

for base_place, places in zip([1, 25000, 50000, 100000, 150000], [25, 1000, 1000, 1000, 1000]):
    print(f'\nPlaces {base_place}-{places+base_place}\n==================')

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

    team_cnts = df_lineups[df_lineups.pos.isin(['QB', 'WR', 'TE'])].groupby(['place', 'team', 'week', 'year']).agg(player_cnts=('player', 'count')).reset_index()
    max_cnts = team_cnts.groupby(['place', 'week', 'year']).agg(player_cnts=('player_cnts', 'max')).reset_index()
    team_cnts = pd.merge(team_cnts, max_cnts, on=['place', 'week','year', 'player_cnts'])
    print('Number of players on same team:\n', 
          team_cnts.player_cnts.value_counts() / team_cnts.shape[0])

    df_lineups = pd.merge(df_lineups, player_ownership, on=['player', 'week', 'year'], how='left')
    df_lineups = pd.merge(df_lineups, pred_player_ownership, on=['player', 'week', 'year'])

    # df_lineups.pred_ownership = df_lineups.pred_ownership / df_lineups.avg_proj_pts
    df_lineups['pct_drafted'] = df_lineups.pct_drafted.apply(lambda x: np.log(x/100))
    # df_lineups['pred_ownership'] = df_lineups.pred_ownership.apply(lambda x: np.log(x/100))

    # def product_sum(x):
    #     return 1000000000*np.exp(np.sum(np.log(x/100)))

    def product_sum(x):
        return np.sum(x)

    drafted_pct = df_lineups.groupby(['place', 'week']).agg(pct_drafted_sum=('pct_drafted', lambda x: product_sum(x)),
                                                            pred_drafted_sum=('pred_ownership', lambda x: product_sum(x)),
                                                            lineup_position=('player', 'count'))
    

                                                            
    drafted_pct['pct_drafted'] = drafted_pct.pct_drafted_sum / (drafted_pct.lineup_position / 9)
    drafted_pct['pred_drafted'] = drafted_pct.pred_drafted_sum / (drafted_pct.lineup_position / 9)

    print('\nAvg Pct Drafted:', np.mean(drafted_pct.pct_drafted), 
            '\nAvg Projected Drafted:', np.mean(drafted_pct.pred_drafted), 
            # '\nStd Perc Drafted:', np.std(drafted_pct.pct_drafted),
            '\nStd Perc Drafted:', np.std(drafted_pct.pred_drafted),
            # '\nAvg Min Pct Drafted:', np.mean(drafted_pct.pct_drafted_min), 
            # '\nAvg Min Pred Drafted:', np.mean(drafted_pct.pred_drafted_min), 
            # '\nAvg Max Pct Drafted:', np.mean(drafted_pct.pct_drafted_max), 
            # '\nAvg Max Pred Drafted:', np.mean(drafted_pct.pred_drafted_max), 
            )

#%%
player_ownership = dm.read("SELECT * FROM Million_Ownership", 'DK_Results')

val_week_min = 8
val_year_min = 2021

player_ownership.player.unique()[:50]
#%%
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
df_test.y_act = np.log(df_test.y_act/100)

df_train = df_train[df_train.y_act!=0].reset_index(drop=True)
df_train.y_act = np.log(df_train.y_act/100)

skm = SciKitModel(df_train)
X, y = skm.Xy_split('y_act', to_drop=['player', 'team'])

skm_test = SciKitModel(df_test)
X_test, y_test = skm_test.Xy_split('y_act', to_drop=['player', 'team'])

test_predict = df_test[['player', 'team', 'week', 'year']].copy()
val_predict_gbm, test_predict = run_model_mean('gbm', test_predict, X_test, cv_time_input)
val_predict_rf, test_predict = run_model_mean('rf', test_predict, X_test, cv_time_input)

val_predict = pd.merge(val_predict_gbm, val_predict_rf.drop('y_act', axis=1), on=['player', 'team', 'week', 'year'])
val_predict['pred_ownership'] = val_predict[[c for c in val_predict.columns if 'pred' in c]].mean(axis=1)
test_predict['pred_ownership'] = test_predict[[c for c in test_predict.columns if 'pred' in c]].mean(axis=1)
