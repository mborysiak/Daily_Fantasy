#%%
import pandas as pd
import numpy as np
from ff import data_clean as dc
import datetime as dt

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
from skmodel import SciKitModel
import zModel_Functions as mf
# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

set_year = 2022
set_week = 9
contest = 'Million'

#%%

team_map = {'Cardinals': 'ARI',
            'Falcons': 'ATL',
            'Ravens': 'BAL',
            'Bills': 'BUF',
            'Panthers': 'CAR',
            'Bears': 'CHI',
            'Bengals': 'CIN',
            'Browns': 'CLE',
            'Cowboys': 'DAL',
            'Broncos': 'DEN',
            'Lions': 'DET',
            'Packers': 'GB',
            'Texans': 'HOU',
            'Colts': 'IND',
            'Jaguars': 'JAC',
            'Chiefs': 'KC',
            'Chargers': 'LAC',
            'Rams': 'LAR',
            'Dolphins': 'MIA',
            'Vikings': 'MIN',
            'Patriots': 'NE',
            'Saints': 'NO',
            'Giants': 'NYG',
            'Jets': 'NYJ',
            'Raiders': 'LVR',
            'Eagles': 'PHI',
            'Steelers': 'PIT',
            '49ers': 'SF',
            '49Ers': 'SF',
            'Seahawks': 'SEA',
            'Buccaneers': 'TB',
            'Titans': 'TEN',
            'Redskins': 'WAS',
            'Football Team': 'WAS',
            'Commanders': 'WAS'}

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
    def_proj1 = dm.read('''SELECT team player, team, week, year, projected_points
                        FROM FantasyPros
                        WHERE pos='DST'
                                ''', 'Pre_PlayerData')

    def_proj2 = dm.read('''SELECT offteam player, position pos, offTeam team, week, year, 
                                  dk_salary, fantasyPoints, `Proj Pts` ProjPts 
                           FROM PFF_Expert_Ranks
                           JOIN (SELECT offteam, week, year, salary dk_salary, fantasyPoints
                                 FROM PFF_Proj_Ranks)
                                 USING (offTeam, week, year)''', 'Pre_TeamData')
    def_proj = pd.merge(def_proj1, def_proj2, on=['player', 'team', 'week', 'year']).dropna()

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

    proj = pd.concat([proj, def_proj], axis=0)

    proj['avg_proj_pts'] = proj[['projected_points', 'fantasyPoints', 'ProjPts']].mean(axis=1)

    proj.pos = proj.pos.apply(lambda x: x.upper())
    df = pd.merge(df, proj, on=['player', 'week', 'year'])

    return df


def drop_player_weeks(df, drop_list):
    df['to_drop'] = df.player + df.week.astype('str') + df.year.astype('str')
    df = df[~df.to_drop.isin(drop_list)].reset_index(drop=True).drop('to_drop', axis=1)
    return df


def get_drop_teams(week, year):

    import datetime as dt

    df = dm.read(f'''SELECT away_team, home_team, gametime 
                    FROM Gambling_Lines 
                    WHERE week={week} 
                        and year={year} 
                ''', 'Pre_TeamData')
    df.gametime = pd.to_datetime(df.gametime)
    df['day_of_week'] = df.gametime.apply(lambda x: x.weekday())
    df['hour_in_day'] = df.gametime.apply(lambda x: x.hour)
    df = df[(df.day_of_week!=6) | (df.hour_in_day > 16) | (df.hour_in_day < 11)]
    drop_teams = list(df.away_team.values)
    drop_teams.extend(list(df.home_team.values))

    return drop_teams


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
        df[c+'_over_sal'] = df[c] / (df.dk_salary + 1000)

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
    skm = SciKitModel(df_train, model_obj='reg', r2_wt=1, sera_wt=5)

    # get the model pipe for stacking setup and train it on meta features
    pipe = skm.model_pipe([
                           skm.piece('std_scale'),
                           skm.piece('k_best'),
                           skm.piece(m)
                        ])

    params = skm.default_params(pipe)
    params['k_best__k'] = range(1, X.shape[1]+1)
    
    # run the model with parameter search
    best_models, oof_data, _ = skm.time_series_cv(pipe, X, y, 
                                                 params, n_iter=25,
                                                 bayes_rand='custom_rand',
                                                 col_split='game_date',
                                                 time_split=time_split)

    val_predict = oof_data['full_hold'].copy()
    val_predict = val_predict.rename(columns={'pred': f'pred_ownership'})

    predictions = []
    for bm in best_models:
        predictions.append(bm.fit(X, y).predict(X_test))

    test_predict[f'pred_ownership'] = pd.DataFrame(predictions).T.mean(axis=1)

    return val_predict, test_predict, best_models


def drop_teams(data):

    df = dm.read(f'''SELECT week, year, away_team, home_team, gametime
                    FROM Gambling_Lines 
                    WHERE year >= 2021
            ''', 'Pre_TeamData')
    df.gametime = pd.to_datetime(df.gametime)
    df['day_of_week'] = df.gametime.apply(lambda x: x.weekday())
    df['hour_in_day'] = df.gametime.apply(lambda x: x.hour)
    df = df[(df.day_of_week!=6) | (df.hour_in_day > 16) | (df.hour_in_day < 11)]
    drop_teams = df[['week','year', 'away_team']].rename(columns={'away_team': 'home_team'})
    drop_teams = pd.concat([drop_teams, df[['week','year', 'home_team']]], axis=0)
    drop_teams['to_drop'] = 1
    drop_teams = drop_teams.rename(columns={'home_team': 'team'})

    
    data = pd.merge(data, drop_teams, on=['team', 'week', 'year'], how='left')
    data = data[data.to_drop.isnull()].reset_index(drop=True).drop('to_drop', axis=1)

    return data

def filter_snap_counts(df):
    snaps = dm.read('''SELECT player, snap_counts, week, year
                FROM Snap_Counts_V2
                WHERE snap_counts!='bye' 
                        AND week != 18
                ''', 'Post_PlayerData')

    snaps.snap_counts = snaps.snap_counts.apply(lambda x: int(float(x)))
    snaps.week = snaps.week.astype('int')

    df = pd.merge(df, snaps, on=['player', 'year', 'week'], how='left')
    df = df[(df.snap_counts > 0) | \
            (df.pos=='DST') | \
            ((df.week==set_week) & (df.year==set_year))].reset_index(drop=True).drop('snap_counts', axis=1)

    return df

def add_gambling_lines(df):

    lines = dm.read("SELECT * FROM Gambling_Lines", 'Pre_TeamData')

    away = lines[['away_team', 'away_line', 'away_moneyline', 'over_under', 'week', 'year']]
    home = lines[['home_team', 'home_line', 'home_moneyline', 'over_under', 'week', 'year']]
    home = home.assign(is_home=1)
    away = away.assign(is_home=0)
    away.columns = ['team', 'line', 'moneyline', 'over_under', 'week', 'year', 'is_home']
    home.columns = ['team', 'line', 'moneyline', 'over_under', 'week', 'year', 'is_home']

    lines = pd.concat([home, away], axis=0)
    lines = dc.convert_to_float(lines)
    lines['implied_points_for'] = (lines.over_under / 2) + (lines.line / 2) 
    lines['implied_points_against'] = (lines.over_under / 2) - (lines.line / 2) 

    df = pd.merge(df, lines, on=['team', 'week', 'year'])

    return df

def adjust_owernship(df, col, adjust_type):
    if adjust_type == 'inverse': df[col] = -(1 / (df[col] + 1))
    elif adjust_type =='ln': df[col] = np.log(df[col]/100)
    elif adjust_type == 'ln_prob': df[col] = np.log(df[col]+0.01)
    elif adjust_type == 'exp': df[col] = np.exp(df[col]+1)
    return df

def remove_covid_games(df):

    df = df[~(
            (df.team.isin(['PIT', 'TEN', 'NE', 'KC'])) & \
            (df.week==4) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['BUF', 'TEN'])) & \
            (df.week==5) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['NE', 'DEN'])) & \
            (df.week==6) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['PIT', 'BAL'])) & \
            (df.week==7) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['LAC', 'DEN'])) & \
            (df.week==8) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['PIT', 'BAL'])) & \
            (df.week==12) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['DAL', 'BAL', 'PIT', 'WAS', 'BUF', 'SF'])) & \
            (df.week==13) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['CLE', 'LVR', 'WAS', 'PHI', 'SEA', 'LAR'])) & \
            (df.week==15) & \
            (df.year==2021)
            )].reset_index(drop=True)

    return df

    

def add_model_projections(df, set_week, set_year, pred_vers, ensemble_vers):
    pred = dm.read(f'''SELECT player, 
                            week, 
                            year, 
                            AVG(pred_fp_per_game) pred_fp_per_game
                         FROM Model_Validations
                         WHERE set_week={set_week}
                               AND set_year={set_year}
                               AND version='{pred_vers}'
                               AND ensemble_vers='{ensemble_vers}'
                               AND std_dev_type='{std_dev_type}'
                               AND pos !='K'
                               AND pos IS NOT NULL
                               AND player!='Ryan Griffin'
                                ''', 'Simulation')

    df = pd.merge(df, pred, on=['player', 'week', 'year'])

    return df
    

def remove_duplicates(df):

    max_pts = df.groupby(['player', 'week', 'year']).agg({'avg_proj_pts': 'max'}).reset_index()
    df = pd.merge(df, max_pts, on=['player', 'week', 'year', 'avg_proj_pts'])
    return df



#%%
#================
# Predict Ownership Pct
#================

# for set_week, set_year in zip([15, 16, 17, 4], 
#                               [2021, 2021, 2021, 2022]):

print(f'Running week {set_week} year {set_year}')
val_week_min = 8
val_year_min = 2021

player_ownership = dm.read(f'''SELECT * 
                            FROM Contest_Ownership
                            WHERE week <= 16
                                    AND Contest='{contest}' ''', 'DK_Results').drop('player_points', axis=1)

player_ownership.loc[player_ownership.player.isin(team_map.keys()), 'player'] = \
    player_ownership.loc[player_ownership.player.isin(team_map.keys()), 'player'].map(team_map)

player_ownership = player_ownership[~((player_ownership.week==set_week) & \
                                        (player_ownership.year==set_year))].drop('Contest', axis=1).reset_index(drop=True)

model_players = dm.read(f'''SELECT DISTINCT player
                            FROM Model_Predictions
                            WHERE week={set_week}
                                AND year={set_year}
                            ''', 'Simulation').player.values
ty_players = dm.read(f'''SELECT DISTINCT player, team, 0 as pct_drafted, {set_week} week, {set_year} year
                        FROM Player_Teams
                            ''', 'Simulation')

drop_teams = get_drop_teams(set_week, set_year)
ty_players = ty_players[~ty_players.team.isin(drop_teams)].drop('team', axis=1).reset_index(drop=True)
ty_players = ty_players[ty_players.player.isin(model_players)].reset_index(drop=True)

player_ownership = pd.concat([player_ownership, ty_players])


df = add_proj(player_ownership)

drop_list = ['Dalvin Cook32021', 'Calvin Ridley82021', 'Odell Beckham152021', 'Cooper Kupp152021', 'Van Jefferson152021',
            "D'Andre Swift112021", 'Josh Johnson162021', 'Kyler Murray92021', 'Darren Waller72021',
            'Kyler Murray132021', 'Deandre Hopkins92021', 'Lamar Jackson112021']

df = drop_player_weeks(df, drop_list)
df = add_injuries(df)
df = add_gambling_lines(df)
df = feature_engineering(df)
df = df.rename(columns={'pct_drafted': 'y_act'})
df = filter_snap_counts(df)
df = remove_covid_games(df)
df = remove_duplicates(df)

for c in ['pos', 'practice_status', 'game_status', 'practice_game']:
    df = pd.concat([df, pd.get_dummies(df[c], drop_first=True)], axis=1).drop(c, axis=1)

df, cv_time_input, train_time_split = create_game_date(df, val_year_min, val_week_min, year_week_to_date)

df_train = df[df.game_date < train_time_split].reset_index(drop=True)
df_test = df[df.game_date == train_time_split].reset_index(drop=True)
df_train = df_train[df_train.y_act!=0].reset_index(drop=True)

df_train = adjust_owernship(df_train, 'y_act', 'ln')
df_test = adjust_owernship(df_test, 'y_act', 'ln')

skm = SciKitModel(df_train)
X, y = skm.Xy_split('y_act', to_drop=['player', 'team'])

skm_test = SciKitModel(df_test)
X_test, y_test = skm_test.Xy_split('y_act', to_drop=['player', 'team'])


test_predict = df_test[['player', 'team', 'week', 'year']].copy()
val_predict, test_predict, best_models = run_model_mean('lgbm', test_predict, X_test, cv_time_input)

mf.show_scatter_plot(val_predict.pred_ownership, val_predict.y_act)
# val_predict = pd.merge(val_predict_gbm, val_predict_rf.drop('y_act', axis=1), on=['player', 'team', 'week', 'year'])
# val_predict['pred_ownership'] = val_predict[[c for c in val_predict.columns if 'pred' in c]].mean(axis=1)
# test_predict['pred_ownership'] = test_predict[[c for c in test_predict.columns if 'pred' in c]].mean(axis=1)

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

# test_predict.loc[test_predict.min_score < 0, 'min_score'] = 0 
test_predict.loc[test_predict.max_score < test_predict.pred_ownership, 'max_score'] = \
    test_predict.loc[test_predict.max_score < test_predict.pred_ownership, 'pred_ownership'] * 1.5
# test_predict.loc[test_predict.std_dev < 0, 'std_dev'] = 1

dm.delete_from_db('Simulation', 'Predicted_Ownership_Only', f'week={set_week} AND year={set_year}')
dm.write_to_db(test_predict, 'Simulation', 'Predicted_Ownership_Only', 'append')

val_predict = val_predict[['player', 'team', 'week', 'year', 'pred_ownership', 'std_dev', 'min_score', 'max_score']]
dm.write_to_db(val_predict, 'Simulation', 'Predicted_Ownership_Validation', 'replace')

#%%

val_predict_chk = pd.merge(df_train[['player', 'week', 'year', 'y_act']], val_predict, on=['player', 'week', 'year'])
val_predict_chk['error'] = val_predict_chk.pred_ownership - val_predict_chk.y_act
display(val_predict_chk.sort_values(by='error').iloc[-25:])
display(val_predict_chk.sort_values(by='error').iloc[:25])

#%%

full_entries = dm.read(f"SELECT * FROM Contest_Results WHERE Contest='{contest}'", 'DK_Results')
player_ownership = dm.read(f"SELECT * FROM Contest_Ownership WHERE Contest='{contest}'", 'DK_Results')

#%%

pred_version = 'sera1_rsq0_brier1_matt1_lowsample_perc'
ens_version = 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3'
std_dev_type = 'pred_spline_class80_matt1_brier1_kfold3'

def create_constraint_metrics(prob_table, ownership_table, pred_version, ensemble_vers, std_dev_type, week, year):

    pred_player_ownership = dm.read(f'''SELECT player, 
                                            team,
                                            week, 
                                            year, 
                                            AVG(pred_ownership) pred_ownership, 
                                            AVG(std_dev) std_dev,
                                            AVG(min_score) min_score,
                                            AVG(max_score) max_score
                                        FROM {ownership_table} 
                                        GROUP BY player, week, year
                                        ''', 'Simulation')

    # df = dm.read(f'''SELECT player, 
    #                         week, 
    #                         year, 
    #                         AVG(pred_fp_per_game) pred_prob
    #                  FROM {prob_table}
    #                  WHERE pred_version='{pred_version}'
    #                           AND ensemble_vers='{ensemble_vers}' 
    #                           AND std_dev_type='{std_dev_type}'
    #                           AND set_week={week}
    #                           AND set_year={year}
    #                  GROUP BY player, week, year
    #                 ''', 'Simulation')

    # pred_player_ownership = pd.merge(pred_player_ownership, df, on=['player', 'week', 'year'])

    # for c in ['pred_ownership', 'std_dev', 'min_score', 'max_score']:
    #     pred_player_ownership[c] = pred_player_ownership.pred_prob * pred_player_ownership[c]

    return pred_player_ownership

pred_player_ownership = create_constraint_metrics('Model_Validations_Class', 'Predicted_Ownership_Validation', 
                                                  pred_version, ens_version, std_dev_type, set_week, set_year)



#%%
mean_var = []
full_dist = {}
base_places = [1, 10000, 25000, 50000, 75000, 100000, 150000]
for base_place, places in zip(base_places, [50, 1000, 1000, 1000, 1000, 1000, 1000]):
    print(f'\nPlaces {base_place}-{places+base_place}\n==================')

    df_lineups = format_lineups(full_entries, min_place=base_place, max_place=base_place+places)
    df_lineups.player = df_lineups.player.apply(dc.name_clean)
    df_lineups.loc[df_lineups.lineup_position=='DST', 'player'] = df_lineups.loc[df_lineups.lineup_position=='DST', 'player'].map(team_map)
    
    player_pos = dm.read('''SELECT DISTINCT player, pos FROM FantasyPros''', 'Pre_PlayerData')
    df_lineups = pd.merge(df_lineups, player_pos, on='player', how='left')
    df_lineups.loc[df_lineups.pos.isnull(), 'pos'] = df_lineups.loc[df_lineups.pos.isnull(), 'lineup_position']

    # print the flex numbers by position
    print('Flex Pct by Position\n',
           df_lineups.loc[df_lineups.lineup_position=='FLEX', 'pos'].value_counts() / \
           df_lineups.loc[df_lineups.lineup_position=='FLEX', 'pos'].shape[0])

    # print the number of players from the same team
    teams = dm.read('''SELECT * FROM Player_Teams''', 'Simulation')
    df_lineups = pd.merge(df_lineups, teams, on=['player'])

    team_cnts = df_lineups[df_lineups.pos.isin(['QB', 'RB', 'WR', 'TE'])].groupby(['place', 'team', 'week', 'year']).agg(player_cnts=('player', 'count')).reset_index()
    max_cnts = team_cnts.groupby(['place', 'week', 'year']).agg(player_cnts=('player_cnts', 'max')).reset_index()
    team_cnts = pd.merge(team_cnts, max_cnts, on=['place', 'week','year', 'player_cnts'])
    print('Number of players on same team:\n', 
          team_cnts.player_cnts.value_counts() / team_cnts.shape[0])

    df_lineups = pd.merge(df_lineups, player_ownership, on=['player', 'week', 'year'], how='left')
    df_lineups = pd.merge(df_lineups, pred_player_ownership, on=['player', 'week', 'year'])

    # df_lineups.pred_ownership = df_lineups.pred_ownership / df_lineups.avg_proj_pts
    
    # df_lineups = adjust_owernship(df_lineups, 'pct_drafted', 'ln')
    # df_lineups = adjust_owernship(df_lineups, 'pred_ownership', 'ln_prob')

    def product_sum(x):
        return np.sum(x)

    drafted_pct = df_lineups.groupby(['place', 'week']).agg(pct_drafted=('pct_drafted', lambda x: product_sum(x)),
                                                            pred_drafted=('pred_ownership', lambda x: product_sum(x)),
                                                            lineup_position=('player', 'count'))
    
    drafted_pct = drafted_pct[drafted_pct.lineup_position==9]

    full_dist[str(base_place)] = drafted_pct.pred_drafted
                                                            
    # drafted_pct['pct_drafted'] = drafted_pct.pct_drafted / (drafted_pct.lineup_position / 9)
    # drafted_pct['pred_drafted'] = drafted_pct.pred_drafted / (drafted_pct.lineup_position / 9)

    pred_drafted_mean = np.mean(drafted_pct.pred_drafted)
    pred_drafted_std = np.std(drafted_pct.pred_drafted)

    print('\nAvg Pct Drafted:', np.mean(drafted_pct.pct_drafted), 
            '\nAvg Projected Drafted:', pred_drafted_mean, 
            '\nStd Perc Drafted:', pred_drafted_std,
            )


    mean_var.append([pred_drafted_mean, pred_drafted_std, base_place])

    if base_place==1:
        mean_output = pd.DataFrame([pred_drafted_mean, pred_drafted_std, set_week, set_year], 
                                    index=['ownership_mean', 'ownership_std', 'week', 'year']).T
        dm.delete_from_db('Simulation', 'Mean_Ownership', f"year={set_year} AND week={set_week}")
        dm.write_to_db(mean_output, 'Simulation', 'Mean_Ownership', 'append')


# %%

from scipy.stats import ttest_ind

for bp in base_places[1:]:
    print('Base Place:', bp, 'ttest p_value:', ttest_ind(full_dist['1'], full_dist[str(bp)], axis=0, equal_var=True, alternative='greater')[1])


# %%

sim_values = create_constraint_metrics('Predicted_Probability', 'Predicted_Ownership_Only',
                                       pred_version, ens_version, std_dev_type, set_week, set_year)
sim_values = sim_values.loc[(sim_values.week==set_week) & (sim_values.year==set_year),
                            ['player', 'team', 'week', 'year', 'pred_ownership', 'std_dev', 'min_score', 'max_score']]

dm.delete_from_db('Simulation', 'Predicted_Ownership', f'week={set_week} AND year={set_year}')
dm.write_to_db(sim_values, 'Simulation', 'Predicted_Ownership', 'append')
# %%
