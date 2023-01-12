#%%
import pandas as pd
import numpy as np
from ff import data_clean as dc
import datetime as dt

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
from skmodel import SciKitModel
import zModel_Functions as mf
from Fix_Standard_Dev import *

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

pred_version = 'sera1_rsq0_brier1_matt1_lowsample_perc_ffa_fc'
ens_version = 'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3val_fullstack'
std_dev_type = 'pred_spline_class80_q80_matt0_brier1_kfold3'

set_year = 2022
set_week = 18
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


def pull_player_ownership(contest, set_week, set_year):
    player_ownership = dm.read(f'''SELECT * 
                                FROM Contest_Ownership
                                WHERE Contest='{contest}'
                                      AND ((week <= 17 AND year < {set_year})
                                            OR (week < {set_week} AND year = {set_year})) 
                              ''', 'DK_Results').drop(['Contest', 'player_points'], axis=1)

    # rename defenses
    player_ownership.loc[player_ownership.player.isin(team_map.keys()), 'player'] = \
        player_ownership.loc[player_ownership.player.isin(team_map.keys()), 'player'].map(team_map)

    # remove players from current week
    player_ownership = player_ownership[~((player_ownership.week==set_week) & \
                                          (player_ownership.year==set_year))].reset_index(drop=True)
    return player_ownership


def pull_this_week_players(set_week, set_year):


    model_players = dm.read(f'''SELECT DISTINCT player
                                FROM Model_Predictions
                                WHERE week={set_week}
                                    AND year={set_year}
                                ''', 'Simulation').player.values
    current_players = dm.read(f'''SELECT DISTINCT player, team, 0 as pct_drafted, week, year
                            FROM Player_Teams
                            WHERE week = {set_week}
                                AND year = {set_year}
                                ''', 'Simulation')

    drop_teams = get_drop_teams(set_week, set_year)
    current_players = current_players[~(current_players.team.isin(drop_teams)) & \
                                       (current_players.player.isin(model_players))] \
                                        .drop('team', axis=1).reset_index(drop=True)
    return current_players



# calculate ownership projections
def add_proj(df):

    def_proj = dm.read('''SELECT player, 'DST' as pos, player as team, week, year, 
                             dk_salary, projected_points, fantasyPoints, ProjPts,
                             ffa_points, fc_proj_fantasy_pts_fc, ffa_position_rank, 
                             avg_proj_rank, avg_proj_points avg_proj_pts, max_proj_points,
                             fp_rank, rankadj_fp_rank, playeradj_fp_rank,  
                             expertConsensus, expertIanHartitz,
                             expertConsensus as rankadj_expertConsensus, 
                             expertConsensus as playeradj_expertConsensus
                           --  1 as team_proj_avg_proj_points, 1 as team_proj_share_avg_proj_points,
                           --  1 as team_proj_share_log_avg_proj_rank
                     FROM Defense_Data
        ''', 'Model_Features')

    proj = dm.read('''SELECT player, pos, team, week, year, 
                             dk_salary, projected_points, fantasyPoints, ProjPts,
                             ffa_points, fc_proj_fantasy_pts_fc, ffa_position_rank, 
                             avg_proj_rank, avg_proj_points avg_proj_pts, max_proj_points,
                             fp_rank, rankadj_fp_rank, playeradj_fp_rank,  
                             expertConsensus, expertIanHartitz,
                             rankadj_expertConsensus, playeradj_expertConsensus
                           --  team_proj_avg_proj_points, team_proj_share_avg_proj_points,
                           --  team_proj_share_log_avg_proj_rank
                     FROM Backfill
        ''', 'Model_Features')

    proj = pd.concat([proj, def_proj], axis=0)

    df = pd.merge(df, proj, on=['player', 'week', 'year'])

    return df


def drop_player_weeks(df):
    drop_list = ['Dalvin Cook32021', 'Calvin Ridley82021', 'Odell Beckham152021', 'Cooper Kupp152021', 'Van Jefferson152021',
                  'Josh Johnson162021', 'Kyler Murray92021', 'Darren Waller72021',
                 'Kyler Murray132021', 'Deandre Hopkins92021', 'Lamar Jackson112021']

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


def add_team_points(df, c):
    team_pts = df.groupby(['team', 'week', 'year']).agg({c: 'sum'}).reset_index()
    team_pts = team_pts.rename(columns={c: 'team_' + c})
    df = pd.merge(df, team_pts, on=['team','week', 'year'])
    df['team_frac_' + c] = df[c] / df['team_' + c]
    return df

def feature_engineering(df):

    for c in ['projected_points', 'fantasyPoints', 'ProjPts', 'avg_proj_pts', 
              'max_proj_pts', 'pred_fp_per_game', 'pred_prob', 'ffa_points', 'fc_proj_fantasy_pts_fc']:
        try: df[c+'_over_sal'] = df[c] / (df.dk_salary + 1000)
        except: pass

    for c in ['projected_points', 'avg_proj_pts', 'max_proj_pts', 'ffa_points', 'pred_fp_per_game', 'pred_prob']:
        try: df = add_team_points(df, c)
        except: pass

    for c in ['avg_proj_rank', 'min_rank', 'avg_expert', 'avg_expert_rank']:
        try: df[c+'_times_sal'] = df[c] * df.dk_salary
        except: pass

    df = df.sort_values(by=['team', 'pos', 'year', 'week', 'avg_proj_pts']).reset_index(drop=True)
    df['team_pos_week_order'] = df.groupby(['team', 'pos', 'week', 'year']).cumcount().values

    df = df.sort_values(by=['pos', 'year', 'week', 'avg_proj_pts']).reset_index(drop=True)
    df['pos_week_order'] = df.groupby(['pos', 'year', 'week']).cumcount().values

    df = df.sort_values(by=['pos', 'year', 'week', 'avg_proj_pts']).reset_index(drop=True)
    df['pos_week_order_over_sal'] = df.groupby(['pos', 'year', 'week']).cumcount().values


    avg_per_sal = df.groupby(['pos', 'year', 'week']).agg(avg_pts_per_sal_pos=('avg_proj_pts', 'mean'))
    df = pd.merge(df, avg_per_sal, on=['pos', 'year', 'week'])
    df['avg_pts_per_sal_diff'] = df.avg_proj_pts_over_sal - df.avg_pts_per_sal_pos

    # avg_per_sal = df.groupby(['pos', 'year', 'week']).agg(ffa_pts_per_sal_pos=('ffa_points_over_sal', 'mean'))
    # df = pd.merge(df, avg_per_sal, on=['pos', 'year', 'week'])
    # df['ffa_points_per_sal_diff'] = df.ffa_points_over_sal - df.ffa_pts_per_sal_pos

    df = df.sort_values(by=['year', 'week']).reset_index(drop=True)

    return df

def get_cv_time_input(df, back_weeks):
    max_date = str(df.game_date.max())
    year = int(max_date[:4])
    week = int(max_date[-2:])

    for i in range(back_weeks):
        if week > 1:
            week -= 1
        else:
            year -= 1
            week = 17
    cv_time_input = int(dt.datetime(year, 1, week).strftime('%Y%m%d'))
    print(f'Begin Validation on Week {week}, {year}')
    return cv_time_input


def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))


def create_game_date(df, back_weeks, set_week, set_year):
    
    # set up the date column for sorting
    df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
    cv_time_input = get_cv_time_input(df, back_weeks)
    train_time_split = int(dt.datetime(set_year, 1, set_week).strftime('%Y%m%d'))

    return df, cv_time_input, train_time_split


def drop_teams(data):

    df = dm.read(f'''SELECT week, year, away_team, home_team, gametime
                    FROM Gambling_Lines 
                    WHERE year >= 2020
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
    lines['implied_points_for'] = (lines.over_under / 2) - (lines.line / 2) 
    lines['implied_points_against'] = (lines.over_under / 2) + (lines.line / 2) 

    lines = pd.merge(lines, df[['team', 'week', 'year']].drop_duplicates(), on=['team', 'week', 'year'])

    lines = lines.sort_values(by=['year', 'week', 'over_under', 'implied_points_for'], ascending=[True, True, False, False])
    lines['over_under_rank'] = lines.groupby(['year', 'week'])['over_under'].cumcount().values

    lines = lines.sort_values(by=['year', 'week', 'implied_points_for'], ascending=[True, True, False])
    lines['implied_points_for_rank'] = lines.groupby(['year', 'week'])['implied_points_for'].cumcount().values
    
    df = pd.merge(df, lines, on=['team', 'week', 'year'])

    return df

def adjust_ownership(df, col, adjust_type):
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
            (df.team.isin(['NE', 'DEN'])) & \
            (df.week==6) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['LAC', 'DEN'])) & \
            (df.week==8) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['DAL', 'BAL', 'PIT', 'WAS', 'BUF', 'SF'])) & \
            (df.week==13) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['PIT', 'BAL'])) & \
            (df.week==12) & \
            (df.year==2020)
            )].reset_index(drop=True)

    df = df[~(
            (df.team.isin(['CLE', 'LVR', 'WAS', 'PHI', 'SEA', 'LAR'])) & \
            (df.week==15) & \
            (df.year==2021)
            )].reset_index(drop=True)

    return df

    

def add_model_projections(df, set_week, set_year, pred_vers, ensemble_vers, table_name):
    
    if table_name in ('Model_Predictions', 'Predicted_Probability'):
        w = 'week'
        y = 'year'
        v = 'version'
    else:
        w = 'set_week'
        y = 'set_year'
        v = 'pred_version'
    pred = dm.read(f'''SELECT player, 
                            week, 
                            year, 
                            AVG(pred_fp_per_game) pred_fp_per_game
                         FROM {table_name}
                         WHERE {w}={set_week}
                               AND {y}={set_year}
                               AND {v}='{pred_vers}'
                               AND ensemble_vers='{ensemble_vers}'
                               AND std_dev_type='{std_dev_type}'
                               AND pos !='K'
                               AND pos IS NOT NULL
                               AND player!='Ryan Griffin'
                        GROUP BY player, week, year
                                ''', 'Simulation')

    if 'Class' in table_name: pred = pred.rename(columns={'pred_fp_per_game': 'pred_prob'})

    df = pd.merge(df, pred, on=['player', 'week', 'year'])

    return df
    

def remove_duplicates(df):

    max_pts = df.groupby(['player', 'week', 'year']).agg({'avg_proj_pts': 'max'}).reset_index()
    df = pd.merge(df, max_pts, on=['player', 'week', 'year', 'avg_proj_pts'])
    df = df.drop_duplicates(subset=['player', 'week', 'year']).reset_index(drop=True)

    return df

def train_test_split(df, train_time_split):
    df_train = df[df.game_date < train_time_split].reset_index(drop=True)
    df_test = df[df.game_date == train_time_split].reset_index(drop=True)
    df_train = df_train[df_train.y_act > 0].reset_index(drop=True)
    return df_train, df_test


def run_model_mean(m, df_train, df_test, time_split):   

    print(f"\n===========Running {m}=============\n")
    skm = SciKitModel(df_train, model_obj='reg', r2_wt=1, sera_wt=10, mse_wt=0)
    X, y = skm.Xy_split('y_act', to_drop=['player', 'team'])

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

    skm_test = SciKitModel(df_test)
    X_test, y_test = skm_test.Xy_split('y_act', to_drop=['player', 'team'])

    predictions = []
    for bm in best_models:
        predictions.append(bm.fit(X, y).predict(X_test))

    test_predict = df_test[['player', 'team', 'week', 'year']].copy()
    test_predict[f'pred_ownership'] = pd.DataFrame(predictions).T.mean(axis=1)

    return val_predict, test_predict, best_models


def calc_std_dev(val_predict, test_predict):
    
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

    sd_max_test = sc.transform(test_predict[['pred_ownership']])
    test_predict['std_dev'] = sd_m(sd_max_test)
    test_predict['max_score'] = max_m(sd_max_test)
    test_predict['min_score'] = min_m(sd_max_test)

    test_predict = test_predict[['player', 'team', 'week', 'year', 'pred_ownership', 'std_dev', 'min_score', 'max_score']]
    test_predict.loc[test_predict.max_score < test_predict.pred_ownership, 'max_score'] = \
        test_predict.loc[test_predict.max_score < test_predict.pred_ownership, 'pred_ownership'] * 1.5

    return val_predict, test_predict


def check_std_dev(val_predict):

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


#===================
# Functions for past results
#===================

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

def create_millions_lineups(full_entries, base_place, places):
    df = format_lineups(full_entries, min_place=base_place, max_place=base_place+places)
    df.player = df.player.apply(dc.name_clean)
    df.loc[df.lineup_position=='DST', 'player'] = df.loc[df.lineup_position=='DST', 'player'].map(team_map)
    return df

def add_pos_to_million(df):
    
    player_pos = dm.read('''SELECT player, pos, week, year
                            FROM (
                                    SELECT player, pos, week, year, 
                                    row_number() OVER (PARTITION BY player, week, year ORDER BY projected_points DESC) rn 
                                    FROM FantasyPros
                            )
                            WHERE rn=1
                            ''', 'Pre_PlayerData')
    df = pd.merge(df, player_pos, on=['player', 'week', 'year'], how='left')
    df.loc[df.pos.isnull(), 'pos'] = df.loc[df.pos.isnull(), 'lineup_position']

    # print the flex numbers by position
    print('Flex Pct by Position\n',
            df.loc[df.lineup_position=='FLEX', 'pos'].value_counts() / \
            df.loc[df.lineup_position=='FLEX', 'pos'].shape[0])
    
    return df

def add_team_to_million(df):
    
    # print the number of players from the same team
    teams = dm.read(f'''SELECT player, team, week, year
                    FROM Player_Teams 
                    ''', 'Simulation')
    df = pd.merge(df, teams, on=['player', 'week', 'year'])
    
    # count the lineups with QB, WR, TE on same team
    team_cnts = df[df.pos.isin(['QB', 'WR', 'TE'])].groupby(['place', 'team', 'week', 'year']).agg(player_cnts=('player', 'count')).reset_index()
    max_cnts = team_cnts.groupby(['place', 'week', 'year']).agg(player_cnts=('player_cnts', 'max')).reset_index()
    team_cnts = pd.merge(team_cnts, max_cnts, on=['place', 'week','year', 'player_cnts'])
    print('Number of players on same team:\n', 
            team_cnts.player_cnts.value_counts() / team_cnts.shape[0])

    return df

def agg_ownership_group(df, full_dist):

    drafted_pct = df.groupby(['place', 'week']).agg(pct_drafted=('pct_drafted', 'sum'),
                                                    pred_drafted=('pred_ownership', 'sum'),
                                                    lineup_position=('player', 'count'))

    drafted_pct = drafted_pct[drafted_pct.lineup_position==9]
    full_dist[str(base_place)] = drafted_pct.pred_drafted

    pred_drafted_mean = np.mean(drafted_pct.pred_drafted)
    pred_drafted_std = np.std(drafted_pct.pred_drafted)

    mean_output = pd.DataFrame([pred_drafted_mean, pred_drafted_std, set_week, set_year], 
                                index=['ownership_mean', 'ownership_std', 'week', 'year']).T
    mean_output['ownership_vers'] = ownership_vers

    print('\nAvg Pct Drafted:', np.mean(drafted_pct.pct_drafted), 
            '\nAvg Projected Drafted:', pred_drafted_mean, 
            '\nStd Perc Drafted:', pred_drafted_std)

    return full_dist, mean_output


def run_ttest(full_dist, greater_or_less='greater'):
    from scipy.stats import ttest_ind
    for bp in base_places[1:]:
        print('Base Place:', bp, 'ttest p_value:', ttest_ind(full_dist['1'], 
              full_dist[str(bp)], axis=0, equal_var=True, alternative=greater_or_less)[1])

def pull_ownership(ownership_table, ownership_vers, db_name):

    pred_player_ownership = dm.read(f'''SELECT player, 
                                            team,
                                            week, 
                                            year, 
                                            AVG(pred_ownership) pred_ownership, 
                                            AVG(std_dev) std_dev,
                                            AVG(min_score) min_score,
                                            AVG(max_score) max_score
                                        FROM {ownership_table} 
                                        WHERE ownership_vers='{ownership_vers}'
                                        GROUP BY player, week, year
                                        ''', db_name)

    return pred_player_ownership


def add_pred_values(ownership_df, prob_table, pred_version, ens_version, std_dev_type, week, year, ownership_vers):
    
    if 'Validation' in prob_table: db_name = 'Validations'
    else: db_name = 'Simulation'

    df = dm.read(f'''SELECT player, 
                            week, 
                            year, 
                            AVG(pred_fp_per_game) pred_prob
                     FROM {prob_table}
                     WHERE pred_version='{pred_version}'
                              AND ensemble_vers='{ens_version}' 
                              AND std_dev_type='{std_dev_type}'
                              AND set_week={week}
                              AND set_year={year}
                     GROUP BY player, week, year
                    ''', db_name)

    df['min_score_mil'] = df.pred_prob / 10
    df['max_score_mil'] = np.where(df.pred_prob*2 > 1, 1, df.pred_prob*2)
    df['std_dev_mil'] = df.pred_prob / 3

    ownership_df = pd.merge(ownership_df, df, on=['player', 'week', 'year'])
    ownership_df = ownership_df.rename(columns={'min_score': 'min_score_own', 'max_score': 'max_score_own'})
    
    if ownership_vers == 'mil_times_standard_ln':
        ownership_df['pred_ownership'] = ownership_df.pred_prob * -ownership_df.pred_ownership
        ownership_df['min_score'] = ownership_df.min_score_mil * -ownership_df.max_score_own
        ownership_df['max_score'] = ownership_df.max_score_mil * -ownership_df.min_score_own
        ownership_df['std_dev'] = ownership_df.std_dev_mil * ownership_df.std_dev


    elif ownership_vers == 'mil_div_standard_ln':
        ownership_df['pred_ownership'] = ownership_df.pred_prob / -ownership_df.pred_ownership
        ownership_df['min_score'] = ownership_df.min_score_mil / -ownership_df.min_score_own
        ownership_df['max_score'] = ownership_df.max_score_mil / -ownership_df.max_score_own
        ownership_df['std_dev'] = ownership_df.std_dev_mil / ownership_df.std_dev

    elif ownership_vers == 'mil_only':
        ownership_df['pred_ownership'] = ownership_df.pred_prob
        ownership_df['min_score'] = ownership_df.min_score_mil
        ownership_df['max_score'] = ownership_df.max_score_mil
        ownership_df['std_dev'] = ownership_df.std_dev_mil

    else:
        ownership_df = ownership_df.rename(columns={'min_score_own': 'min_score', 'max_score_own': 'max_score'})

    return ownership_df


def save_current_week_pred(ownership_vers, set_week, set_year):

    sim_values = pull_ownership('Predicted_Ownership_Only', 'standard_ln', 'Simulation')
    
    sim_values = add_pred_values(sim_values, 'Predicted_Million', pred_version, 
                                 ens_version, std_dev_type, set_week, set_year, ownership_vers)

    sim_values = sim_values.loc[(sim_values.week==set_week) & (sim_values.year==set_year),
                                ['player', 'team', 'week', 'year', 'pred_ownership', 'std_dev', 'min_score', 'max_score']]

    sim_values['ownership_vers'] = ownership_vers
    
    display(sim_values.sort_values(by='pred_ownership', ascending=False).iloc[:50])
    dm.delete_from_db('Simulation', 'Predicted_Ownership', f"week={set_week} AND year={set_year} AND ownership_vers='{ownership_vers}'")
    dm.write_to_db(sim_values, 'Simulation', 'Predicted_Ownership', 'append')

#%%
#================
# Predict Ownership Pct
#================

for set_week, set_year in zip([#13, 14, 15, 16, 17, 
                               1, 2, 3, 4, 5, 6, 7,# 8, 9, 10, 11, 12, 13
                               15, 16, 17], 
                              [#2021, 2021, 2021, 2021, 2021,
                               2022, 2022, 2022, 2022, 2022, 2022, 2022, #2022, 2022, 2022, 2022, 2022, 2022
                               2022, 2022, 2022]):

    print(f'Running week {set_week} year {set_year}')
    ownership_vers = 'standard_ln'

    if (set_year == 2022 and set_week <= 6) or (set_year<=2021):
        back_weeks=24
    else:
        back_weeks=32

    player_ownership = pull_player_ownership(contest, set_week, set_year)
    current_players = pull_this_week_players(set_week, set_year)
    player_ownership = pd.concat([player_ownership, current_players])

    df = add_proj(player_ownership)

    df = drop_player_weeks(df)
    df = add_injuries(df)
    df = add_gambling_lines(df)
    df = feature_engineering(df)
    df = df.rename(columns={'pct_drafted': 'y_act'})
    df = filter_snap_counts(df)
    df = remove_covid_games(df)
    df = remove_duplicates(df)

    for c in ['pos', 'practice_status', 'game_status', 'practice_game']:
        df = pd.concat([df, pd.get_dummies(df[c])], axis=1).drop(c, axis=1)

    df, cv_time_input, train_time_split = create_game_date(df, back_weeks, set_week, set_year)
    df_train, df_test = train_test_split(df, train_time_split)

    df_train = adjust_ownership(df_train, 'y_act', 'ln')
    df_test = adjust_ownership(df_test, 'y_act', 'ln')

    val_predict_pos, test_predict_pos, best_models = run_model_mean('lgbm', df_train[df_train['DST']==0].reset_index(drop=True), 
                                                                    df_test[df_test['DST']==0].reset_index(drop=True), cv_time_input)
    val_predict_dst, test_predict_dst, best_models = run_model_mean('lgbm', df_train[df_train['DST']==1].reset_index(drop=True), 
                                                                    df_test[df_test['DST']==1].reset_index(drop=True), cv_time_input)
    val_predict = pd.concat([val_predict_pos, val_predict_dst], axis=0)
    test_predict = pd.concat([test_predict_pos, test_predict_dst], axis=0)
    mf.show_scatter_plot(val_predict.pred_ownership, val_predict.y_act)


    val_predict, test_predict = calc_std_dev(val_predict, test_predict)
    check_std_dev(val_predict)

    val_predict['error'] = val_predict.pred_ownership - val_predict.y_act
    display(test_predict.sort_values(by='pred_ownership', ascending=False).iloc[:50])
    display(val_predict.sort_values(by='error').iloc[-25:])
    display(val_predict.sort_values(by='error').iloc[:25])

    test_predict['ownership_vers'] = ownership_vers
    dm.delete_from_db('Simulation', 'Predicted_Ownership_Only', f"week={set_week} AND year={set_year} AND ownership_vers='{ownership_vers}'", create_backup=False)
    dm.write_to_db(test_predict, 'Simulation', 'Predicted_Ownership_Only', 'append')

    val_predict = val_predict[['player', 'team', 'week', 'year', 'pred_ownership', 'std_dev', 'min_score', 'max_score']]
    val_predict['ownership_vers'] = ownership_vers
    dm.delete_from_db('Validations', 'Predicted_Ownership_Validation', f"ownership_vers='{ownership_vers}'", create_backup=False)
    dm.write_to_db(val_predict, 'Validations', 'Predicted_Ownership_Validation', 'append')

    #==================
    # Compare predicted ownership to past entries
    #==================

    full_entries = dm.read(f'''SELECT * 
                               FROM Contest_Results 
                               WHERE Contest='{contest}' 
                                     AND ((week <= 17 AND year < {set_year})
                                           OR (week < {set_week} AND year = {set_year})) 
                                     ''', 'DK_Results')

    player_ownership = dm.read(f"SELECT * FROM Contest_Ownership WHERE Contest='{contest}'", 'DK_Results')


    for ownership_vers in ['standard_ln', 'mil_times_standard_ln', 'mil_div_standard_ln', 'mil_only']:

        pred_player_ownership = pull_ownership('Predicted_Ownership_Validation', 'standard_ln', 'Validations')
        pred_player_ownership = add_pred_values(pred_player_ownership, 'Model_Validations_Million', pred_version, 
                                                ens_version, std_dev_type, set_week, set_year, ownership_vers)

        mean_var = []
        full_dist = {}
        base_places = [1, 10000, 25000, 50000, 75000, 100000, 150000]
        for base_place, places in zip(base_places, [50, 1000, 1000, 1000, 1000, 1000, 1000]):
            print(f'\nPlaces {base_place}-{places+base_place}\n==================')

            df_lineups = create_millions_lineups(full_entries, base_place, places)
            df_lineups = add_pos_to_million(df_lineups)
            df_lineups = add_team_to_million(df_lineups)
            
            df_lineups = pd.merge(df_lineups, player_ownership, on=['player', 'week', 'year'], how='left')
            df_lineups = pd.merge(df_lineups, pred_player_ownership, on=['player', 'week', 'year'])
            full_dist, mean_output = agg_ownership_group(df_lineups, full_dist)

            if base_place==1:
                
                dm.delete_from_db('Simulation', 'Mean_Ownership', f"year={set_year} AND week={set_week} AND ownership_vers='{ownership_vers}'", create_backup=False)
                dm.write_to_db(mean_output, 'Simulation', 'Mean_Ownership', 'append')

        run_ttest(full_dist, greater_or_less='greater')
        save_current_week_pred(ownership_vers, set_week, set_year)
    

# %%
df_lineups[df_lineups.place==1].groupby(['week', 'year']).agg({'player': 'count'})
# %%
df_lineups[(df_lineups.week==14) & (df_lineups.year==2021) & (df_lineups.place==1)]
# %%
full_entries[(full_entries.week==14) & (full_entries.year==2021) & (full_entries.Rank==1)].Lineup.values[0]
# %%
