#%%

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
import numpy as np
import pandas as pd
import ff.data_clean as dc

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

def create_sd_max_metrics(df):
 
    df['sd_metric'] = (df.fantasyPoints + df.ProjPts + \
                        df.projected_points + df.roll_std) / 4
    df['max_metric'] = (df.fantasyPoints + df.ProjPts + \
                        df.projected_points + df.roll_max) / 4
    
    df = df.drop(['fantasyPoints', 'ProjPts', 'projected_points', 
                    'roll_std', 'roll_max'], axis=1)

    return df


def rolling_max_std(pos, week, year):

    if pos != 'Defense':
        df = dm.read(f'''SELECT player, team, week, season year, fantasy_pts y_act
                        FROM {pos}_Stats
                        WHERE season >= 2020
                              AND (
                                    (week < {week} AND year = {year})
                                    OR 
                                    (year < {year})
                                  )
                            ''', 'FastR')
    else:
        df = dm.read(f'''SELECT defTeam player, defTeam team, week, season year, fantasy_pts y_act
                         FROM {pos}_Stats
                         WHERE season >= 2020
                              AND (
                                    (week < {week} AND year = {year})
                                    OR 
                                    (year < {year})
                                  )
                             ''', 'FastR')

    df = df.sort_values(by=['player', 'year', 'week']).reset_index(drop=True)
    df['roll_pts'] = df.y_act#df.groupby(['player'])['y_act'].shift(1)
    df['roll_max'] = df.groupby('player')['roll_pts'].rolling(8, min_periods=1).apply(lambda x: pd.Series(x).nlargest(2).iloc[-1]).values

    df['roll_mean'] = df.groupby('player')['roll_pts'].rolling(9, min_periods=1).apply(lambda x: pd.Series(x).nlargest(5).iloc[-1]).values
    df['roll_std'] = df.roll_max - df.roll_mean
    df = df.drop(['roll_pts', 'roll_mean'], axis=1)
    
    most_recent = df.drop_duplicates(subset=['player'], keep='last')
    most_recent = most_recent[['player', 'roll_max', 'roll_std']]
    most_recent = most_recent.assign(week=week, year=year)
    return df, most_recent


def projection_data(pos):

    if pos == 'Defense':

        # pull in the salary and actual results data
        proj = dm.read('''SELECT team player, team, week, year, projected_points
                            FROM FantasyPros
                            WHERE pos='DST'
                                   ''', 'Pre_PlayerData')

        proj_2 = dm.read('''SELECT offteam player, offTeam team, defTeam, week, year, fantasyPoints, `Proj Pts` ProjPts 
                            FROM PFF_Expert_Ranks
                            JOIN (SELECT offteam, week, year, fantasyPoints
                                  FROM PFF_Proj_Ranks)
                                  USING (offTeam, week, year)''', 'Pre_TeamData')
        proj = pd.merge(proj, proj_2, on=['player', 'team', 'week', 'year'])
    
    else:
        # pull in the salary and actual results data
        proj = dm.read(f'''SELECT player, offTeam team, defTeam, week, year, projected_points, fantasyPoints, ProjPts
                            FROM PFF_Proj_Ranks
                            JOIN (SELECT player, team offTeam, week, year, projected_points 
                                  FROM FantasyPros)
                                  USING (player, offTeam, week, year)
                            JOIN (SELECT player, offTeam, week, year, `Proj Pts` ProjPts 
                                  FROM PFF_Expert_Ranks)
                                  USING (player, offTeam, week, year)
                            WHERE position='{pos.lower()}' 
                                  ''', 'Pre_PlayerData')

    return proj



def get_max_metrics(week, year):

    corr_data = pd.DataFrame()
    current_data = pd.DataFrame()

    for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:

        proj = projection_data(pos)

        if pos=='QB':
            proj = proj[(proj.ProjPts > 8) & (proj.projected_points > 8)].reset_index(drop=True)

        # get the rolling stats data    
        stats, current = rolling_max_std(pos, week, year)

        # join together and calculate sd and max metrics
        past = pd.merge(proj, stats, on=['player', 'team', 'week', 'year'])
        past = create_sd_max_metrics(past)
        past['pos'] = pos
        corr_data = pd.concat([corr_data, past], axis=0)

        current = pd.merge(proj, current, on=['player', 'week', 'year'])
        current = create_sd_max_metrics(current)
        current['pos'] = pos
        current_data = pd.concat([current_data, current], axis=0)
    
    return corr_data, current_data


def create_pos_rank(df, opponent=False):

    if opponent:
        df = df.drop('team', axis=1).rename(columns={'defTeam': 'team'})
        prefix = 'Opp'
    else:
        prefix = ''

    df = df.sort_values(by=['team', 'pos', 'year', 'week', 'max_metric'],
                        ascending=[True, True, True, True, False]).reset_index(drop=True)

    df['pos_rank'] = prefix + df.pos + df.groupby(['team', 'pos', 'year', 'week']).cumcount().apply(lambda x: str(x))


    return df

def get_team_totals(df, col):
    team_totals = df[df[col].isin(['QB0', 'RB0', 'RB1', 'WR0', 'WR1', 'WR2', 'TE0'])]
    team_totals = team_totals[['team', 'week', 'year', 'max_metric']].drop_duplicates().reset_index(drop=True)
    team_totals = team_totals.groupby(['week', 'year', 'team']).agg(team_total=('max_metric', 'sum')).reset_index()

    df = pd.merge(df, team_totals, on=['week','year','team'])

    return df


def get_group_covars(corr_data):

    matrices = pd.DataFrame()
    percs = np.percentile(corr_data.team_total, [0, 20, 40, 60, 80, 100])
    for i in range(len(percs) - 1): 
        perc_low = percs[i]
        perc_high =  percs[i+1]

        cor_matrix = corr_data[(corr_data.team_total > perc_low) & (corr_data.team_total <= perc_high)]
        cor_matrix = cor_matrix.pivot_table(index=['team', 'week', 'year'], columns='pos_rank', values='y_act').fillna(0)

        # calculate covariance matrix and convert to long format
        cov_matrix = cor_matrix.cov()
        cov_matrix = cov_matrix.rename_axis(None).rename_axis(None, axis=1)
        cov_matrix = cov_matrix.stack().reset_index()
        cov_matrix.columns = ['pos_rank1', 'pos_rank2', 'covariance']
        cov_matrix['perc'] = i

        matrices = pd.concat([matrices, cov_matrix])

    return matrices, percs

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


def get_predictions(drop_teams, pred_vers,std_dev_type, set_week, set_year, full_model_rel_weight):

    preds = dm.read(f'''SELECT * 
                        FROM Model_Predictions 
                        WHERE version='{pred_vers}'
                              AND std_dev_type = '{std_dev_type}'
                              AND ensemble_vers='{ensemble_vers}'
                              AND week = '{set_week}'
                              AND year = '{set_year}' 
                              AND player != 'Ryan Griffin'
                ''', 'Simulation')

    preds['weighting'] = 1
    preds.loc[preds.model_type=='full_model', 'weighting'] = full_model_rel_weight

    _, current = get_max_metrics(set_week, set_year)
    preds = pd.merge(preds, current[['player', 'pos', 'max_metric']], on=['player', 'pos'])

    score_cols = ['pred_fp_per_game', 'std_dev', 'max_metric']
    for c in score_cols: preds[c] = preds[c] * preds.weighting

    # Groupby and aggregate with namedAgg [1]:
    preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                                  'std_dev': 'sum',
                                                                  'weighting': 'sum',
                                                                  'max_metric': 'sum'})

    for c in score_cols: preds[c] = preds[c] / preds.weighting
    preds = preds.drop('weighting', axis=1)

    teams = dm.read(f"SELECT player, team FROM Player_Teams WHERE week={set_week} AND year={set_year}", 'Simulation')
    preds = pd.merge(preds, teams, on=['player'])
    preds = preds[~preds.team.isin(drop_teams)].reset_index(drop=True)

    preds = preds.assign(week=set_week, year=set_year)
    
    return preds


def create_player_matches(preds, opponent=False):

    preds = create_pos_rank(preds)

    pred_cov1 = preds[['player', 'team', 'pos_rank']]
    pred_cov2 = preds[['player', 'team', 'pos_rank', 'std_dev', 'max_metric', 'week', 'year']]

    if opponent:
        matchups = get_matchups()
        pred_cov2 = pd.merge(pred_cov2, matchups, on=['team','week', 'year'])
        pred_cov2 = pred_cov2.drop('team', axis=1).rename(columns={'defTeam': 'team'})
        pred_cov2.pos_rank = pred_cov2.pos_rank.apply(lambda x: 'Opp'+x)

    pred_cov = pd.merge(pred_cov1, pred_cov2, on='team')
    pred_cov.columns = ['player1', 'team', 'pos_rank1', 'player2', 'pos_rank2', 'std_dev', 'max_metric', 'week', 'year']
    pred_cov = pred_cov[['player1', 'player2', 'team', 'pos_rank1', 'pos_rank2', 'std_dev', 'max_metric', 'week', 'year']]


    return pred_cov


def get_matchups():
    return dm.read('''SELECT offTeam team, defTeam, week, year
                      FROM (
                            SELECT  *, 
                                    row_number() OVER (PARTITION BY offTeam, week, year 
                                                        ORDER BY cnts DESC) AS rnk
                            FROM (
                                    SELECT offTeam, defTeam, week, year, count(*) cnts
                                    FROM PFF_Expert_Ranks
                                    GROUP BY offTeam, defTeam, week, year
                                 )
                       )
                      WHERE rnk=1''', 'Pre_PlayerData')


def apply_group_covar(pred_cov, matrices, percs):
    pred_cov['perc'] = 0
    for i in range(len(percs) - 1): 

        perc_low = percs[i]
        perc_high =  percs[i+1]
        pred_cov.loc[(pred_cov.team_total > perc_low) & (pred_cov.team_total <= perc_high), ['perc']] = i

    pred_cov.loc[(pred_cov.team_total > perc_high), ['perc']] = i
    pred_cov = pd.merge(pred_cov, matrices, on=['pos_rank1', 'pos_rank2', 'perc'], how='left')

    pred_cov.loc[pred_cov.player1==pred_cov.player2, 'covariance'] = pred_cov.loc[pred_cov.player1==pred_cov.player2, 'std_dev']**2
    pred_cov = pd.pivot_table(pred_cov, index='player1', columns='player2', values='covariance').fillna(0)
    pred_cov = pred_cov.rename_axis(None).rename_axis(None, axis=1)

    return pred_cov


def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

def cleanup_pred_covar(pred_cov):

    pred_cov_final = pd.DataFrame(get_near_psd(pred_cov))
    pred_cov_final.columns = pred_cov.columns
    pred_cov_final.index = pred_cov.index
    pred_cov_final = pred_cov_final.reset_index().rename(columns={'index': 'player'})

    for c in pred_cov_final.columns[1:]:
        pred_cov_final.loc[abs(pred_cov_final[c]) < 0.00001, c] = 0

    pred_cov_final = pd.melt(pred_cov_final, id_vars='player', var_name=['player_two'], value_name='covar')

    pred_cov_final = pred_cov_final.assign(week=set_week, year=set_year,
                                           pred_vers=pred_vers, 
                                           ensemble_vers=ensemble_vers,
                                           std_dev_type=std_dev_type,
                                           covar_type=covar_type,
                                           full_model_rel_weight=full_model_rel_weight)

    return pred_cov_final


def get_mean_points(preds):
    preds = preds.set_index('player').rename_axis(None)
    mean_points = preds.loc[pred_cov.index.values, ['pos', 'team', 'pred_fp_per_game']].reset_index()
    mean_points = mean_points.rename(columns={'index': 'player'})
    mean_points.loc[mean_points.pos=='Defense', 'pos'] = 'DEF' 
    mean_points = mean_points.assign(week=set_week, year=set_year, 
                                    pred_vers=pred_vers, ensemble_vers=ensemble_vers,
                                    std_dev_type=std_dev_type,
                                    covar_type=covar_type,
                                    full_model_rel_weight=full_model_rel_weight,
                                    )

    return mean_points

#%%

covar_type = 'team_points_trunc'

# set the model version
set_weeks = [
    1, 2, 3, 4, 5, 6,
    7, 8
        ]

set_years = [
      2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022
]

pred_versions = [   
                'sera1_rsq0_brier1_matt1_lowsample_perc',
                 'sera1_rsq0_brier1_matt1_lowsample_perc',
                  'sera1_rsq0_brier1_matt1_lowsample_perc',
                   'sera1_rsq0_brier1_matt1_lowsample_perc',
                    'sera1_rsq0_brier1_matt1_lowsample_perc',
                     'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate',
                      'sera1_rsq0_brier1_matt0_lowsample_perc',
                       'sera1_rsq0_brier1_matt1_lowsample_perc',

]

ensemble_versions = [
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',

]

std_dev_types = [
                'pred_spline_class80_q80_matt1_brier1_kfold3', 
                'pred_spline_class80_q80_matt1_brier1_kfold3', 
                'pred_spline_class80_q80_matt1_brier1_kfold3', 
                'pred_spline_class80_q80_matt1_brier1_kfold3', 
                'pred_spline_class80_q80_matt1_brier1_kfold3', 
                'pred_spline_class80_q80_matt1_brier1_kfold3', 
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3'
]


sim_types = [
             'ownership_ln_pos_fix',
             'ownership_ln_pos_fix',
             'ownership_ln_pos_fix',
             'ownership_ln_pos_fix',
             'ownership_ln_pos_fix',
             'ownership_ln_pos_fix',
             'ownership_ln_pos_fix',
             'ownership_ln_pos_fix',
]

# # set the model version
# set_weeks = [
#     9
# ]

# set_years = [
#       2022
# ]

# pred_versions = [
#                 'sera1_rsq0_brier1_matt1_lowsample_perc',
                
# ]

# ensemble_versions = [
#                     'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
#  ]

# std_dev_types = [
#                 'pred_spline_class80_q80_matt1_brier1_kfold3', 
# ]


# sim_types = [
#              'ownership_ln_pos'
# ]

full_model_weights = [0.2, 1, 5]

i = 0
iter_cats = zip(set_weeks, set_years, pred_versions, ensemble_versions, std_dev_types)
for set_week, set_year, pred_vers, ensemble_vers, std_dev_type in iter_cats:

    print('Week:', set_week)

    for full_model_rel_weight in full_model_weights:

        # get the player and opposing player data to create correlation matrices
        player_data, _ = get_max_metrics(set_week, set_year)
        corr_data = create_pos_rank(player_data)
        opp_corr_data = create_pos_rank(player_data, opponent=True)
        opp_corr_data = opp_corr_data[~opp_corr_data.team.isnull()].reset_index(drop=True)
        corr_data = pd.concat([corr_data, opp_corr_data], axis=0)

        # use  team level data to get the covariance for team-level groups
        corr_data = get_team_totals(corr_data, 'pos_rank')
        corr_data = corr_data[['team', 'week', 'year', 'pos_rank', 'max_metric', 'y_act', 'team_total']]
        matrices, percs = get_group_covars(corr_data)

        # pull in the prediction data and create player matches for position type
        drop_teams = get_drop_teams(set_week, set_year)
        preds = get_predictions(drop_teams, pred_vers,std_dev_type, set_week, set_year, full_model_rel_weight)
        pred_cov = create_player_matches(preds, opponent=False)
        opp_pred_cov = create_player_matches(preds, opponent=True)
        pred_cov = pd.concat([pred_cov, opp_pred_cov], axis=0).reset_index(drop=True)

        # apply the clusters
        pred_cov = get_team_totals(pred_cov, 'pos_rank2')
        pred_cov = apply_group_covar(pred_cov, matrices, percs)
        
        pred_cov_final = cleanup_pred_covar(pred_cov)
        mean_points = get_mean_points(preds)

        if i == 0:
            dm.write_to_db(mean_points, 'Simulation', 'Covar_Means', 'replace')
            dm.write_to_db(pred_cov_final, 'Simulation', 'Covar_Matrix', 'replace')
            i += 1
        else:
            dm.write_to_db(mean_points, 'Simulation', 'Covar_Means', 'append')
            dm.write_to_db(pred_cov_final, 'Simulation', 'Covar_Matrix', 'append')


run_params = pd.DataFrame({
    'week': [set_week],
    'year': [set_year],
    'pred_vers': [pred_vers],
    'ensemble_vers': [ensemble_vers],
    'std_dev_type': [std_dev_type],
    'full_model_rel_weight': ['np.random.choice([0.2, 5], p=[0.5, 0.5])'],
    'drop_player_multiple': ['np.random.choice([0, 4], p=[0.5, 0.5])'],
    'covar_type': ["np.random.choice(['team_points_trunc'], p=[1])"],
    'use_covar': ["np.random.choice([True, False], p=[0.5, 0.5])"],
    'use_ownership': ['np.random.choice([True, False], p=[0.5, 0.5])'],
    'adjust_select': ["np.random.choice([True, False], p=[0.5, 0.5])"],
    'min_players_opp_team': ["np.random.choice([0, 'Auto'], p=[0.5, 0.5])"]
})

dm.delete_from_db('Simulation', 'Run_Params', f"week={set_week} AND year={set_year}")
dm.write_to_db(run_params, 'Simulation', 'Run_Params', 'append')

#%%
