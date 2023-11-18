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


def get_val_predictions(label, val_table, pred_version, ensemble_vers, set_week, set_year):

    df = dm.read(f'''SELECT player, 
                            team,
                            pos,
                            week, 
                            year, 
                            AVG(pred_fp_per_game) {label}
                     FROM {val_table}
                     WHERE pred_vers='{pred_version}'
                              AND reg_ens_vers='{ensemble_vers}' 
                              AND set_week={set_week}
                              AND set_year={set_year}
                     GROUP BY player, team, pos, week, year
                    ''', 'Validations')
    return df


def pull_actual_pts(set_pos):

    if set_pos=='Defense': pl = 'defTeam'
    else: pl = 'player'

    actual_pts = dm.read(f'''SELECT {pl} player, week, season year, fantasy_pts y_act
                             FROM {set_pos}_Stats 
                             WHERE season >= 2020''', 'FastR')
    return actual_pts


def add_actual(df):
    all_points = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
        cur_pts = pull_actual_pts(pos)
        all_points = pd.concat([all_points, cur_pts], axis=0)
    
    df = pd.merge(df, all_points, on=['player', 'week', 'year'])
    
    return df        


def create_pos_rank(df, opponent=False):

    if opponent:
        df = df.drop('team', axis=1).rename(columns={'defTeam': 'team'})
        prefix = 'Opp'
    else:
        prefix = ''

    df = df.sort_values(by=['team', 'pos', 'year', 'week', 'pred_fp_per_game'],
                        ascending=[True, True, True, True, False]).reset_index(drop=True)

    df['pos_rank'] = prefix + df.pos + df.groupby(['team', 'pos', 'year', 'week']).cumcount().apply(lambda x: str(x))

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


def get_predictions(drop_teams, pred_vers, reg_ens_vers, std_dev_type, set_week, set_year, full_model_rel_weight):

    preds = dm.read(f'''SELECT * 
                        FROM Model_Predictions 
                        WHERE pred_vers='{pred_vers}'
                              AND reg_ens_vers='{reg_ens_vers}'
                              AND std_dev_type='{std_dev_type}'
                              AND week = '{set_week}'
                              AND year = '{set_year}' 
                              AND player != 'Ryan Griffin'
                ''', 'Simulation')

    preds['weighting'] = 1
    preds.loc[preds.model_type=='full_model', 'weighting'] = full_model_rel_weight

    score_cols = ['pred_fp_per_game', 'std_dev']
    for c in score_cols: preds[c] = preds[c] * preds.weighting

    # Groupby and aggregate with namedAgg [1]:
    preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                                  'std_dev': 'sum',
                                                                  'weighting': 'sum',
                                                                  })

    for c in score_cols: preds[c] = preds[c] / preds.weighting
    preds = preds.drop('weighting', axis=1)

    teams = dm.read(f'''SELECT * 
                        FROM Player_Teams
                        WHERe week={set_week}
                             AND year={set_year}
                       ''', 'Simulation')
    preds = pd.merge(preds, teams, on=['player'])
    preds = preds[~preds.team.isin(drop_teams)].reset_index(drop=True)

    preds = preds.assign(week=set_week, year=set_year)
    
    return preds


def create_player_matches(preds, opponent=False):

    preds = create_pos_rank(preds)

    pred_cov1 = preds[['player', 'team', 'pos_rank']]
    pred_cov2 = preds[['player', 'team', 'pos_rank', 'std_dev', 'pred_fp_per_game', 'week', 'year']]

    if opponent:
        pred_cov2 = get_matchups(pred_cov2)
        pred_cov2 = pred_cov2.drop('team', axis=1).rename(columns={'defTeam': 'team'})
        pred_cov2.pos_rank = pred_cov2.pos_rank.apply(lambda x: 'Opp'+x)

    pred_cov = pd.merge(pred_cov1, pred_cov2, on='team')
    pred_cov.columns = ['player1', 'team', 'pos_rank1', 'player2', 'pos_rank2', 'std_dev', 'pred_fp_per_game', 'week', 'year']
    pred_cov = pred_cov[['player1', 'player2', 'team', 'pos_rank1', 'pos_rank2', 'std_dev', 'pred_fp_per_game', 'week', 'year']]


    return pred_cov


def get_matchups(df):
    matchups = dm.read('''SELECT offTeam team, defTeam, week, year
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

    df = pd.merge(df, matchups, on=['team', 'week', 'year'])
    return df



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
                                           reg_ens_vers=reg_ens_vers,
                                           std_dev_type=std_dev_type,
                                           covar_type=covar_type,
                                           full_model_rel_weight=full_model_rel_weight)
    
    pred_cov_final = pred_cov_final[pred_cov_final.covar != 0].reset_index(drop=True)


    return pred_cov_final


def get_mean_points(preds):
    preds = preds.set_index('player').rename_axis(None)
    mean_points = preds.loc[pred_cov.index.values, ['pos', 'team', 'pred_fp_per_game']].reset_index()
    mean_points = mean_points.rename(columns={'index': 'player'})
    mean_points.loc[mean_points.pos=='Defense', 'pos'] = 'DEF' 
    mean_points = mean_points.assign(week=set_week, year=set_year, 
                                    pred_vers=pred_vers, reg_ens_vers=reg_ens_vers,
                                    std_dev_type=std_dev_type,
                                    covar_type=covar_type,
                                    full_model_rel_weight=full_model_rel_weight,
                                    )

    return mean_points

def pivot_pos_data(df):

    # df = df[df['pos_rank'].isin(['QB0', 'RB0', 'RB1', 'WR0', 'WR1', 'WR2', 'TE0', 'Defense0', 
    #                             'OppQB0', 'OppDefense0', 'OppWR1'
    #                              #'OppRB0', 'OppWR0', 'OppWR1', 'OppTE0', 
    #                              ])]
    df = df[df['pos_rank'].isin(['QB0', 'RB0', 'WR0', 'WR1', 'TE0', 'Defense0', 'OppQB0' ])]
    df = df.pivot_table(index=['team', 'week', 'year'], columns='pos_rank', values='pred_fp_per_game')
    df = df.reset_index().sort_values(by=['team', 'year', 'week']).fillna(df.mean()).set_index(['team', 'week', 'year'])
    df = df.dropna(axis=0)
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

    lines = lines[['team', 'week', 'year', 'implied_points_for', 'implied_points_against']]
    lines[['implied_points_for', 'implied_points_against']] = lines[['implied_points_for', 'implied_points_against']]
    df = df.reset_index()
    df = pd.merge(df, lines, on=['team', 'week', 'year'], how='left')
    df = df.set_index(['team', 'week', 'year'])

    return df


def get_best_clusters(df, corr_data, min_n=5, max_n=12):

    from sklearn.metrics import davies_bouldin_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.cluster import KMeans

    X = df.values

    scores = []
    n_clusters = []
    for n in range(min_n, max_n):
        km = KMeans(n_clusters=n, random_state=1234, n_init='auto')
        km.fit(X)

        cur_score = np.round(davies_bouldin_score(X, km.labels_),3)
        scores.append(cur_score)
        n_clusters.append(n)

        # print(f'{n} Clusters:', cur_score)

    best_idx = np.argmin(scores)
    best_n = n_clusters[best_idx]

    print('Running best n:', best_n)
    km = KMeans(n_clusters=best_n, random_state=1234, n_init='auto')
    km.fit(X)

    df = df.reset_index()[['team', 'year', 'week']].drop_duplicates()
    df['label'] = km.labels_
    corr_data = pd.merge(df, corr_data, on=['team', 'week','year'])

    return corr_data, km

def show_cluster_heatmap(df, km):
    from collections import Counter
    import seaborn as sns

    heat_map = df[df['pos_rank'].isin(['QB0', 'RB0', 'RB1', 'WR0', 'WR1', 'TE0', 'Defense0', 
                                       'OppQB0', 'OppWR0', 'OppTE0', 'OppRB0', 'OppDefense0'])].copy()
    heat_map = pd.pivot_table(heat_map, index='label', columns='pos_rank', values='y_act', aggfunc='mean')

    print(Counter(km.labels_))
    sns.set(rc={'figure.figsize':(15,10)})
    sns.heatmap(heat_map, center=True, annot=True)


def show_historical_cluster_teams(df, label):
    return df.loc[df.label==label, ['team', 'week', 'year']].drop_duplicates() \
        .groupby(['team', 'year']).agg(cnts=('week', 'count')) \
            .sort_values(by=['year','cnts'], ascending=False)

def get_kmeans_covar(df):
    matrices = pd.DataFrame()
    for i in range(len(set(df.label))): 

        cor_matrix = df[df.label == i]
        cor_matrix = cor_matrix.pivot_table(index=['team', 'week', 'year'], columns='pos_rank', values='y_act').fillna(0)

        # calculate covariance matrix and convert to long format
        cov_matrix = cor_matrix.cov()
        cov_matrix = cov_matrix.rename_axis(None).rename_axis(None, axis=1)
        cov_matrix = cov_matrix.stack().reset_index()
        cov_matrix.columns = ['pos_rank1', 'pos_rank2', 'covariance']
        cov_matrix['label'] = i

        matrices = pd.concat([matrices, cov_matrix])

    return matrices

def assign_kmean_labels(pred_cov, pred_cov_stats, km):

    pred_cov_stats = pred_cov_stats.dropna()
    pred_cov_labels = pred_cov_stats.reset_index()[['team', 'week','year']].assign(label=km.predict(pred_cov_stats.values))
    pred_cov = pd.merge(pred_cov, pred_cov_labels, on=['team', 'week', 'year'])

    return pred_cov

def pred_covar_matrix(pred_cov, matrices):

    pred_cov = pd.merge(pred_cov, matrices, on=['pos_rank1', 'pos_rank2', 'label'], how='left')

    pred_cov.loc[pred_cov.player1==pred_cov.player2, 'covariance'] = pred_cov.loc[pred_cov.player1==pred_cov.player2, 'std_dev']**2
    pred_cov = pd.pivot_table(pred_cov, index='player1', columns='player2', values='covariance').fillna(0)
    pred_cov = pred_cov.rename_axis(None).rename_axis(None, axis=1)

    return pred_cov




#%%


import itertools

covar_type = 'kmeans_pred_trunc_new'

# # set the model version
# set_weeks = [
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
#     1, 2, 3, 4, 5, 6
# ]

# set_years = [
#       2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 
#       2023, 2023, 2023, 2023, 2023, 2023
# ]

set_weeks = [11]
set_years = [2023,2023]

pred_versions = [
                 'sera0_rsq0_mse1_brier1_matt1_bayes',
                #  'sera1_rsq0_mse0_brier1_matt0_bayes',
                #  'sera0_rsq0_mse1_brier1_matt0_bayes',
                 ]

reg_ens_versions = [
                    # 'random_sera0_rsq0_mse1_include2_kfold3',
                    # 'random_kbest_sera0_rsq0_mse1_include2_kfold3',
                    # 'kbest_sera0_rsq0_mse1_include2_kfold3',
                    'random_full_stack_sera0_rsq0_mse1_include2_kfold3',
                    # 'random_sera1_rsq0_mse0_include2_kfold3',
                    # 'random_kbest_sera1_rsq0_mse0_include2_kfold3',
                    # 'random_full_stack_sera0_rsq1_mse0_include2_kfold3',
                    # 'random_full_stack_sera0_rsq0_mse1_include2_kfold3_rand'
                ]


std_dev_types = [
                 'spline_pred_class80_q80_matt0_brier1_kfold3',
                 'spline_pred_class80_matt0_brier1_kfold3',
                 'spline_pred_q80_matt0_brier1_kfold3',
                 'spline_class80_q80_matt0_brier1_kfold3',
                #  'spline_pred_class80_q80_matt1_brier1_kfold3',
                #  'spline_pred_class80_matt1_brier1_kfold3',
                #  'spline_pred_q80_matt1_brier1_kfold3',
                #  'spline_class80_q80_matt1_brier1_kfold3',
                #   'spline_pred_class80_q80_matt1_brier5_kfold3',
                #  'spline_pred_class80_matt1_brier5_kfold3',
                #  'spline_pred_q80_matt1_brier5_kfold3',
                #  'spline_class80_q80_matt1_brier5_kfold3',
                 ]

full_model_weights = [0.2, 5]

iter_cats = list(set(itertools.product(pred_versions, reg_ens_versions, std_dev_types, full_model_weights)))
iter_cats = pd.DataFrame(iter_cats).sort_values(by=[0, 1]).values


i = 10
for set_week, set_year in zip(set_weeks, set_years):
    print(set_week, set_year)
    for pred_vers, reg_ens_vers, std_dev_type, full_model_rel_weight in iter_cats:

        # get the player and opposing player data to create correlation matrices
        player_data = get_val_predictions('pred_fp_per_game', 'Model_Validations', pred_vers, reg_ens_vers, set_week, set_year)
        player_data = add_actual(player_data)
        player_data = get_matchups(player_data)

        # create the rankings for self team and opposing team
        corr_data = create_pos_rank(player_data)
        opp_corr_data = create_pos_rank(player_data, opponent=True)
        opp_corr_data = opp_corr_data[~opp_corr_data.team.isnull()].reset_index(drop=True)
        corr_data = pd.concat([corr_data, opp_corr_data], axis=0)

        kmean_input = pivot_pos_data(corr_data)
        kmean_input = add_gambling_lines(kmean_input)

        kmeans_out, km = get_best_clusters(kmean_input, corr_data, min_n=8, max_n=16)
        # show_cluster_heatmap(kmeans_out, km)
        # show_historical_cluster_teams(kmeans_out, label=3)
        matrices = get_kmeans_covar(kmeans_out)

        # pull in the prediction data and create player matches for position type
        drop_teams = get_drop_teams(set_week, set_year)
        preds = get_predictions(drop_teams, pred_vers, reg_ens_vers, std_dev_type, set_week, set_year, full_model_rel_weight)
        pred_cov = create_player_matches(preds, opponent=False)
        opp_pred_cov = create_player_matches(preds, opponent=True)
        pred_cov = pd.concat([pred_cov, opp_pred_cov], axis=0).reset_index(drop=True)

        pred_cov_stats = pred_cov[['player2', 'team', 'pos_rank2', 'week', 'year', 'pred_fp_per_game']]
        pred_cov_stats.columns = ['player', 'team', 'pos_rank', 'week', 'year', 'pred_fp_per_game']
        pred_cov_stats = pivot_pos_data(pred_cov_stats)
        pred_cov_stats = add_gambling_lines(pred_cov_stats)

        pred_cov = assign_kmean_labels(pred_cov, pred_cov_stats, km)
        pred_cov = pred_covar_matrix(pred_cov, matrices)

        pred_cov_final = cleanup_pred_covar(pred_cov)        
        mean_points = get_mean_points(preds)

        if i == 0:
            dm.write_to_db(mean_points, 'Simulation', 'Covar_Means', 'replace')
            dm.write_to_db(pred_cov_final, 'Simulation', 'Covar_Matrix', 'replace')
            i += 1
        else:
            del_str = f'''week={set_week} 
                          AND year={set_year} 
                          AND pred_vers='{pred_vers}' 
                          AND reg_ens_vers='{reg_ens_vers}' 
                          AND std_dev_type='{std_dev_type}' 
                          AND covar_type='{covar_type}' 
                          AND full_model_rel_weight={full_model_rel_weight}'''
            
            dm.delete_from_db('Simulation', 'Covar_Means', del_str, create_backup=False)
            dm.delete_from_db('Simulation', 'Covar_Matrix', del_str, create_backup=False)
            dm.write_to_db(mean_points, 'Simulation', 'Covar_Means', 'append')
            dm.write_to_db(pred_cov_final, 'Simulation', 'Covar_Matrix', 'append')


# %%
teams = dm.read(f'''SELECT * 
                        FROM Player_Teams
                       ''', 'Simulation').rename(columns={'team': 'past_teams'})
corr_data = pd.merge(corr_data, teams, on=['player', 'week', 'year'])

#%%
pl1 = corr_data.loc[corr_data.team==corr_data.past_teams, 
                 ['player', 'week', 'year', 'past_teams', 'y_act']].rename(columns={'player': 'player1', 'y_act': 'y_act1'})
pl2 = corr_data.loc[corr_data.team==corr_data.past_teams, 
                  ['player', 'week', 'year', 'past_teams', 'y_act']].rename(columns={'player': 'player2', 'y_act': 'y_act2'})
compare = pd.merge(pl1, pl2, on=['week','year', 'past_teams'])
compare = compare[compare.player1!=compare.player2].reset_index(drop=True)
cnts = compare.groupby(['player1', 'player2']).agg(game_cnts=('week', 'count')).reset_index()
compare = pd.merge(compare, cnts, on=['player1', 'player2'])
compare = compare[compare.game_cnts > 16]
compare = compare.groupby(['player1', 'player2'])[['y_act1', 'y_act2']].cov()['y_act1'].unstack()['y_act2']
compare = compare.reset_index().rename(columns={'player1': 'player', 'player2': 'player_two'}).sort_values(by='y_act2')

# pred_cov_final = dm.read("SELECT * FROM Covar_Matrix WHERE full_model_rel_weight=1 AND week=9", 'Simulation')

compare = pd.merge(compare, pred_cov_final, on=['player', 'player_two'])
compare = compare[compare.covar!=0]

# %%

from sklearn.metrics import mean_squared_error, r2_score

print(np.sqrt(mean_squared_error(compare.y_act2, compare.covar)))
print(compare.corr()['covar']['y_act2'])
print(r2_score(compare.y_act2, compare.covar))

compare.plot.scatter(x='covar', y='y_act2')

# %%

compare[(compare.covar > 20)]
# %%
