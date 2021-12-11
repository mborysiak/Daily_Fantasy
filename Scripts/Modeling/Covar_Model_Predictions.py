#%%

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
import pandas as pd

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


def rolling_max_std(pos):

    if pos != 'Defense':
        df = dm.read(f'''SELECT player, team, week, season year, fantasy_pts y_act
                        FROM {pos}_Stats
                        WHERE season >= 2020
                            AND week < 17 ''', 'FastR')
    else:
        df = dm.read(f'''SELECT defTeam player, defTeam team, week, season year, fantasy_pts y_act
                         FROM {pos}_Stats
                         WHERE season >= 2020
                                AND week < 17 ''', 'FastR')

    df = df.sort_values(by=['player', 'year', 'week']).reset_index(drop=True)
    df['roll_pts'] = df.y_act#df.groupby(['player'])['y_act'].shift(1)
    df['roll_max'] = df.groupby('player')['roll_pts'].rolling(8, min_periods=1).apply(lambda x: pd.Series(x).nlargest(2).iloc[-1]).values

    df['roll_mean'] = df.groupby('player')['roll_pts'].rolling(9, min_periods=1).apply(lambda x: pd.Series(x).nlargest(5).iloc[-1]).values
    df['roll_std'] = df.roll_max - df.roll_mean
    df = df.drop(['roll_pts', 'roll_mean'], axis=1)
    
    most_recent = df.drop_duplicates(subset=['player'], keep='last')
    most_recent = most_recent[['player', 'roll_max', 'roll_std']]

    return df, most_recent


def projection_data(pos):

    if pos == 'Defense':

        # pull in the salary and actual results data
        proj = dm.read('''SELECT team player, team, week, year, projected_points
                            FROM FantasyPros
                            WHERE pos='DST'
                                  AND week < 17 ''', 'Pre_PlayerData')

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
                            WHERE week < 17
                                  AND position='{pos.lower()}' 
                                  ''', 'Pre_PlayerData')

    return proj


def get_std_splines(pos, show_plot=False, k=2, s=2000):
    

    proj = projection_data(pos)

    if pos=='QB':
        proj = proj[(proj.ProjPts > 8) & (proj.projected_points > 8)].reset_index(drop=True)

    # get the rolling stats data    
    stats, _ = rolling_max_std(pos)

    # join together and calculate sd and max metrics
    df = pd.merge(proj, stats, on=['player', 'team', 'week', 'year'])
    df = create_sd_max_metrics(df)

    # create the groups    
    df = df.dropna()
    min_grps = int(df.shape[0] / 100)
    max_grps = int(df.shape[0] / 40)

    splines = {}
    X_max = {}
    y_max = {}
    max_r2 = {}
    for x_val, met in zip(['sd_metric', 'max_metric'], ['std_dev', 'perc_99']):
        
        df = df.sort_values(by=x_val).reset_index(drop=True)

        max_r2[met] = 0
        for num_grps in range(min_grps, max_grps, 1):

            # create equal sizes groups going down the dataframe ordered by each metric
            df_len = len(df)
            repeats = math.ceil(df.shape[0] / num_grps)
            grps = np.repeat([i for i in range(num_grps)], repeats)
            df['grps'] = grps[:df_len]

            # calculate the standard deviations of each group
            std_devs = df.groupby('grps').agg({'y_act': [np.std, lambda x: np.percentile(x, 99)],
                                               'sd_metric': 'mean',
                                               'max_metric': 'mean',
                                               'player': 'count'})
            std_devs.columns = ['std_dev', 'perc_99', 'sd_metric', 'max_metric', 'player_cnts']

            # fit a spline to the group datasets
            X = std_devs[[x_val]]
            y = std_devs[[met]]
            spl = UnivariateSpline(X, y, k=k, s=s)

            r2 = r2_score(y, spl(X))
            if r2 > max_r2[met]:
                max_r2[met] = r2
                splines[met] = spl
                X_max[met] = X
                y_max[met] = y

        if show_plot:

            print(met)
            X_pred = list(range(int(X_max[met].min()), int(X_max[met].max()+3), 1))
            plt.scatter(X, y)
            plt.scatter(X_max[met], y_max[met])
            plt.plot(X_pred, splines[met](X_pred), 'g', lw=3)
            plt.show()


    return splines['std_dev'], splines['perc_99'] 


#%%

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


def get_max_metrics():

    corr_data = pd.DataFrame()

    for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:

        proj = projection_data(pos)

        if pos=='QB':
            proj = proj[(proj.ProjPts > 8) & (proj.projected_points > 8)].reset_index(drop=True)

        # get the rolling stats data    
        stats, _ = rolling_max_std(pos)

        # join together and calculate sd and max metrics
        df = pd.merge(proj, stats, on=['player', 'team', 'week', 'year'])
        df = create_sd_max_metrics(df)

        df['pos'] = pos
        corr_data = pd.concat([corr_data, df], axis=0)

    return corr_data


def create_pos_rank(df):

    df = df.sort_values(by=['team', 'pos', 'year', 'week', 'max_metric'],
                        ascending=[True, True, True, True, False]).reset_index(drop=True)

    df['pos_rank'] = df.pos + df.groupby(['team', 'pos', 'year', 'week']).cumcount().apply(lambda x: str(x))

    return df

def get_team_totals(df, col):
    team_totals = df[df[col].isin(['QB0', 'RB0', 'RB1', 'WR0', 'WR1', 'WR2', 'TE0'])]
    team_totals = team_totals[['team', 'week', 'year', 'max_metric']].drop_duplicates().reset_index(drop=True)
    team_totals = team_totals.groupby(['week', 'year', 'team']).agg(team_total=('max_metric', 'sum')).reset_index()

    df = pd.merge(df, team_totals, on=['week','year','team'])

    return df

player_data = get_max_metrics()
corr_data = create_pos_rank(player_data)
corr_data = get_team_totals(corr_data, 'pos_rank')
corr_data = corr_data[['team', 'week', 'year', 'pos_rank', 'max_metric', 'y_act', 'team_total']]

# %%
matrices = pd.DataFrame()
percs = np.percentile(corr_data.team_total, [0, 25, 50, 75, 100])
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

# %%

# set year to analyze
set_year = 2021
set_week = 14

# full_model or backfill
vers = 'standard'
drop_teams = ['MIN', 'PIT', 'CHI', 'GB', 'LAR', 'ARI']


def get_predictions(drop_teams, vers, set_week, set_year):

    preds = dm.read(f'''SELECT * 
                        FROM Model_Predictions 
                        WHERE version='{vers}'
                            AND week = '{set_week}'
                            AND year = '{set_year}' 
                            AND player != 'Ryan Griffin'
                ''', 'Simulation')

    preds['weighting'] = 1
    preds.loc[preds.model_type=='full_model', 'weighting'] = 1

    score_cols = ['pred_fp_per_game', 'std_dev']
    for c in score_cols: preds[c] = preds[c] * preds.weighting

    # Groupby and aggregate with namedAgg [1]:
    preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                                  'std_dev': 'sum',
                                                                  'weighting': 'sum'})

    for c in score_cols: preds[c] = preds[c] / preds.weighting
    preds = preds.drop('weighting', axis=1)

    teams = dm.read("SELECT * FROM Player_Teams", 'Simulation')
    preds = pd.merge(preds, teams, on=['player'])
    preds = preds[~preds.team.isin(drop_teams)].reset_index(drop=True)
    
    return preds


def create_player_matches(preds):

    preds = create_pos_rank(preds)

    pred_cov = preds[['player', 'team', 'pos_rank', 'std_dev', 'max_metric']]
    pred_cov = pd.merge(pred_cov.drop(['std_dev', 'max_metric'], axis=1), pred_cov, on='team')
    pred_cov.columns = ['player1', 'team', 'pos_rank1', 'player2', 'pos_rank2', 'std_dev', 'max_metric']
    pred_cov = pred_cov[['player1', 'player2', 'team', 'pos_rank1', 'pos_rank2', 'std_dev', 'max_metric']]
    pred_cov['week'] = set_week
    pred_cov['year'] = set_year

    return pred_cov


preds = get_predictions(drop_teams, vers, set_week, set_year)
max_metrics = player_data.loc[(player_data.week==set_week) & (player_data.year==set_year), 
                             ['player', 'pos', 'week', 'year', 'max_metric']]
preds = pd.merge(preds, max_metrics, on=['player', 'pos'], how='left')
preds.max_metric = preds.max_metric.fillna(np.percentile(preds.max_metric, 25))
preds = preds.assign(week=set_week).assign(year=set_year)
pred_cov = create_player_matches(preds)
pred_cov = get_team_totals(pred_cov, 'pos_rank2')

pred_cov['perc'] = 0
for i in range(len(percs) - 1): 

    perc_low = percs[i]
    perc_high =  percs[i+1]
    pred_cov.loc[(pred_cov.team_total > perc_low) & (pred_cov.team_total <= perc_high), ['perc']] = i

pred_cov.loc[(pred_cov.team_total > perc_high), ['perc']] = i

pred_cov = pd.merge(pred_cov, matrices, on=['pos_rank1', 'pos_rank2', 'perc'])

pred_cov.loc[pred_cov.player1==pred_cov.player2, 'covariance'] = pred_cov.loc[pred_cov.player1==pred_cov.player2, 'std_dev']**2
pred_cov = pd.pivot_table(pred_cov, index='player1', columns='player2', values='covariance').fillna(0)
pred_cov = pred_cov.rename_axis(None).rename_axis(None, axis=1)

import numpy as np

def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

pred_cov_final = pd.DataFrame(get_near_psd(pred_cov))
pred_cov_final.columns = pred_cov.columns
pred_cov_final.index = pred_cov.index
pred_cov_final = pred_cov_final.reset_index().rename(columns={'index': 'player'})

preds = preds.set_index('player').rename_axis(None)
mean_points = preds.loc[pred_cov.index.values, ['pos', 'team', 'pred_fp_per_game']].reset_index()
mean_points = mean_points.rename(columns={'index': 'player'})
mean_points.loc[mean_points.pos=='Defense', 'pos'] = 'DEF' 

dm.write_to_db(mean_points, 'Simulation', 'Covar_Means', 'replace')
dm.write_to_db(pred_cov_final, 'Simulation', 'Covar_Matrix', 'replace')

#%%

import scipy.stats as ss
dist = ss.multivariate_normal(mean=mean_points.pred_fp_per_game.values, 
                              cov=pred_cov_final.drop('player', axis=1), allow_singular=True)
predictions = pd.DataFrame(dist.rvs(1000)).T
predictions.index = pred_cov_final.index

# predictions = pd.merge(preds[['pos']], predictions, left_index=True, right_index=True)
# predictions = predictions.reset_index().rename(columns={'index': 'player'})
# %%
dm.write_to_db(predictions, 'Simulation', f'week{set_week}_year{set_year}', 'replace')

# %%
xx = predictions.copy()
xx = xx.T
xx.plot.scatter(x='Justin Herbert', y='Mike Williams')
# %%
yy = predictions.copy()
yy = yy[~yy.pos.str.contains('FLEX')].drop(['pos'], axis=1).set_index('player').T
std_dev = pd.DataFrame(yy.std(axis=0))

yy = yy.corr() 
name = 'Justin Herbert'
yy = yy[[name]].sort_values(by=name, ascending=False).iloc[:25]
yy = pd.merge(yy, std_dev, left_index=True, right_index=True)
yy['cov'] = yy['Justin Herbert'] * yy[0] * 10
yy
# %%
