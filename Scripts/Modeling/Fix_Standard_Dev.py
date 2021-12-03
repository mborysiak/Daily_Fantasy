#%%

def create_sd_max_metrics(df):
 
    df['sd_metric'] = (df.fantasyPoints + df.ProjPts + \
                        df.projected_points + df.roll_std) / 4
    df['max_metric'] = (df.fantasyPoints + df.ProjPts + \
                        df.projected_points + df.roll_max) / 4
    
    df = df.drop(['fantasyPoints', 'ProjPts', 'projected_points', 
                    'roll_std', 'roll_max'], axis=1)

    return df


def rolling_max_std(pos):

    from ff.db_operations import DataManage   
    import ff.general as ffgeneral 
    import pandas as pd

    # set the root path and database management object
    root_path = ffgeneral.get_main_path('Daily_Fantasy')
    db_path = f'{root_path}/Data/Databases/'
    dm = DataManage(db_path)

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

    # # log test
    # import numpy as np
    # df = df[df.y_act > 1].reset_index(drop=True)
    # df.y_act = np.log(df.y_act)

    df = df.sort_values(by=['player', 'year', 'week']).reset_index(drop=True)
    df['roll_pts'] = df.y_act#df.groupby(['player'])['y_act'].shift(1)
    df['roll_max'] = df.groupby('player')['roll_pts'].rolling(8, min_periods=1).apply(lambda x: pd.Series(x).nlargest(2).iloc[-1]).values
    
    # df['roll_std'] = df.groupby('player')['roll_pts'].rolling(8, min_periods=1).std().values
    # df = df.drop(['roll_pts'], axis=1)

    df['roll_mean'] = df.groupby('player')['roll_pts'].rolling(9, min_periods=1).apply(lambda x: pd.Series(x).nlargest(5).iloc[-1]).values
    df['roll_std'] = df.roll_max - df.roll_mean
    df = df.drop(['roll_pts', 'roll_mean'], axis=1)
    
    most_recent = df.drop_duplicates(subset=['player'], keep='last')
    most_recent = most_recent[['player', 'roll_max', 'roll_std']]

    return df, most_recent


def get_std_splines(pos, show_plot=False, k=2, s=2000):
    
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
    
    if pos == 'Defense':

        # pull in the salary and actual results data
        proj = dm.read('''SELECT team player, team, week, year, projected_points
                            FROM FantasyPros
                            WHERE pos='DST'
                                    AND week < 17 ''', 'Pre_PlayerData')

        proj_2 = dm.read('''SELECT offteam player, offTeam team, week, year, fantasyPoints, `Proj Pts` ProjPts 
                            FROM PFF_Expert_Ranks
                            JOIN (SELECT offteam, week, year, fantasyPoints
                                  FROM PFF_Proj_Ranks)
                                  USING (offTeam, week, year)''', 'Pre_TeamData')
        proj = pd.merge(proj, proj_2, on=['player', 'team', 'week', 'year'])
    
    else:
        # pull in the salary and actual results data
        proj = dm.read('''SELECT player, offTeam team, week, year, projected_points, fantasyPoints, ProjPts
                            FROM PFF_Proj_Ranks
                            JOIN (SELECT player, team offTeam, week, year, projected_points 
                                  FROM FantasyPros)
                                  USING (player, offTeam, week, year)
                            JOIN (SELECT player, offTeam, week, year, `Proj Pts` ProjPts 
                                  FROM PFF_Expert_Ranks)
                                  USING (player, offTeam, week, year)
                            WHERE week < 17''', 'Pre_PlayerData')

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


corr_data = pd.DataFrame()

for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:

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
        proj = dm.read('''SELECT player, offTeam team, week, year, defTeam, projected_points, fantasyPoints, ProjPts
                            FROM PFF_Proj_Ranks
                            JOIN (SELECT player, team offTeam, week, year, projected_points 
                                    FROM FantasyPros)
                                    USING (player, offTeam, week, year)
                            JOIN (SELECT player, offTeam,  week, year, `Proj Pts` ProjPts 
                                    FROM PFF_Expert_Ranks)
                                    USING (player, offTeam, week, year)
                            WHERE week < 17''', 'Pre_PlayerData')

    if pos=='QB':
        proj = proj[(proj.ProjPts > 8) & (proj.projected_points > 8)].reset_index(drop=True)


    # get the rolling stats data    
    stats, _ = rolling_max_std(pos)

    # join together and calculate sd and max metrics
    df = pd.merge(proj, stats, on=['player', 'team', 'week', 'year'])
    df = create_sd_max_metrics(df)

    df['pos'] = pos
    corr_data = pd.concat([corr_data, df], axis=0)

corr_data = corr_data.sort_values(by=['team', 'pos', 'year', 'week', 'max_metric'],
                                  ascending=[True, True, True, True, False]).reset_index(drop=True)
corr_data['pos_rank'] = corr_data.pos + corr_data.groupby(['team', 'pos', 'year', 'week']).cumcount().apply(lambda x: str(x))

team_totals = corr_data[corr_data.pos_rank.isin(['QB0', 'RB0', 'RB1', 'WR0', 'WR1', 'WR2', 'TE1', 'TE2'])]
team_totals = team_totals.groupby(['week', 'year', 'team']).agg(team_total=('max_metric', 'sum')).reset_index()

corr_data = pd.merge(corr_data, team_totals, on=['week','year','team'])

# opposing_team = corr_data[['team', 'defTeam', 'week', 'year', 'pos_rank', 'max_metric', 'y_act']].copy()
# opposing_team.loc[:, 'team'] = opposing_team.loc[:, 'defTeam']
# opposing_team = opposing_team.drop('defTeam', axis=1).dropna()
# opposing_team['pos_rank'] = 'Opp' + opposing_team['pos_rank']

# corr_data = corr_data[['team', 'week', 'year', 'pos_rank', 'max_metric', 'y_act']]
# corr_data = pd.concat([corr_data, opposing_team], axis=0)

# %%
matrices = []
percs = [0, 25, 50, 75, 100]
for i in range(len(percs) - 1): 
    perc_low = np.percentile(corr_data.team_total, percs[i])
    perc_high = np.percentile(corr_data.team_total, percs[i+1])

    cor_matrix = corr_data[(corr_data.team_total > perc_low) & (corr_data.team_total <= perc_high)]
    cor_matrix = cor_matrix.pivot_table(index=['team', 'week', 'year'], columns='pos_rank', values='y_act').fillna(0)

    # calculate covariance matrix and convert to long format
    cov_matrix = cor_matrix.cov()
    matrices.append(cov_matrix)
    # cov_matrix = cov_matrix.rename_axis(None).rename_axis(None, axis=1)
    # cov_matrix = cov_matrix.stack().reset_index()
    # cov_matrix.columns = ['pos_rank1', 'pos_rank2', 'covariance']

# %%

# set year to analyze
set_year = 2021
set_week = 12

# full_model or backfill
vers = 'roll8_fullhist_kbestallstack_WRTEDEFkeep25_QBRBdrophalf'

preds = dm.read(f'''SELECT * 
                    FROM Model_Predictions 
                    WHERE version='{vers}'
                          AND week = '{set_week}'
                          AND year = '{set_year}' 
                          AND player != 'Ryan Griffin'
            ''', 'Simulation')

preds['weighting'] = 1
preds.loc[preds.model_type=='full_model', 'weighting'] = 1

score_cols = ['pred_fp_per_game', 'std_dev', 'max_score']
for c in score_cols: preds[c] = preds[c] * preds.weighting

# Groupby and aggregate with namedAgg [1]:
preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                              'std_dev': 'sum',
                                                              'max_score': 'sum',
                                                              'min_score': 'sum',
                                                              'weighting': 'sum'})



for c in score_cols: preds[c] = preds[c] / preds.weighting
preds = preds.drop('weighting', axis=1)


drop_teams = ['DET','CHI', 'LVR','DAL', 'BUF', 'NO', 'CLE', 'BAL', 'SEA', 'WAS']

teams = dm.read(f'''SELECT player, team
                    FROM (
                    SELECT CASE WHEN pos!='DST' THEN player ELSE team END player, 
                        team,
                        row_number() OVER (PARTITION BY player ORDER BY projected_points DESC) rn 
                    FROM FantasyPros
                    WHERE week={set_week} AND year={set_year}
                    ) WHERE rn=1''', 'Pre_PlayerData')

# dm.delete_from_db('Simulation', 'Player_Teams', f"week={set_week} AND year={set_year}")
dm.write_to_db(teams, 'Simulation', 'Player_Teams', 'replace')

preds = pd.merge(preds, teams, on=['player'])
preds = preds[~preds.team.isin(drop_teams)].reset_index(drop=True)


preds = preds.sort_values(by=['team', 'pos', 'pred_fp_per_game'],
                                ascending=[True, True, False]).reset_index(drop=True)
preds['pos_rank'] = preds.pos + preds.groupby(['team', 'pos']).cumcount().apply(lambda x: str(x))

pred_cov = preds[['player', 'team', 'pos_rank', 'std_dev']]
pred_cov = pd.merge(pred_cov.drop('std_dev', axis=1), pred_cov, on='team')
pred_cov.columns = ['player1', 'team', 'pos_rank1', 'player2', 'pos_rank2', 'std_dev']
pred_cov = pred_cov[['player1', 'player2', 'team', 'pos_rank1', 'pos_rank2', 'std_dev']]

pred_cov = pd.merge(pred_cov, cov_matrix, on=['pos_rank1', 'pos_rank2'])
pred_cov.loc[pred_cov.player1==pred_cov.player2, 'covariance'] = pred_cov.loc[pred_cov.player1==pred_cov.player2, 'std_dev']**2
# pred_cov = pred_cov.sort_values(by=['team', 'pos_rank1', 'pos_rank2']).reset_index(drop=True)
pred_cov = pd.pivot_table(pred_cov, index='player1', columns='player2', values='covariance').fillna(0)
# pred_cov = pred_cov.groupby(['player1', 'player2'], sort=False)['covariance'].sum().unstack('player2').fillna(0)
pred_cov = pred_cov.rename_axis(None).rename_axis(None, axis=1)

import numpy as np

def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

pred_cov2 = pd.DataFrame(get_near_psd(pred_cov))
pred_cov2.columns = pred_cov.columns
pred_cov2.index = pred_cov.index

import scipy.stats as ss
preds = preds.set_index('player').rename_axis(None)
mean_points = preds.loc[pred_cov.index.values, 'pred_fp_per_game'].values


dist = ss.multivariate_normal(mean=mean_points, cov=pred_cov2, allow_singular=True)
predictions = pd.DataFrame(dist.rvs(1000)).T
predictions.index = pred_cov.index
predictions = pd.merge(preds[['pos']], predictions, left_index=True, right_index=True)
predictions = predictions.reset_index().rename(columns={'index': 'player'})
# %%
dm.write_to_db(predictions, 'Simulation', f'week{set_week}_year{set_year}', 'replace')

# %%
xx = predictions.copy()
xx = xx[~xx.pos.str.contains('FLEX')].drop(['pos'], axis=1).set_index('player').T
xx.plot.scatter(x='Aaron Rodgers', y='Randall Cobb')

# %%
