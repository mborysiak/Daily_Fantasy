#%%

from ff.db_operations import DataManage   
import ff.general as ffgeneral 
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)



def create_sd_max_metrics(df, sd_cols, max_cols):

    df['sd_metric'] = df[sd_cols].mean(axis=1)
    df['max_metric'] = df[max_cols].mean(axis=1)
    
    df = df[[c for c in df.columns if c not in max_cols and c not in sd_cols]]

    return df


def pull_actual_stats(pos, week, year):
    if pos=='Defense': player = 'defTeam'; team = 'defTeam'
    else: player = 'player'; team = 'team'

    df = dm.read(f'''SELECT {player} player, {team} team, week, season year, fantasy_pts y_act
                     FROM {pos}_Stats
                     WHERE season >= 2020
                           AND (
                                (week < {week} AND season = {year})
                                OR 
                                (season < {year})
                                )
                     ORDER BY {player}, year, week
                  ''', 'FastR')
    return df
    
def pull_projections(pos):

    if pos == 'Defense':

        # pull in the salary and actual results data
        proj = dm.read('''SELECT team player, team, week, year, projected_points
                            FROM FantasyPros
                            WHERE pos='DST'
                                   ''', 'Pre_PlayerData')

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
                            ''', 'Pre_PlayerData')

        if pos=='QB':
            proj = proj[(proj.ProjPts > 8) & (proj.projected_points > 8)].reset_index(drop=True)

    return proj


def model_predictions(pos, week, year):
    
    df = dm.read(f'''SELECT player, week, year, AVG(pred_fp_per_game) pred_fp_per_game
                     FROM Model_Predictions
                     WHERE year >= 2020
                           AND (
                                (week < {week} AND year = {year})
                                OR 
                                (year < {year})
                                )
                           AND version LIKE 'standard%'
                           AND ensemble_vers LIKE 'no_weight%'
                           AND pos='{pos}'
                           AND player!='Ryan Griffin'
                     GROUP BY player, week, year
                        ''', 'Simulation')

    return df

def rolling_max_std(df):

    df['roll_pts'] = df.y_act
    df['roll_max'] = df.groupby('player')['roll_pts'].rolling(8, min_periods=1).apply(lambda x: pd.Series(x).nlargest(2).iloc[-1]).values
    df['roll_mean'] = df.groupby('player')['roll_pts'].rolling(9, min_periods=1).apply(lambda x: pd.Series(x).nlargest(5).iloc[-1]).values
    df['roll_std'] = df.roll_max - df.roll_mean
    df = df.drop(['roll_pts', 'roll_mean'], axis=1)
    return df

def recent_roll_stats(df):
    most_recent = df.drop_duplicates(subset=['player'], keep='last')
    most_recent = most_recent[['player', 'roll_max', 'roll_std']]
    return most_recent


def create_groups(df, num_grps):
    # create equal sizes groups going down the dataframe ordered by each metric
    df_len = len(df)
    repeats = math.ceil(df.shape[0] / num_grps)
    grps = np.repeat([i for i in range(num_grps)], repeats)
    df['grps'] = grps[:df_len]
    return df

def show_spline_fit(splines, met, X, y, X_max, y_max):
    print(met)
    X_pred = list(range(int(X_max[met].min()), int(X_max[met].max()+3), 1))
    plt.scatter(X, y)
    plt.scatter(X_max[met], y_max[met])
    plt.plot(X_pred, splines[met](X_pred), 'g', lw=3)
    plt.show()

def get_std_splines(pos, week, year, sd_cols, max_cols, show_plot=False, k=2, s=2000):
    
    # pull actual stats, rolling stats, and projections
    actual_stats = pull_actual_stats(pos, week, year)
    stats = rolling_max_std(actual_stats)
    
    if 'pred_fp_per_game' in sd_cols:
        proj = model_predictions(pos, week, year)
    else:
        proj = pull_projections(pos)

    # join together and calculate sd and max metrics
    df = pd.merge(proj, stats, on=['player', 'week', 'year'])

    # calculate sd and max metrics
    df = create_sd_max_metrics(df, sd_cols, max_cols)

    # create the groups    
    df = df.dropna()
    min_grps = int(df.shape[0] / 100)
    max_grps = int(df.shape[0] / 60)

    splines = {}; X_max = {}; y_max = {}; max_r2 = {}
    for x_val, met in zip(['sd_metric', 'max_metric'], ['std_dev', 'perc_99']):
        
        df = df.sort_values(by=x_val).reset_index(drop=True)

        max_r2[met] = 0
        for num_grps in range(min_grps, max_grps, 1):

            # create the groups to aggregate for std dev and max metrics
            df = create_groups(df, num_grps)

            # calculate the standard deviation and max of each group
            Xy = df.groupby('grps').agg({'y_act': [np.std, lambda x: np.percentile(x, 99)],
                                         'sd_metric': 'mean',
                                         'max_metric': 'mean',
                                         'player': 'count'})
            Xy.columns = ['std_dev', 'perc_99', 'sd_metric', 'max_metric', 'player_cnts']

            # fit a spline to the group datasets
            X = Xy[[x_val]]
            y = Xy[[met]]
            spl = UnivariateSpline(X, y, k=k, s=s)

            r2 = r2_score(y, spl(X))
            if r2 > max_r2[met]:
                max_r2[met] = r2
                splines[met] = spl
                X_max[met] = X
                y_max[met] = y

        if show_plot:
            show_spline_fit(splines, met, X, y, X_max, y_max)
            
    return splines['std_dev'], splines['perc_99'] 

