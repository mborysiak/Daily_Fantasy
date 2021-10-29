#%%

def create_sd_max_metrics(df, defense=False):

    if defense: 
        df['sd_metric'] = df.projected_points
        df['max_metric'] = df.projected_points
        df = df.drop(['projected_points', 'roll_max', 'roll_std'], axis=1)

    else: 
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
    
    df = df.sort_values(by=['player', 'year', 'week']).reset_index(drop=True)
    df['roll_pts'] = df.groupby(['player'])['y_act'].shift(1)
    df['roll_max'] = df.groupby('player')['roll_pts'].rolling(8, min_periods=1).apply(lambda x: pd.Series(x).nlargest(2).iloc[-1]).values
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

    # get the rolling stats data    
    stats, _ = rolling_max_std(pos)

    # join together and calculate sd and max metrics
    df = pd.merge(proj, stats, on=['player', 'team', 'week', 'year'])
    if pos == 'Defense': df = create_sd_max_metrics(df, defense=True)
    else: df = create_sd_max_metrics(df, defense=False)

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


# def fix_std_dev(df, bridge, spline, models):

#     import numpy as np
    
#     df['std_dev'] = bridge
    
#     std_error = np.mean(df.loc[:3, 'std_dev'] / df.loc[:3, 'pred_fp_per_game'])
#     serr_std_dev = df.pred_fp_per_game * std_error
    
#     df['std_dev'] = (serr_std_dev + spline) / 2
#     df['std_dev'] = df.std_dev + models
#     # df.loc[df.dk_salary > 4000, 'std_dev'] = df.loc[df.dk_salary > 4000, 'std_dev'] + models

#     return df


# %%

full_stats, recent = rolling_max_std('WR')
recent.dropna().sort_values(by='roll_max', ascending=False).iloc[:50]
# %%
get_std_splines('WR', show_plot=True, k=2, s=2000)
# %%
