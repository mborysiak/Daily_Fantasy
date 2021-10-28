#%%


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
        dk_sal = dm.read('''SELECT team player, team, week, year, projected_points
                            FROM FantasyPros
                            WHERE pos='DST' ''', 'Pre_PlayerData')

        stats = dm.read(f'''SELECT defTeam player, defTeam team, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020 ''', 'FastR')
    
    else:
        # pull in the salary and actual results data
        dk_sal = dm.read('''SELECT player, offTeam team, week, year, projected_points, fantasyPoints, ProjPts
                            FROM PFF_Proj_Ranks
                            JOIN (SELECT player, team offTeam, week, year, projected_points 
                                  FROM FantasyPros)
                                  USING (player, offTeam, week, year)
                            JOIN (SELECT player, offTeam, week, year, `Proj Pts` ProjPts 
                                  FROM PFF_Expert_Ranks)
                                  USING (player, offTeam, week, year)''', 'Pre_PlayerData')

        stats = dm.read(f'''SELECT player, team, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020 ''', 'FastR')
        
    # join together and sort by dk_salary to prepare for group creation
    df = pd.merge(dk_sal, stats, on=['player', 'team', 'week', 'year'])
    df = df.sort_values(by=['player', 'year', 'year']).reset_index(drop=True)
    df['fantasy_pts_score'] = df.groupby(['player'])['y_act'].shift(1)
    df['rmax8_fantasy_pts_score'] = df.groupby(['player'])[['fantasy_pts_score']].rolling(8, min_periods=1).max().values
    df['rstd8_fantasy_pts_score'] = df.groupby(['player'])[['fantasy_pts_score']].rolling(8, min_periods=1).std().values

    if pos == 'Defense':
        df['sd_metric'] = df.projected_points# (df.projected_points + df.rstd8_fantasy_pts_score) / 2
        df['max_metric'] = df.projected_points#(df.projected_points + df.rmax8_fantasy_pts_score) / 2
        df = df.drop(['projected_points', 'rmax8_fantasy_pts_score', 'rstd8_fantasy_pts_score'], axis=1)

    else:
        df['sd_metric'] = (df.projected_points + df.fantasyPoints + \
                               df.rstd8_fantasy_pts_score + df.ProjPts ) / 4
        df['max_metric'] = (df.projected_points + df.fantasyPoints + \
                                df.rmax8_fantasy_pts_score + df.ProjPts) / 4

        df = df.drop(['projected_points', 'ProjPts', 'fantasyPoints', 
                      'rmax8_fantasy_pts_score', 'rstd8_fantasy_pts_score' ], axis=1)

    # join together and sort by dk_salary to prepare for group creation
    df = df.dropna().sort_values(by='sd_metric').reset_index(drop=True)

    min_grps = int(df.shape[0] / 100)
    max_grps = int(df.shape[0] / 40)

    splines = {}
    X_max = {}
    y_max = {}
    max_r2 = {}
    for x_val, met in zip(['sd_metric', 'max_metric'], ['std_dev', 'perc_99']):

        max_r2[met] = 0
        for num_grps in range(min_grps, max_grps, 1):

            # create equal sizes groups going down the dataframe ordered by dk_salary
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


def fix_std_dev(df, bridge, spline, models):

    import numpy as np
    
    df['std_dev'] = bridge
    
    std_error = np.mean(df.loc[:3, 'std_dev'] / df.loc[:3, 'pred_fp_per_game'])
    serr_std_dev = df.pred_fp_per_game * std_error
    
    df['std_dev'] = (serr_std_dev + spline) / 2
    df['std_dev'] = df.std_dev + models
    # df.loc[df.dk_salary > 4000, 'std_dev'] = df.loc[df.dk_salary > 4000, 'std_dev'] + models

    return df
