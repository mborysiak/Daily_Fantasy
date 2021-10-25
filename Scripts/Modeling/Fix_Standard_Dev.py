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
        dk_sal = dm.read('''SELECT team player, team, week, year, projected_points sd_metric
                            FROM FantasyPros
                            WHERE pos='DST' ''', 'Pre_PlayerData')

        stats = dm.read(f'''SELECT defTeam player, defTeam team, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats ''', 'FastR')
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
                            FROM {pos}_Stats ''', 'FastR')
        
        dk_sal['sd_metric'] = (dk_sal.projected_points + dk_sal.fantasyPoints + dk_sal.ProjPts)/3
        dk_sal['max_metric'] = (dk_sal.projected_points + dk_sal.fantasyPoints + dk_sal.ProjPts)/3
        dk_sal = dk_sal.drop(['projected_points', 'ProjPts', 'fantasyPoints'], axis=1)

    # join together and sort by dk_salary to prepare for group creation
    df = pd.merge(dk_sal, stats, on=['player', 'team', 'week', 'year'])
    if pos in ('QB', 'Defense'): k = 1
    df = df.sort_values(by='sd_metric').reset_index(drop=True)

    min_grps = int(df.shape[0] / 100)
    max_grps = int(df.shape[0] / 50)

    splines = {}
    for met in ['std_dev', 'perc_99']:
        r_squares = []
        max_r2 = 0
        for num_grps in range(min_grps, max_grps, 1):

            # create equal sizes groups going down the dataframe ordered by dk_salary
            df_len = len(df)
            repeats = math.ceil(df.shape[0] / num_grps)
            grps = np.repeat([i for i in range(num_grps)], repeats)
            df['grps'] = grps[:df_len]

            # calculate the standard deviations of each group
            std_devs = df.groupby('grps').agg({'y_act': [np.std, lambda x: np.percentile(x, 99)],
                                            'sd_metric': 'mean',
                                            'player': 'count'})
            std_devs.columns = ['std_dev', 'perc_99', 'sd_metric', 'player_cnts']

            # fit a spline to the group datasets
            X = std_devs[['sd_metric']]
            y = std_devs[[met]]
            spl = UnivariateSpline(X, y, k=k, s=s)

            r2 = r2_score(y, spl(X))
            r_squares.append([num_grps, r2])
            if r2 > max_r2:
                max_r2 = r2
                max_spl = spl
                X_max = X
                y_max = y

        splines[met] = max_spl

        if show_plot:
            print(met)
            X_pred = list(range(int(df.sd_metric.min()), int(df.sd_metric.max()), 1))
            plt.scatter(X, y)
            plt.scatter(X_max, y_max)
            plt.plot(X_pred, max_spl(X_pred), 'g', lw=3)
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

# %%
