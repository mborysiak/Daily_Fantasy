
def get_std_splines(pos, pos_grps, show_plot=False, k=2, s=100):
    
    from ff.db_operations import DataManage   
    import ff.general as ffgeneral 
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import UnivariateSpline
    import matplotlib.pyplot as plt
    import pandas as pd

    # set the root path and database management object
    root_path = ffgeneral.get_main_path('Daily_Fantasy')
    db_path = f'{root_path}/Data/Databases/'
    dm = DataManage(db_path)
    
    num_grps = pos_grps[pos]

    if pos == 'Defense':

         # pull in the salary and actual results data
        dk_sal = dm.read('''SELECT team player, team, week, year, dk_salary
                            FROM Daily_Salaries''', 'Pre_TeamData')

        stats = dm.read(f'''SELECT defTeam player, defTeam team, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats ''', 'FastR')
    else:
        # pull in the salary and actual results data
        dk_sal = dm.read('''SELECT player, team, week, year, dk_salary
                            FROM Daily_Salaries''', 'Pre_PlayerData')

        stats = dm.read(f'''SELECT player, team, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats ''', 'FastR')


    # join together and sort by dk_salary to prepare for group creation
    df = pd.merge(dk_sal, stats, on=['player', 'team', 'week', 'year'])
    df = df.sort_values(by='dk_salary').reset_index(drop=True)

    # create equal sizes groups going down the dataframe ordered by dk_salary
    df_len = len(df)
    repeats = math.ceil(df.shape[0] / num_grps)
    grps = np.repeat([i for i in range(num_grps)], repeats)
    df['grps'] = grps[:df_len]

    # calculate the standard deviations of each group
    std_devs = df.groupby('grps').agg({'y_act': np.std,
                                    'dk_salary': 'mean',
                                    'player': 'count'})
    std_devs.columns = ['std_dev', 'dk_salary', 'player_cnts']

    # fit a spline to the group datasets
    X = std_devs[['dk_salary']]
    y = std_devs[['std_dev']]
    spl = UnivariateSpline(X, y, k=k, s=s)

    # plot the results of the fit to check
    if show_plot:
        X_pred = list(range(df.dk_salary.min(), df.dk_salary.max(), 500))
        plt.scatter(X, y)
        plt.plot(X_pred, spl(X_pred), 'g', lw=3)
        plt.show()

    return spl


def fix_std_dev(df, bridge, spline, models):

    import numpy as np
    
    df['std_dev'] = bridge
    
    std_error = np.mean(df.loc[:10, 'std_dev'] / df.loc[:10, 'pred_fp_per_game'])
    serr_std_dev = df.pred_fp_per_game * std_error
    
    df['std_dev'] = (serr_std_dev + spline) / 2
    # df.loc[df.dk_salary > 4000, 'std_dev'] = df.loc[df.dk_salary > 4000, 'std_dev'] + models

    return df