#%%
from zSim_Helper_Covar import *
#%%
# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#===============
# Settings and User Inputs
#===============

for week in [6]:

    year = 2021
    salary_cap = 50000
    pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
    num_iters = 100

    pred_vers = 'standard_proba_update'
    ensemble_vers = 'no_weight_no_kbest_randsample'
    std_dev_type = 'spline'
    TOTAL_LINEUPS = 10

    print(f'\nWeek {week} PredVer: {pred_vers} EnsVer: {ensemble_vers} SDType:{std_dev_type}\n===============\n')

    min_players_same_team = 2
    set_max_team = None
    

    def get_stats(pos):
        if pos=='Defense': colname='defTeam'
        else: colname='player'
        return dm.read(f'''SELECT {colname} AS player, fantasy_pts
                            FROM {pos}_Stats
                            WHERE week={week}
                                AND season={year}''', 'FastR')

    def calc_winnings(to_add):
        results = pd.DataFrame(to_add, columns=['player'])
        results = pd.merge(results, points, on='player')
        total_pts = results.fantasy_pts.sum()
        idx_match = np.argmin(abs(prizes.Points - total_pts))
        prize_money = prizes.loc[idx_match, 'prize']

        return np.round(total_pts,1), prize_money

    def rand_drop_selected(total_add, drop_multiplier):
        to_drop = []
        total_selections = dict(Counter(total_add))
        for k, v in total_selections.items():
            prob_drop = (v * drop_multiplier) / TOTAL_LINEUPS
            drop_val = np.random.uniform() * prob_drop
            if  drop_val >= 0.5:
                to_drop.append(k)
        return to_drop

    def get_my_results(week):
        path = f'//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/MBorysiak/DK/Results/week{week}.csv'
        my_results = pd.read_csv(path, low_memory=False)
        my_results = my_results.loc[my_results.EntryName.str.contains('mborysi'), 'Points'].values

        actual_winnings = []
        for mr in my_results:
            idx_match = np.argmin(abs(prizes.Points - mr))
            prize_money = prizes.loc[idx_match, 'prize']
            actual_winnings.append(prize_money)

        return actual_winnings, my_results


    def summary_results(winnings, points):
        if len(winnings) != 10:
            frac = 10 / len(winnings)
        else:
            frac = 1
        total_winnings = np.sum(winnings) * frac
        mean_points = np.mean(points)
        max_points = np.max(points)
        max_winnings = np.max(winnings)
        num_placed = len([i for i in winnings if i>0]) * frac

        results = pd.DataFrame([[num_placed, total_winnings, max_winnings, mean_points, max_points]],
                                columns=['NumberPlaced', 'TotalWinnings', 'MaxWinnings', 'MeanPoints', 'MaxPoints'])
        results = round(results, 1)
        return results

    def rand_drop_teams(unique_teams, drop_frac):
        drop_teams = np.random.choice(unique_teams, size=int(drop_frac*len(unique_teams)))
        return list(player_teams.loc[player_teams.team.isin(drop_teams), 'player'].values)


    player_teams = dm.read(f'''SELECT DISTINCT player, team 
                               FROM Covar_Means
                               WHERE week={week}
                                     AND year={year}
                                     AND pred_vers='{pred_vers}'
                                     ''', 'Simulation')
    unique_teams = player_teams.team.unique()

    points = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
        points = pd.concat([points, get_stats(pos)])

    prizes = dm.read(f'''SELECT Rank, Points, prize
                        FROM Million_Results
                        WHERE week={week}
                            AND year={year}''', 'DK_Results')

    min_prize_pts = round(prizes.loc[prizes.prize > 0, 'Points'].min(),1)
    mean_prize_pts = round(prizes.loc[prizes.prize > 0, 'Points'].mean(),1)
    max_prize_pts = round(prizes.loc[prizes.prize > 0, 'Points'].max(),1)
    print(f'Min Prize Points: {min_prize_pts}\nMean Prize Points: {mean_prize_pts}\nMax Prize Points: {max_prize_pts}')

    actual_winnings, actual_points = get_my_results(week)
    actual_results = summary_results(actual_winnings, actual_points)
    actual_results.columns = ['My'+ar for ar in actual_results]
    print(actual_results)


    from itertools import product

    def dict_configs(d):
        for vcomb in product(*d.values()):
            yield dict(zip(d.keys(), vcomb))

    G = {
        'adjust_pos_counts': [True, False], 
        'drop_player_multiple': [0, 4], 
        'drop_team_frac': [0, 0.1],
        'top_n_choices': [0, 4],
        'full_model_rel_weight': [0.2, 1, 5],
        'covar_type': ['team_points', 'no_covar'],
        'iter': [0, 1, 2],
        }

    params = []
    for config in dict_configs(G):
        params.append(list(config.values()))


    def sim_winnings(adjust_select, player_drop_multiplier, team_drop_frac, top_n_choices, full_model_rel_weight, covar_type):

        if covar_type=='team_points': use_covar=True
        elif covar_type=='no_covar': use_covar=False

        sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                                 pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                                 covar_type=covar_type, full_model_rel_weight=full_model_rel_weight, 
                                 use_covar=use_covar)

        winnings = []
        points_record = []
        total_add = []
        to_drop_selected = []
        for _ in range(10):
            
            to_add = []
            to_drop = rand_drop_teams(unique_teams, team_drop_frac)
            to_drop.extend(to_drop_selected)

            for i in range(8):
                results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, adjust_select)
                prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
                selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
                to_add.append(selected_player)
            
            total_pts, prize_money = calc_winnings(to_add)
            winnings.append(prize_money); points_record.append(total_pts)

            total_add.extend(to_add)
            to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier)

        sim_results = summary_results(winnings, points_record)
        print(sim_results)

        return list(sim_results.values)

    # sim_winnings(True, 2, 0, 3, 1, 'team_points')

#%%
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=-1, verbose=10)(delayed(sim_winnings)(adj, pdm, tdf, tn, fmw, ct) for adj, pdm, tdf, tn, fmw, ct, i in params)
    results = [r[0] for r in results]

    output = pd.concat([pd.DataFrame(params), pd.DataFrame(results)], axis=1)
    output = pd.concat([output, actual_results], axis=1).fillna(method='ffill')

    cols = list(G.keys())
    cols.extend(['lineups_placed', 'total_winnings', 'max_winnings', 'avg_points', 'max_points', 
                'my_number_placed', 'my_total_winnings', 'my_max_winnings', 'my_mean_points', 'my_max_points'])
    output.columns = cols
    print(output)

    output = output.assign(min_prize_points=min_prize_pts, mean_prize_points=mean_prize_pts, max_prize_points=max_prize_pts,
                           week=week, year=year, pred_vers=pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                           min_player_same_team=min_players_same_team)

    dm.write_to_db(output, 'Simulation', 'Winnings_Optimize', 'append')

#%%
# from sklearn.linear_model import Ridge
# import sklearn
# from sklearn.ensemble import RandomForestRegressor
# from lightgbm import LGBMRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score
# from ff.db_operations import DataManage
# from ff import general as ffgeneral
# import pandas as pd
# import numpy as np

# root_path = ffgeneral.get_main_path('Daily_Fantasy')
# db_path = f'{root_path}/Data/Databases/'
# dm = DataManage(db_path)

# df = dm.read('''SELECT * 
#                 FROM Winnings_Optimize
#                 WHERE week > 2
#                 ORDER BY year, week''', 'Simulation')
# df.loc[:, 'std_dev_type'] = 'spline'
# df.loc[df.week < 6, 'std_dev_type'] = 'bridge'

# X = df[['adjust_pos_counts', 'drop_player_multiple',  'drop_team_frac', 'top_n_choices', 
#         'week', 'pred_vers', 'ensemble_vers', 'covar_type', 'full_model_rel_weight', 'std_dev_type']]

# def one_hot(X):
#     for c in ['week', 'pred_vers', 'ensemble_vers', 'covar_type', 'std_dev_type']:
#         X = pd.concat([X, pd.get_dummies(X[c], prefix=c, drop_first=True)], axis=1)
    
#         if c!='week':
#             X = X.drop(c, axis=1)
    
#     return X


# X = one_hot(X)
# y = df.total_winnings

# # m = Ridge(alpha=100)
# # m = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2)
# m = LGBMRegressor(n_estimators=25, max_depth=5, min_samples_leaf=1)

# if type(m) == sklearn.linear_model._ridge.Ridge:
#     sc = StandardScaler()
#     sc.fit(X)
#     X = pd.DataFrame(sc.transform(X), columns=X.columns)

# scores = cross_val_score(m, X, y, cv=5, scoring='neg_mean_squared_error')
# scores = np.sqrt(-np.mean(scores))
# print(scores)
# m.fit(X,y)

# try:
#     pd.Series(m.coef_, index=X.columns).sort_values().plot.barh(figsize=(10,10))

# except:
#     import shap
#     shap_values = shap.TreeExplainer(m).shap_values(X)
#     shap.summary_plot(shap_values, X, feature_names=X.columns, plot_size=(8,10), max_display=20, show=False)


# # %%

# X_pred = pd.DataFrame({
#  'adjust_pos_counts': [1], 
#  'drop_player_multiple': [0], 
#  'drop_team_frac': [0],
#  'top_n_choices': [0], 
#  'week': [19], 
#  'full_model_rel_weight': [0], 
# #  'week_3': [0], 
#  'week_4': [0],
#  'week_5': [0], 
#  'week_6': [0], 
#  'week_7': [0], 
#  'week_8': [0], 
#  'week_9': [0], 
#  'week_10': [0], 
#  'week_11': [0],
#  'week_12': [0], 
#  'week_13': [1], 
#  'week_14': [0], 
#  'week_15': [0], 
#  'week_16': [0], 
#  'week_17': [0],
#  'week_18': [0], 
#  'pred_vers_standard_proba': [0],
#  'pred_vers_standard_proba_quant': [0],
#  'pred_vers_standard_proba_sweight': [1],
#  'ensemble_vers_linear_weight': [0],
#  'ensemble_vers_no_weight': [0],
#  'covar_type_team_points': [0], 
#  'std_dev_type_spline': [1],
#  }, index=[0])

# if type(m) == sklearn.linear_model._ridge.Ridge:
#     X_pred = pd.DataFrame(sc.transform(X_pred), columns=X_pred.columns)

# print('Optimal Avg Winnings:', m.predict(X_pred)[0])

# my_avg_winnings = dm.read('''SELECT DISTINCT week, year, my_total_winnings 
#                              FROM Winnings_Optimize''', 'Simulation').my_total_winnings.mean()
# print('My Avg Winnings:', my_avg_winnings)

# # %%
