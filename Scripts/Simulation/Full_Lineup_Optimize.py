#%%
from zSim_Helper_Covar import *

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

    pred_vers = 'standard_proba_sera_brier_lowsample'
    ensemble_vers = 'no_weight_yes_kbest_sera'
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
        lineups = []
        for _ in range(10):
            
            to_add = []
            to_drop = rand_drop_teams(unique_teams, team_drop_frac)
            to_drop.extend(to_drop_selected)

            for i in range(8):
                results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, adjust_select)
                prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
                selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
                to_add.append(selected_player)
            lineups.append(to_add)

            total_pts, prize_money = calc_winnings(to_add)
            winnings.append(prize_money); points_record.append(total_pts)

            total_add.extend(to_add)
            to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier)

        sim_results = summary_results(winnings, points_record)
        print(sim_results)

        return list(sim_results.values), lineups


#%%
    from joblib import Parallel, delayed

    par_out = Parallel(n_jobs=-1, verbose=10)(delayed(sim_winnings)(adj, pdm, tdf, tn, fmw, ct) for adj, pdm, tdf, tn, fmw, ct, i in params)
    
    lineups = []
    for o in par_out:
        lineups.extend(o[1])
    lineups = pd.DataFrame(lineups)
    lineups = lineups.assign(week=week, year=year, pred_vers=pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type)

    results = [list(o[0][0]) for o in par_out]
    output = pd.concat([pd.DataFrame(params), pd.DataFrame(results)], axis=1)
    output = pd.concat([output, actual_results], axis=1).fillna(method='ffill')

    cols = list(G.keys())
    cols.extend(['lineups_placed', 'total_winnings', 'max_winnings', 'avg_points', 'max_points', 
                'my_number_placed', 'my_total_winnings', 'my_max_winnings', 'my_mean_points', 'my_max_points'])
    
    output.columns = cols
    output = output.assign(min_prize_points=min_prize_pts, mean_prize_points=mean_prize_pts, max_prize_points=max_prize_pts,
                           week=week, year=year, pred_vers=pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                           min_player_same_team=min_players_same_team, num_iters=num_iters)

    
    output['pred_proba'] = 0
    output.loc[output.pred_vers.str.contains('proba'), 'pred_proba'] = 1

    output['pred_sera'] = 0
    output.loc[output.pred_vers.str.contains('sera'), 'pred_sera'] = 1

    output['pred_brier'] = 0
    output.loc[output.pred_vers.str.contains('brier'), 'pred_brier'] = 1

    output['pred_lowsample'] = 0
    output.loc[output.pred_vers.str.contains('lowsample'), 'pred_lowsample'] = 1

    output['ens_sample_weights'] = 0
    output.loc[output.ensemble_vers.str.contains('yes_weight'), 'ens_sample_weights'] = 1

    output['ens_kbest'] = 0
    output.loc[output.ensemble_vers.str.contains('yes_kbest'), 'ens_kbest'] = 1

    output['ens_randsample'] = 0
    output.loc[output.ensemble_vers.str.contains('randsample'), 'ens_randsample'] = 1

    output['ens_sera'] = 0
    output.loc[output.ensemble_vers.str.contains('sera'), 'ens_sera'] = 1

    output['std_spline'] = 0
    output.loc[output.std_dev_type.str.contains('spline'), 'std_spline'] = 1

    output['std_quantile'] = 0
    output.loc[output.std_dev_type.str.contains('quantile'), 'std_quantile'] = 1

    dm.write_to_db(output, 'Results', 'Winnings_Optimize', 'append')
    dm.write_to_db(lineups, 'Results', 'Lineups_Optimize', 'append')

#%%

from zSim_Helper_Covar import *
import pandas as pd

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


df = dm.read('''SELECT * FROM Winnings_Optimize''', 'Results')
df.ensemble_vers.unique()

df['pred_proba'] = 0
df.loc[df.pred_vers.str.contains('proba'), 'pred_proba'] = 1

df['pred_sera'] = 0
df.loc[df.pred_vers.str.contains('sera'), 'pred_sera'] = 1

df['pred_brier'] = 0
df.loc[df.pred_vers.str.contains('brier'), 'pred_brier'] = 1

df['pred_lowsample'] = 0
df.loc[df.pred_vers.str.contains('lowsample'), 'pred_lowsample'] = 1

df['ens_sample_weights'] = 0
df.loc[df.ensemble_vers.str.contains('yes_weight'), 'ens_sample_weights'] = 1

df['ens_kbest'] = 0
df.loc[df.ensemble_vers.str.contains('yes_kbest'), 'ens_kbest'] = 1

df['ens_randsample'] = 0
df.loc[df.ensemble_vers.str.contains('randsample'), 'ens_randsample'] = 1

df['ens_sera'] = 0
df.loc[df.ensemble_vers.str.contains('sera'), 'ens_sera'] = 1

df['std_spline'] = 0
df.loc[df.std_dev_type.str.contains('spline'), 'std_spline'] = 1

df['std_quantile'] = 0
df.loc[df.std_dev_type.str.contains('quantile'), 'std_quantile'] = 1

dm.write_to_db(df, 'Results', 'Winnings_Optimize', 'replace')

# %%
