#%%
from zSim_Helper_Covar import *
import pprint

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#===============
# Settings and User Inputs
#===============

year=2022
week=2


pred_vers = 'sera1_rsq0_brier1_matt0_bayes'
ensemble_vers = 'random_kbest_sera1_rsq0_mse0_include2_kfold3'
std_dev_type = 'spline_pred_class80_q80_matt0_brier1_kfold3'
ownership_vers = 'mil_only'

salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
num_lineups = 15
set_max_team = None


#%%

def rand_drop_selected(total_add, drop_multiplier):
    to_drop = []
    total_selections = dict(Counter(total_add))
    for k, v in total_selections.items():
        prob_drop = (v * drop_multiplier) / lineups_per_param
        drop_val = np.random.uniform() * prob_drop
        if  drop_val >= 0.5:
            to_drop.append(k)
    return to_drop


def get_matchups():
    df = dm.read(f'''SELECT away_team, home_team
                    FROM Gambling_Lines 
                    WHERE week={week} 
                        and year={year} 
                ''', 'Simulation')

    matchups = []
    for away, home in df.values:
        matchups.append([home, away])

    return matchups


def rand_drop_teams(matchups, drop_matchups):
    drop_teams = matchups[np.random.choice(matchups.shape[0], drop_matchups, replace=False), :][0]
    return list(player_teams.loc[player_teams.team.isin(drop_teams), 'player'].values)


player_teams = dm.read(f'''SELECT DISTINCT player, team 
                            FROM Covar_Means
                            WHERE week={week}
                                    AND year={year}
                                    AND pred_vers='{pred_vers}'
                                    ''', 'Simulation')
unique_teams = player_teams.team.unique()
matchups = np.array([m for m in get_matchups() if m[0] in unique_teams and m[1] in unique_teams])

#%%

def pull_best_params(best_trials):
    params = {}
    params_opt = dm.read(f'''SELECT * 
                             FROM Entry_Optimize_Params
                             WHERE trial_num IN {best_trials}''', 'Results')
    params_opt = params_opt.groupby(['param', 'param_option']).agg({'option_value': 'mean'}).reset_index()

    for p, opt, val in params_opt.values:
        if p not in params.keys(): params[p] = {}
        if p in ('adjust_pos_counts', 'static_top_players', 'qb_set_max_max', 'qb_solo_start'):
            if opt=='0': opt=False
            if opt=='1': opt=True
        else:
            try: opt = int(opt)
            except: 
                try: opt = float(opt)
                except: pass
    
        params[p][opt] = np.round(val,2)

    for k, v in params.items():
        if np.sum(list(v.values())) != 1:
            print('Not summed to 1:', k)
    print('\n')

    return params

best_trials = (161, 161)

opt_params = pull_best_params(best_trials)
pprint.pprint(opt_params)

#%%
        
d = {'adjust_pos_counts': {False: 0.04, True: 0.96},
 'covar_type': {'no_covar': 0.74, 'team_points_trunc': 0.26},
 'full_model_weight': {0.2: 0.3, 5: 0.7},
 'matchup_drop': {0: 0.6, 1: 0.3, 2: 0.1},
 'max_salary_remain': {200: 0.05, 500: 0.3, 1000: 0.25, 1500: 0.4},
 'max_team_type': {'player_points': 1.0},
 'min_player_same_team': {2: 0.44, 3: 0.26, 'Auto': 0.3},
 'min_players_opp_team': {1: 0.4, 2: 0.18, 'Auto': 0.42},
 'num_iters': {50: 0.26, 100: 0.74},
 'num_top_players': {2: 0.41, 3: 0.46, 5: 0.13},
 'own_neg_frac': {0.8: 0.99, 1: 0.01},
 'player_drop_multiple': {0: 0.38, 2: 0.32, 4: 0.3},
 'qb_min_iter': {0: 0.96, 9: 0.04},
 'qb_set_max_team': {0: 0.2, 1: 0.8},
 'qb_solo_start': {False: 0.39, True: 0.61},
 'static_top_players': {False: 0.23, True: 0.77},
 'top_n_choices': {0: 0.65, 1: 0.24, 2: 0.11},
 'use_ownership': {0.8: 0.14, 0.9: 0.25, 1: 0.61}}

lineups_per_param = 3

params = []
for i in range(int(num_lineups/lineups_per_param)):
    cur_params = []
    for param, param_options in d.items():
        param_vars = list(param_options.keys())
        param_prob = list(param_options.values())
        cur_params.append(np.random.choice(param_vars, p=param_prob))

    params.append(cur_params)

#%%
def sim_winnings(adjust_select,covar_type, full_model_rel_weight,matchup_drop,salary_remain_max,max_team_type,
                 min_players_same_team, min_players_opp_team, num_iters, num_top_players,
                 own_neg_frac, player_drop_multiplier, 
                 qb_min_iter, qb_set_max_team, qb_solo_start,
                 static_top_players, top_n_choices, use_ownership
                 ):

    try: min_players_opp_team = int(min_players_opp_team)
    except: pass

    try: min_players_same_team = int(min_players_same_team)
    except: pass
    
    if covar_type=='no_covar': use_covar=False
    else: use_covar=True

    sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                             pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                             covar_type=covar_type, full_model_rel_weight=full_model_rel_weight, 
                             use_covar=use_covar, use_ownership=use_ownership, salary_remain_max=salary_remain_max)

    total_add = []
    to_drop_selected = []
    lineups = []
    for _ in range(lineups_per_param):
        
        to_add = []
        to_drop = []
        to_drop.extend(to_drop_selected)

        for i in range(9):
            results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, None, min_players_opp_team, adjust_select,
                                      num_matchup_drop=matchup_drop, own_neg_frac=own_neg_frac, max_team_type=max_team_type,
                                      n_top_players=num_top_players, static_top_players=static_top_players,
                                      qb_min_iter=qb_min_iter, qb_set_max_team=qb_set_max_team, qb_solo_start=qb_solo_start)
            
            prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
            selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
            to_add.append(selected_player)
        lineups.append(to_add)

        total_add.extend(to_add)
        to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier)

    return lineups, sim.player_data

def clean_lineup_list(lineups_list, player_data):
    lineups = pd.DataFrame(lineups_list).T
    lineups = pd.melt(lineups)
    lineups.columns = ['TeamNum', 'player']
    lineups = pd.merge(lineups, player_data[['player', 'pos']], on=['player'])
    lineups = lineups.rename(columns={'pos':'Position', 'player': 'Player'})
    lineups = lineups.sort_values(by='TeamNum').reset_index(drop=True)
    return lineups

def create_database_output(my_team, j):

    ids = dm.read(f"SELECT * FROM Player_Ids WHERE year={year} AND league={week}", "Simulation")
    my_team_ids = my_team.rename(columns={'Player': 'player'}).copy()
    dk_output = pd.merge(my_team_ids, ids, on='player')

    for pstn, num_req in zip(['WR', 'RB', 'TE'], [4, 3, 2]):
        if len(dk_output[dk_output.Position == pstn]) == num_req:
            idx_last = dk_output[dk_output.Position == pstn].index[-1]
            dk_output.loc[dk_output.index==idx_last, 'Position'] = 'FLEX'

    pos_map = {
        'QB': 'aQB', 
        'RB': 'bRB',
        'WR': 'cWR',
        'TE': 'dTE',
        'FLEX': 'eFLEX',
        'DST': 'fDST'
    }
    dk_output.Position = dk_output.Position.map(pos_map)
    dk_output = dk_output.sort_values(by='Position').reset_index(drop=True)
    pos_map_rev = {v: k for k,v in pos_map.items()}
    dk_output.Position = dk_output.Position.map(pos_map_rev)

    dk_output_ids = dk_output[['Position', 'player_id']].T.reset_index(drop=True)
    dk_output_players = dk_output[['Position', 'player']].T.reset_index(drop=True)
    dk_output = pd.concat([dk_output_players, dk_output_ids], axis=1)

    dk_output.columns = range(dk_output.shape[1])
    dk_output = pd.DataFrame(dk_output.iloc[1,:]).T

    dk_output['year'] = year
    dk_output['week'] = week

    if j < 10:
        dk_output['contest'] = 'Million'
    else:
        dk_output['contest'] = 'Screen Pass'


    dm.write_to_db(dk_output, 'Simulation', 'Automated_Lineups', 'append')

#%%

# dm.delete_from_db('Simulation', 'Automated_Lineups', f'year={year} AND week={week}', create_backup=False)

from joblib import Parallel, delayed

par_out = Parallel(n_jobs=-1, verbose=0)(delayed(sim_winnings)(adj, ct, fmw, md, msr, mtt, mpst, mpot, ni, ntp, onf, pdm, qmi, qstm, qss, stp, tn, uo) for \
                                                               adj, ct, fmw, md, msr, mtt, mpst, mpot, ni, ntp, onf, pdm, qmi, qstm, qss, stp, tn, uo in params)

lineups_list = []
for p in par_out:
    lineups_list.extend(p[0])

player_data = par_out[0][1]
lineups = clean_lineup_list(lineups_list, player_data)

lineups = lineups.sample(frac=1)

# for j, i in enumerate(lineups.TeamNum.unique()):
#     create_database_output(lineups[lineups.TeamNum==i], j)

# %%

run_params_dict = {
    'week': [week],
    'year': [year],
    'pred_vers': [pred_vers],
    'ensemble_vers': [ensemble_vers],
    'std_dev_type': [std_dev_type],
    'ownership_vers': [ownership_vers]
}
for param, param_options in d.items():
    param_vars = list(param_options.keys())
    param_prob = list(param_options.values())
    run_params_dict[param] = ['np.random.choice(' + str(param_vars) + ', p=' + str(param_prob) + ')']

run_params_df = pd.DataFrame(run_params_dict)

dm_app = DataManage('c:/Users/mborysia/Documents/Github/Daily_Fantasy_App/app/')
dm_app.write_to_db(run_params_df, 'Simulation_App', 'Run_Params', 'replace')

for t in ['Predicted_Ownership', 'Mean_Ownership', 'Gambling_Lines', 'Vegas_Points', 'Salaries', 'Player_Teams', 'Player_Ids']:

    df = dm.read(f"SELECT * FROM {t} WHERE year={year} and week={week}", 'Simulation')
    dm_app.write_to_db(df, 'Simulation_App', t, 'replace')

for t in ['Model_Predictions', 'Covar_Means', 'Covar_Matrix']:
    if t =='Model_Predictions': pred_var = 'version'
    else: pred_var = 'pred_vers'
    df = dm.read(f'''SELECT * 
                    FROM {t}
                    WHERE week={week}
                        AND year={year}
                        AND {pred_var}='{pred_vers}'
                        AND ensemble_vers='{ensemble_vers}'
                        AND std_dev_type='{std_dev_type}'
                        ''', 'Simulation')
    dm_app.write_to_db(df, 'Simulation_App', t, 'replace')


# %%
run_params_df

# %%
