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

year=2022
week=13

pred_vers = 'sera1_rsq0_brier1_matt1_lowsample_perc'
ensemble_vers = 'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3'
std_dev_type = 'pred_spline_class80_q80_matt1_brier1_kfold3'

salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
num_lineups = 30
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

best_trials = (140, 140)

opt_params = pull_best_params(best_trials)
opt_params

#%%
        
d = {'adjust_pos_counts': {False: 0.3, True: 0.7},
 'covar_type': {'kmeans_trunc': 0.0,
  'no_covar': 0.7,
  'team_points_trunc': 0.3},
 'full_model_weight': {0.2: 0.3, 0.5: 0.0, 1: 0.0, 3: 0.0, 5: 0.7},
 'matchup_drop': {0: 0.8, 1: 0.2, 2: 0.0, 3: 0.0},
 'max_salary_remain': {1000: 0.3, 200: 0.3, 300: 0.0, 400: 0.0, 500: 0.4},
 'min_player_same_team': {-1: 0.0, 2: 0.0, 2.5: 0.0, 3: 0.4, 'Auto': 0.6},
 'min_players_opp_team': {0: 0.0, 1: 0.3, 2: 0.0, 'Auto': 0.7},
 'num_iters': {100: 1.0},
 'num_top_players': {2: 0.5, 3: 0.0, 4: 0.0, 5: 0.5},
 'own_neg_frac': {0.5: 0.0, 0.65: 0.0, 0.75: 0.0, 0.85: 0.0, 1: 1.0},
 'player_drop_multiple': {0: 0.4, 1: 0.3, 2: 0.0, 4: 0.3, 6: 0.0},
 'qb_min_iter': {0: 1.0, 1: 0.0, 9: 0.0},
 'qb_set_max_team': {0: 0.0, 1: 1.0},
 'qb_solo_start': {False: 0.7, True: 0.3},
 'static_top_players': {False: 0.3, True: 0.7},
 'top_n_choices': {0: 0.8, 1: 0.2, 2: 0.0, 4: 0.0},
 'use_ownership': {0: 0.0, 0.5: 0.0, 0.9: 1.0, 1: 0.0}}

lineups_per_param = 2

params = []
for i in range(int(num_lineups/lineups_per_param)):
    cur_params = []
    for param, param_options in d.items():
        param_vars = list(param_options.keys())
        param_prob = list(param_options.values())
        cur_params.append(np.random.choice(param_vars, p=param_prob))

    params.append(cur_params)

#%%
def sim_winnings(adjust_select,covar_type, full_model_rel_weight,matchup_drop,salary_remain_max,
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
                                      num_matchup_drop=matchup_drop, own_neg_frac=own_neg_frac,
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

dm.delete_from_db('Simulation', 'Automated_Lineups', f'year={year} AND week={week}', create_backup=False)

from joblib import Parallel, delayed

par_out = Parallel(n_jobs=-1, verbose=0)(delayed(sim_winnings)(adj, ct, fmw, md, msr, mpst, mpot, ni, ntp, onf, pdm, qmi, qstm, qss, stp, tn, uo) for \
                                                               adj, ct, fmw, md, msr, mpst, mpot, ni, ntp, onf, pdm, qmi, qstm, qss, stp, tn, uo in params)

lineups_list = []
for p in par_out:
    lineups_list.extend(p[0])

player_data = par_out[0][1]
lineups = clean_lineup_list(lineups_list, player_data)

lineups = lineups.sample(frac=1)

for j, i in enumerate(lineups.TeamNum.unique()):
    create_database_output(lineups[lineups.TeamNum==i], j)

# %%

run_params_dict = {
    'week': [week],
    'year': [year],
    'pred_vers': [pred_vers],
    'ensemble_vers': [ensemble_vers],
    'std_dev_type': [std_dev_type],
}
for param, param_options in d.items():
    param_vars = list(param_options.keys())
    param_prob = list(param_options.values())
    run_params_dict[param] = ['np.random.choice(' + str(param_vars) + ', p=' + str(param_prob) + ')']

run_params_df = pd.DataFrame(run_params_dict)

dm.delete_from_db('Simulation', 'Run_Params', f"week={week} AND year={year}", create_backup=False)
dm.write_to_db(run_params_df, 'Simulation', 'Run_Params', 'replace')
# %%
