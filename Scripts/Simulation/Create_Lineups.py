#%%
from zSim_Helper_Covar import *
import pprint

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral
from joblib import Parallel, delayed

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
conn = dm.db_connect('Simulation')

#===============
# Settings and User Inputs
#===============

year=2023
week=17

salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
num_lineups = 5
set_max_team = None

pred_vers = 'sera0_rsq0_mse1_brier1_matt1_bayes'

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

def pull_best_params(best_trial):
    params = {}
    params_opt = dm.read(f'''SELECT * 
                             FROM Entry_Optimize_Params
                             WHERE trial_num = {best_trial}''', 'Results')
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

def pull_params_version(best_trial):
    vers = dm.read(f'''SELECT DISTINCT trial_num, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type
                       FROM Entry_Optimize_Results
                       WHERE trial_num = {best_trial}''', 'Results')
    return vers

best_trials = 499

opt_params = pull_best_params(best_trials)
pprint.pprint(opt_params)

best_vers = pull_params_version(best_trials)
pred_vers = best_vers.pred_vers.values[0]
reg_ens_vers = best_vers.reg_ens_vers.values[0]
million_ens_vers = best_vers.million_ens_vers.values[0]
std_dev_type = best_vers.std_dev_type.values[0]
best_vers

#%%
        
d_ordering = ['adjust_pos_counts', 'player_drop_multiple', 'matchup_seed', 'matchup_drop', 
            'top_n_choices', 'full_model_weight', 'covar_type', 'max_team_type',
            'min_player_same_team', 'min_players_opp_team', 'num_top_players', 
            'ownership_vers', 'qb_min_iter', 'qb_set_max_team', 'qb_solo_start', 
            'static_top_players', 'use_ownership', 'own_neg_frac', 
            'max_salary_remain', 'num_iters', 'num_avg_pts']

d = {'adjust_pos_counts': {False: 0.7, True: 0.3},
 'covar_type': {'kmeans_pred_trunc': 0.0,
                'no_covar': 0.0,
                'team_points_trunc': 1.0},
 'full_model_weight': {0.2: 0.3, 5: 0.7},
 'lineups_per_param': {1: 1.0},
 'matchup_drop': {0: 0.7, 1: 0.1, 2: 0.2, 3: 0.0},
 'matchup_seed': {0: 0.3, 1: 0.7},
 'max_salary_remain': {200: 0.0, 500: 0.6, 1000: 0.4, 1500: 0.0},
 'max_team_type': {'player_points': 0.7, 'vegas_points': 0.3},
 'min_player_same_team': {2: 0.2, 3: 0.2, 'Auto': 0.6},
 'min_players_opp_team': {1: 0.1, 2: 0.1, 'Auto': 0.8},
 'num_avg_pts': {1: 0.0, 2: 0.0, 3: 0.0, 5: 0.0, 7: 0.3, 10: 0.7},
 'num_iters': {50: 0.0, 100: 0.0, 150: 1.0},
 'num_top_players': {2: 0.4, 3: 0.6, 5: 0.0},
 'own_neg_frac': {0.8: 0.0, 1: 1.0},
 'ownership_vers': {'mil_div_standard_ln': 0.0,
                    'mil_only': 0.0,
                    'mil_times_standard_ln': 0.3,
                    'standard_ln': 0.7},
 'player_drop_multiple': {0: 0.4, 2: 0.2, 4: 0.4},
 'qb_min_iter': {0: 0.6, 2: 0.4, 4: 0.0, 9: 0.0},
 'qb_set_max_team': {0: 0.7, 1: 0.3},
 'qb_solo_start': {False: 1.0, True: 0.0},
 'qb_stack_wt': {1: 0.0, 2: 0.0, 3: 0.7, 4: 0.3},
 'static_top_players': {False: 0.3, True: 0.7},
 'top_n_choices': {0: 1.0, 1: 0.0, 2: 0.0},
 'use_ownership': {0.7: 0.0, 0.8: 0.2, 0.9: 0.0, 1: 0.8},
 'use_unique_players': {0: 1.0}}

lineups_per_param = int(d['lineups_per_param'][1])

d = {k: d[k] for k in d_ordering}
params = []
for i in range(int(num_lineups/lineups_per_param)):
    cur_params = []
    for param, param_options in d.items():
        param_vars = list(param_options.keys())
        param_prob = list(param_options.values())
        cur_params.append(np.random.choice(param_vars, p=param_prob))

    cur_params.append(i)
    params.append(cur_params)

#%%

def sim_winnings(adjust_select, player_drop_multiplier, matchup_seed, matchup_drop, top_n_choices, 
                full_model_rel_weight, covar_type, max_team_type, min_players_same_team, 
                min_players_opp_team, num_top_players, ownership_vers, qb_min_iter, qb_set_max_team, qb_solo_start,
                static_top_players, use_ownership, own_neg_frac, max_salary_remain, 
                num_iters, num_avg_pts, param_iter
                ):
    
    try: min_players_opp_team = int(min_players_opp_team)
    except: pass

    try: min_players_same_team = float(min_players_same_team)
    except: pass

    if covar_type=='no_covar': use_covar=False
    else: use_covar=True

    total_add = []
    to_drop_selected = []
    lineups = []
    for _ in range(lineups_per_param):

        to_add = []
        sim = FootballSimulation(dm.db_connect('Simulation'), week, year, salary_cap, pos_require_start, num_iters, 
                                pred_vers, reg_ens_vers=reg_ens_vers, million_ens_vers=million_ens_vers,
                                std_dev_type=std_dev_type, covar_type=covar_type, 
                                full_model_rel_weight=full_model_rel_weight, matchup_seed=matchup_seed,
                                use_covar=use_covar, use_ownership=use_ownership, 
                                salary_remain_max=max_salary_remain)
        

        i = 0  # Initialize the iteration counter
        while len(to_add) < 9 and i < 18:  # Use a while loop to control iterations and break if necessary
            
            to_drop = []
            to_drop.extend(to_drop_selected)
            
            results, _ = sim.run_sim(dm.db_connect('Simulation'), to_add, to_drop, min_players_same_team, set_max_team, 
                                    min_players_opp_team_input=min_players_opp_team, max_team_type=max_team_type,
                                    adjust_select=adjust_select, num_matchup_drop=matchup_drop,
                                    own_neg_frac=own_neg_frac, ownership_vers=ownership_vers,
                                    n_top_players=num_top_players, num_avg_pts=num_avg_pts,
                                    static_top_players=static_top_players, qb_min_iter=qb_min_iter,
                                    qb_set_max_team=qb_set_max_team, qb_solo_start=qb_solo_start)
            
            results = results[~results.player.isin(to_add)].reset_index(drop=True)
            prob = results.loc[:top_n_choices, 'SelectionCounts'] / results.loc[:top_n_choices, 'SelectionCounts'].sum()
            
            try: 
                selected_player = np.random.choice(results.loc[:top_n_choices, 'player'], p=prob)
                to_add.append(selected_player)
            except: 
                pass
            i += 1

        lineups.append(to_add)
        total_add.extend(to_add)
        to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier)
            
        to_add.append(param_iter)

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

    ids = dm.read(f"SELECT * FROM Player_Ids WHERE year={year} AND week={week}", "Simulation")
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

# for adj, pdm, mseed, md, tn, fmw, ct, mtt, mpst, mpot, ntp, owvers, qmi, qsmt, qss, stp, uo, onf, msr, ni, nap, param_i in params:
#     sim_winnings(adj, pdm, mseed, md, tn, fmw, ct, mtt, mpst, mpot, ntp, owvers, qmi, qsmt, qss, stp, uo, onf, msr, ni, nap, param_i)

par_out = Parallel(n_jobs=-1, verbose=0)(delayed(sim_winnings)(adj, pdm, mseed, md, tn, fmw, ct, mtt, mpst, mpot, ntp, owvers, qmi, qsmt, qss, stp, uo, onf, msr, ni, nap, param_i) for \
                                                               adj, pdm, mseed, md, tn, fmw, ct, mtt, mpst, mpot, ntp, owvers, qmi, qsmt, qss, stp, uo, onf, msr, ni, nap, param_i in params)

lineups_list = []
for p in par_out:
    lineups_list.extend(p[0])

player_data = par_out[0][1]
lineups = clean_lineup_list(lineups_list, player_data)

lineups = lineups.sample(frac=1)

dm.delete_from_db('Simulation', 'Automated_Lineups', f'year={year} AND week={week}', create_backup=False)
for j, i in enumerate(lineups.TeamNum.unique()):
    if len(lineups[lineups.TeamNum==i]) == 9:
        create_database_output(lineups[lineups.TeamNum==i], j)
    else:
        print('Incomplete Lineup')

# %%
import datetime as dt

run_params_dict = {
    'week': [week],
    'year': [year],
    'last_update': [dt.datetime.now().strftime(f"%A %B %d, %I:%M%p")],
    'pred_vers': [pred_vers],
    'reg_ens_vers': [reg_ens_vers],
    'std_dev_type': [std_dev_type],
    'million_ens_vers': [million_ens_vers]
}

for param, param_options in d.items():
    param_vars = list(param_options.keys())
    param_prob = list(param_options.values())
    run_params_dict[param] = ['np.random.choice(' + str(param_vars) + ', p=' + str(param_prob) + ')']

run_params_df = pd.DataFrame(run_params_dict)

dm_app = DataManage('c:/Users/borys/OneDrive/Documents/Github/Daily_Fantasy_App/app/')
dm_app.write_to_db(run_params_df, 'Simulation_App', 'Run_Params', 'replace')

for t in ['Predicted_Ownership', 'Mean_Ownership', 'Gambling_Lines', 'Vegas_Points', 
          'Salaries', 'Player_Teams', 'Player_Ids', 'Automated_Lineups']:

    df = dm.read(f"SELECT * FROM {t} WHERE year={year} and week={week}", 'Simulation')
    dm_app.write_to_db(df, 'Simulation_App', t, 'replace')

for t in ['Model_Predictions', 'Covar_Means', 'Covar_Matrix']:
    df = dm.read(f'''SELECT * 
                    FROM {t}
                    WHERE week={week}
                        AND year={year}
                        AND pred_vers='{pred_vers}'
                        AND reg_ens_vers='{reg_ens_vers}'
                        AND std_dev_type='{std_dev_type}'
                        ''', 'Simulation')
    dm_app.write_to_db(df, 'Simulation_App', t, 'replace')

# %%
