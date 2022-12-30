#%%
from zSim_Helper_Covar import *

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral
from joblib import Parallel, delayed
import pickle
import gzip
import os
from hyperopt import fmin, tpe, hp, space_eval, Trials
import pprint

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#===============
# Settings and User Inputs
#===============
# set the model version
set_weeks = [
   13, 14, 15, 16, 17,
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
]

set_years = [
      2021, 2021, 2021, 2021, 2021,
      2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022
]

save_path = "c:/Users/mborysia/Documents/Github/Daily_Fantasy/Model_Outputs/2022/Bayes_Sim_Opt/"
    
salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
adjust_winnings = False
set_max_team = None

#================
# Loading Data
#================

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def get_stats(pos, week, year):
    if pos=='Defense': colname='defTeam'
    else: colname='player'
    return dm.read(f'''SELECT {colname} AS player, fantasy_pts
                        FROM {pos}_Stats
                        WHERE week={week}
                            AND season={year}''', 'FastR')

#================
# Sim Helper functions
#================

def calc_winnings(to_add, points, prizes):
    results = pd.DataFrame(to_add, columns=['player'])
    results = pd.merge(results, points, on='player')
    total_pts = results.fantasy_pts.sum()
    idx_match = np.argmin(abs(prizes.Points - total_pts))
    prize_money = prizes.loc[idx_match, 'prize']

    return prize_money

def rand_drop_selected(total_add, drop_multiplier, lineups_per_param):
    to_drop = []
    total_selections = dict(Counter(total_add))
    for k, v in total_selections.items():
        prob_drop = (v * drop_multiplier) / lineups_per_param
        drop_val = np.random.uniform() * prob_drop
        if  drop_val >= 0.5:
            to_drop.append(k)
    return to_drop


def avg_winnings_contest(results):

    avg_winnings = []
    for _ in range(50):
        
        wt_winnings = []
        for w in results:
            wt_winnings.append(w * np.random.choice([1, 0.15], p=[0.34, 0.66]))
        avg_winnings.append(wt_winnings)

    avg_winnings = np.mean(np.array(avg_winnings), axis=0)

    return avg_winnings


def get_prizes(week, year):
    prizes = dm.read(f'''SELECT Rank, Points, prize
                        FROM Contest_Results
                        WHERE week={week}
                            AND year={year}
                            AND Contest='Million' ''', 'DK_Results')
    return prizes


def pull_points(week, year):

    points = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
        points = pd.concat([points, get_stats(pos, week, year)])

    return points

#================
# Saving output
#================

def results_to_df(save_path, fname):

    trials = load_pickle(save_path, fname)
    results_tmp = pd.DataFrame(trials.vals)
    eval_options = {k: v for k,v in init_space.items() if k in results_tmp.columns}

    results = pd.DataFrame()
    for idx, row in results_tmp.iterrows():
        if idx < 10:
            results = pd.concat([results, pd.DataFrame(space_eval(eval_options, row), index=[idx])], axis=0)
        else:
            results = pd.concat([results, pd.DataFrame(space_eval(eval_options, row), index=[idx])], axis=0)

    results['loss'] = [l['loss'] for l in trials.results]
    results = results.sort_values(by='loss').reset_index(drop=True)

    # append trial name to front of results df
    trial_name_df = pd.DataFrame({'trial_name': len(results)*[fname]})
    results = pd.concat([trial_name_df, results], axis=1)
    return results


def add_extras(results, trial_name, rpts):
    extras = pd.DataFrame({
                'trial_name': rpts*[trial_name],
                'pred_vers': rpts*['sera1_rsq0_brier1_matt1_lowsample_perc'],
                'ensemble_vers': rpts*['no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3'],
                'std_dev_type': rpts*['pred_spline_class80_q80_matt1_brier1_kfold3'],
                'ownership_vers': rpts*['standard_ln']
            })

    return pd.concat([extras, results.drop('trial_name', axis=1)], axis=1)


def show_trial_best_params(trial_name, best_result=0):
    results = dm.read(f"SELECT * FROM Entry_Optimize_Bayes WHERE trial_name='{trial_name}'", 'Results')
    results = results.drop(['trial_name', 'loss'], axis=1)
    best_params = results.iloc[best_result].to_dict()
    best_params_output = convert_param_options(best_params)

    vers_names = ('pred_vers', 'ensemble_vers', 'std_dev_type', 'ownership_vers', 'lineups_per_param')
    vers_params = {k: v for k,v in best_params.items() if k in vers_names}
    # best_params_output = {**vers_params, **best_params_output}

    pprint.pprint(vers_params, sort_dicts=False)
    print('\n')
    pprint.pprint(best_params_output, sort_dicts=False)

    
#================
# Run Sim code
#================

def sim_winnings(adjust_select, player_drop_multiplier, matchup_drop, top_n_choices, 
                full_model_rel_weight, covar_type, min_players_same_team, 
                min_players_opp_team, num_top_players, qb_min_iter, qb_set_max_team, qb_solo_start,
                static_top_players, use_ownership, own_neg_frac, max_salary_remain, 
                num_iters, lineups_per_param, 
                week, year, pred_vers, ensemble_vers, std_dev_type, ownership_vers
                ):
    
    prizes = get_prizes(week, year)
    points = pull_points(week, year)

    try: min_players_opp_team = int(min_players_opp_team)
    except: pass

    try: min_players_same_team = float(min_players_same_team)
    except: pass

    if covar_type=='no_covar': use_covar=False
    else: use_covar=True

    sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                                pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                                covar_type=covar_type, ownership_vers=ownership_vers,
                                full_model_rel_weight=full_model_rel_weight, 
                                use_covar=use_covar, use_ownership=use_ownership, 
                                salary_remain_max=max_salary_remain)

    winnings = []        
    total_add = []
    to_drop_selected = []
    for t in range(lineups_per_param):

        to_add = []
        to_drop = []
        to_drop.extend(to_drop_selected)

        for i in range(9):
            results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, 
                                    min_players_opp_team_input=min_players_opp_team, 
                                    adjust_select=adjust_select, num_matchup_drop=matchup_drop,
                                    own_neg_frac=own_neg_frac, n_top_players=num_top_players,
                                    static_top_players=static_top_players, qb_min_iter=qb_min_iter,
                                    qb_set_max_team=qb_set_max_team, qb_solo_start=qb_solo_start)
            
            prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
            try: 
                selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
                to_add.append(selected_player)
            except: 
                pass
            
        prize_money = calc_winnings(to_add, points, prizes)
        winnings.append(prize_money)

        total_add.extend(to_add)
        to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier, lineups_per_param)

    return winnings

#================
# Bayes opt functions
#================

def adjust_high_winnings(tw, max_adjust=10000):
    tw = np.array(tw)
    tw[tw>max_adjust] = max_adjust
    return tw

def show_params(bayes_params):
    print_params = {}
    for k, v in bayes_params.items():
        try: print_params[k] = np.round(v,3)
        except: print_params[k] = v
    print(print_params)


def convert_param_options(bp):
    d = {
        
        'adjust_pos_counts': {
            True: bp['adjust_pos_counts_True'],
            False: 1 - bp['adjust_pos_counts_True']
        },

        'player_drop_multiple': {
            4: bp['player_drop_multiple_4'],
            2: bp['player_drop_multiple_2'],
            0: 1 - bp['player_drop_multiple_4'] - bp['player_drop_multiple_2']
        },

        'matchup_drop': {
            1: bp['matchup_drop_1'],
            2: bp['matchup_drop_2'],
            0: 1 - bp['matchup_drop_1'] - bp['matchup_drop_2']
        }, 

        'top_n_choices': {
            1: bp['top_n_choices_1'],
            2: bp['top_n_choices_2'],
            0: 1 - bp['top_n_choices_1'] - bp['top_n_choices_2']
        },

        'full_model_weight': {
            5: bp['full_model_weight_5'],
            0.2: 1 -  bp['full_model_weight_5']
        }, 

        'covar_type': {
            'no_covar': bp['covar_type_no_covar'],
            'team_points_trunc': 1 -  bp['covar_type_no_covar']
        }, 

        'min_player_same_team': {
            2: bp['min_player_same_team_2'],
            3: bp['min_player_same_team_3'],
            'Auto': 1 - bp['min_player_same_team_2'] - bp['min_player_same_team_3']
        }, 

        'min_player_opp_team': {
            1: bp['min_player_opp_team_1'],
            2: bp['min_player_opp_team_2'],
            'Auto': 1 - bp['min_player_opp_team_1'] - bp['min_player_opp_team_2']
        }, 

        'num_top_players': {
            2: bp['num_top_players_2'],
            3: bp['num_top_players_3'],
            5: 1 - bp['num_top_players_2'] - bp['num_top_players_3']
        },

        'qb_min_iter': {
            0: bp['qb_min_iter_0'],
            9: 1 - bp['qb_min_iter_0']
        },

        'qb_set_max_team': {
            True: bp['qb_set_max_team_True'],
            False: 1 - bp['qb_set_max_team_True']
        },

        'qb_solo_start': {
            True: bp['qb_solo_start_True'],
            False: 1 - bp['qb_solo_start_True']
        },

        'static_top_players': {
            True: bp['static_top_players_True'],
            False: 1 - bp['static_top_players_True']
        },

        'use_ownership': {
            0.9: bp['use_ownership_0.9'],
            0.8: bp['use_ownership_0.8'],
            1: 1 - bp['use_ownership_0.9'] - bp['use_ownership_0.8']
        },

        'own_neg_frac': {
            0.8: bp['own_neg_frac_0.8'],
            1: 1 - bp['own_neg_frac_0.8'],
        },

        'max_salary_remain': {
            200: bp['max_salary_remain_200'],
            500: bp['max_salary_remain_500'],
            1000: bp['max_salary_remain_1000'],
            1500: 1 - bp['max_salary_remain_200'] - bp['max_salary_remain_500'] - bp['max_salary_remain_1000'],
        },

        'num_iters': {
            100: bp['num_iters_100'],
            50: 1 - bp['num_iters_100'],
        },
    }
    return d


def create_params_list(d, lineups_per_param, week, year, pred_vers, ensemble_vers, std_dev_type, ownership_vers):
    params = []
    for i in range(int(30/lineups_per_param)):
        cur_params = []
        for param, param_options in d.items():
            param_vars = list(param_options.keys())
            param_prob = list(param_options.values())
            cur_params.append(np.random.choice(param_vars, p=param_prob))

        cur_params.extend([lineups_per_param, week, year, pred_vers, ensemble_vers, std_dev_type, ownership_vers])
        params.append(cur_params)
    return params


def objective(bayes_params):

    show_params(bayes_params)
    print('\n')

    lineups_per_param = bayes_params['lineups_per_param']
    pred_vers = bayes_params['pred_vers']
    ensemble_vers = bayes_params['ensemble_vers']
    std_dev_type = bayes_params['std_dev_type']
    ownership_vers = bayes_params['ownership_vers']

    d = convert_param_options(bayes_params)
    
    total_winnings = []
    iter_cats = zip(set_weeks, set_years)
    for week, year in iter_cats:
        params = create_params_list(d, lineups_per_param, week, year, pred_vers, ensemble_vers, std_dev_type, ownership_vers)

        # winnings = []
        # for adj, pdm, md, tn, fmw, ct, mpst, mpot, ntp, qmi, qsmt, qss, stp, uo, onf, msr, ni, lpp, week, year, pred_vers, ensemble_vers, std_dev_type, own_vers in params:
        #     cur_winnings = sim_winnings(adj, pdm, md, tn, fmw, ct, mpst, mpot, ntp, qmi, qsmt, qss, stp, uo, onf, msr, ni, lpp, week, year, pred_vers, ensemble_vers, std_dev_type, own_vers)
        #     winnings.append(cur_winnings)
            
        winnings = Parallel(n_jobs=-1, verbose=0)(delayed(sim_winnings)(adj, pdm, md, tn, fmw, ct, mpst, mpot, ntp, 
                                                                        qmi, qsmt, qss, stp, uo, onf, msr, ni,lpp,
                                                                        week, year, pred_vers, ensemble_vers, std_dev_type, ownership_vers) for 
                                                                        adj, pdm, md, tn, fmw, ct, mpst, mpot, ntp, 
                                                                        qmi, qsmt, qss, stp, uo, onf, msr, ni,lpp,
                                                                        week, year, pred_vers, ensemble_vers, std_dev_type, ownership_vers in params)
        
        winnings = [item for sublist in winnings for item in sublist]
        winnings = avg_winnings_contest(winnings)
        winnings = int(np.sum(winnings))
        total_winnings.append(winnings)
        
        print(f'Week {week} Winnings: {winnings}') 
        print(f'Total Cumulative Winnings: {int(np.sum(total_winnings))}\n')
    
    total_winnings = adjust_high_winnings(total_winnings, max_adjust=10000)
    
    mean_loss = -np.mean(total_winnings)
    median_loss = -np.percentile(total_winnings, 50)
    loss = mean_loss + median_loss    
    print('Mean Loss:', mean_loss, 'Median Loss:', median_loss, ' Current Loss:', loss)
    return loss


#%%

#================
# Set Param spaces
#================

init_space = {

        'pred_vers': hp.choice('pred_vers', ['sera1_rsq0_brier1_matt1_lowsample_perc']),
         
        'ensemble_vers':  hp.choice('ensemble_vers', ['no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                                                      'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3']),

        'std_dev_type':  hp.choice('std_dev_type', ['pred_spline_class80_q80_matt1_brier1_kfold3']),

        'ownership_vers': hp.choice('ownership_vers', ['standard_ln', 'standard_ln_rank', 'standard_ln_rank_extra_features']),

        'adjust_pos_counts_True': hp.uniform('adjust_pos_counts_True', 0.7, 1),

        'player_drop_multiple_4': hp.uniform('player_drop_multiple_4', 0, 0.2),
        'player_drop_multiple_2': hp.uniform('player_drop_multiple_2', 0, 0.2),

        'matchup_drop_1': hp.uniform('matchup_drop_1', 0, 0.2),
        'matchup_drop_2': hp.uniform('matchup_drop_2', 0, 0.2),

        'top_n_choices_1': hp.uniform('top_n_choices_1', 0, 0.2),
        'top_n_choices_2': hp.uniform('top_n_choices_2', 0, 0.2),

        'full_model_weight_5': hp.uniform('full_model_weight_5', 0.7, 1),

        'covar_type_no_covar': hp.uniform('covar_type_no_covar', 0.6, 0.9),

        'min_player_same_team_2': hp.uniform('min_player_same_team_2', 0, 0.3),
        'min_player_same_team_3': hp.uniform('min_player_same_team_3', 0.2, 0.5),

        'min_player_opp_team_1': hp.uniform('min_player_opp_team_1', 0.3, 0.5),
        'min_player_opp_team_2': hp.uniform('min_player_opp_team_2', 0, 0.2),

        'num_top_players_2': hp.uniform('num_top_players_2', 0.3, 0.5),
        'num_top_players_3': hp.uniform('num_top_players_3', 0.3, 0.5),

        'qb_min_iter_0': hp.uniform('qb_min_iter_0', 0.8, 1),

        'qb_set_max_team_True': hp.uniform('qb_set_max_team_True', 0.8, 1),

        'qb_solo_start_True': hp.uniform('qb_solo_start_True', 0, 0.2),

        'static_top_players_True': hp.uniform('static_top_players_True', 0.4, 0.6),

        'use_ownership_0.9': hp.uniform('use_ownership_0.9', 0.3, 0.8),
        'use_ownership_0.8': hp.uniform('use_ownership_0.8', 0, 0.2),

        'own_neg_frac_0.8': hp.uniform('own_neg_frac_0.8', 0, 0.5),

        'max_salary_remain_200': hp.uniform('max_salary_remain_200', 0.2, 0.3),
        'max_salary_remain_500': hp.uniform('max_salary_remain_500', 0.2, 0.3),
        'max_salary_remain_1000': hp.uniform('max_salary_remain_1000', 0.2, 0.3),

        'num_iters_100': hp.uniform('num_iters_100', 0.8, 1),

        'lineups_per_param': hp.choice('lineups_per_param', [2, 3])
}


full_space = {

        'pred_vers': hp.choice('pred_vers', ['sera1_rsq0_brier1_matt1_lowsample_perc',
                                             'sera1_rsq0_brier1_matt1_lowsample_perc_ffa_fc']),
         
        'ensemble_vers':  hp.choice('ensemble_vers', ['no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3',
                                                      'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                                                      'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3_fullstack']),

        'std_dev_type':  hp.choice('std_dev_type', ['pred_spline_class80_q80_matt1_brier1_kfold3']),

        'ownership_vers': hp.choice('ownership_vers', ['standard_ln_rank_extra_features']),

        'adjust_pos_counts_True': hp.uniform('adjust_pos_counts_True', 0, 1),

        'player_drop_multiple_4': hp.uniform('player_drop_multiple_4', 0, 0.5),
        'player_drop_multiple_2': hp.uniform('player_drop_multiple_2', 0, 0.5),

        'matchup_drop_1': hp.uniform('matchup_drop_1', 0, 0.5),
        'matchup_drop_2': hp.uniform('matchup_drop_2', 0, 0.5),

        'top_n_choices_1': hp.uniform('top_n_choices_1', 0, 0.5),
        'top_n_choices_2': hp.uniform('top_n_choices_2', 0, 0.5),

        'full_model_weight_5': hp.uniform('full_model_weight_5', 0, 1),

        'covar_type_no_covar': hp.uniform('covar_type_no_covar', 0, 1),

        'min_player_same_team_2': hp.uniform('min_player_same_team_2', 0, 0.5),
        'min_player_same_team_3': hp.uniform('min_player_same_team_3', 0, 0.5),

        'min_player_opp_team_1': hp.uniform('min_player_opp_team_1', 0, 0.5),
        'min_player_opp_team_2': hp.uniform('min_player_opp_team_2', 0, 0.5),

        'num_top_players_2': hp.uniform('num_top_players_2', 0, 0.5),
        'num_top_players_3': hp.uniform('num_top_players_3', 0, 0.5),

        'qb_min_iter_0': hp.uniform('qb_min_iter_0', 0, 1),

        'qb_set_max_team_True': hp.uniform('qb_set_max_team_True', 0, 1),

        'qb_solo_start_True': hp.uniform('qb_solo_start_True', 0, 1),

        'static_top_players_True': hp.uniform('static_top_players_True', 0, 1),

        'use_ownership_0.9': hp.uniform('use_ownership_0.9', 0, 0.4),
        'use_ownership_0.8': hp.uniform('use_ownership_0.8', 0, 0.4),

        'own_neg_frac_0.8': hp.uniform('own_neg_frac_0.8', 0, 1),

        'max_salary_remain_200': hp.uniform('max_salary_remain_200', 0, 0.3),
        'max_salary_remain_500': hp.uniform('max_salary_remain_500', 0, 0.3),
        'max_salary_remain_1000': hp.uniform('max_salary_remain_1000', 0, 0.3),

        'num_iters_100': hp.uniform('num_iters_100', 0, 1),

        'lineups_per_param': hp.choice('lineups_per_param', [2, 3])
}


trial_name = 'adjust10000_week132021_week132022'

#%%

# class BayesTrain:
#     def __init__(self, trial_name, save_path, warm_start_evals=10, full_evals=50):

#         self.trial_name = trial_name
#         self.warm_start_evals = warm_start_evals
#         self.full_evals = full_evals
#         self.save_path = save_path

#     def check_warm_start_exists(self):
#         if os.path.exists(self.save_path+f'warm_start_{self.trial_name}.p'):
#             self.warm_start_exists = True

#     def check_full_trial_exists(self):
#         if os.path.exists(self.save_path+f'full_space_{self.trial_name}.p'):
#             self.full_trial_exists = True

    


if os.path.exists(save_path+f'warm_start_{trial_name}.p'):
    trials  = load_pickle(save_path, f'warm_start_{trial_name}')
    print('Loading Warm Start')
else:
    trials = Trials()
    print('Running Warm Start')
    best = fmin(objective, space=init_space, algo=tpe.suggest, trials=trials, max_evals=10)
    save_pickle(trials, save_path, f'warm_start_{trial_name}')

print('Running Full Space')
best = fmin(objective, space=full_space, algo=tpe.suggest, trials=trials, max_evals=100)
print(space_eval(full_space, best))

save_pickle(trials, save_path, f'full_space_{trial_name}')

results = results_to_df(save_path, f'full_space_{trial_name}')
dm.delete_from_db('Results', 'Entry_Optimize_Bayes', f"trial_name='{trial_name}'", create_backup=False)
dm.write_to_db(results, 'Results', 'Entry_Optimize_Bayes', 'append')

#%%


print('Running Full Space')
best = fmin(objective, space=full_space, algo=tpe.suggest, trials=trials, max_evals=75)
print(space_eval(full_space, best))

save_pickle(trials, save_path, f'full_space_{trial_name}')

#%%
show_trial_best_params('full_space_'+trial_name, 2)

# %%
