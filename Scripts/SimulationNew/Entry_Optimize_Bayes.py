#%%
from zSim_Helper_Covar import *

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral
from joblib import Parallel, delayed
import pickle
import gzip
import os
from hyperopt import fmin, tpe, hp, space_eval, Trials, atpe
import pprint
from functools import partial
import warnings
np.warnings = warnings

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


#===============
# Settings and User Inputs
#===============
# set the model version
set_weeks = [
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

]

set_years = [
      2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 
      2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023
]

save_path = "c:/Users/borys/OneDrive/Documents/Github/Daily_Fantasy/Model_Outputs/2022/Bayes_Sim_Opt/"
    
salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
adjust_winnings = False
set_max_team = None
total_lineups = 15

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


#================
# Saving output
#================

def results_to_df(save_path, fname):

    trials = load_pickle(save_path, fname)
    results = pd.DataFrame()
    for t in range(len(trials.trials)):
        cur_trial = trials.trials[t]
        loss = cur_trial['result']['loss']
        params = cur_trial['misc']['vals']
        for k, v in params.items():
            params[k] = v[0]
        
        params = pd.DataFrame(space_eval(full_space, params), index=[0]).assign(loss=loss)
        results = pd.concat([results, params], axis=0)

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
    print('Loss:', results.iloc[best_result]['loss'])
    results = results.drop(['trial_name', 'loss'], axis=1)
    
    best_params = results.iloc[best_result].to_dict()
    best_params_output = convert_param_options(best_params)
    

    for k, _ in best_params_output.items():
        for k_in, v_in in best_params_output[k].items():
            best_params_output[k][k_in] = np.round(v_in, 2)

    try:
        best_params_output['min_players_opp_team'] = best_params_output['min_player_opp_team']
        del best_params_output['min_player_opp_team']
    except:
        pass

    # adjust if not summed to 1
    for k, v in best_params_output.items():
        pct_off = 1 - np.sum(list(v.values()))
        if pct_off != 0:
            value_keys = list(v.keys())
            best_params_output[k][value_keys[-1]] = np.round(best_params_output[k][value_keys[-1]] + pct_off, 2)

    vers_names = ('pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers', 'ownership_vers', 'lineups_per_param')
    vers_params = {k: v for k,v in best_params.items() if k in vers_names}
    # best_params_output = {**vers_params, **best_params_output}

    pprint.pprint(vers_params, sort_dicts=False)
    print('\n')
    pprint.pprint(best_params_output, sort_dicts=False)

    
#================
# Run Sim code
#================

def convert_param_options(bp):
    d = {
        
        'adjust_pos_counts': {
            True: bp['adjust_pos_counts_True'],
            False: 1 - bp['adjust_pos_counts_True'],
        },

        'player_drop_multiple': {
            30:  bp['player_drop_multiple_30'],
            20:  bp['player_drop_multiple_20'],
            10:  bp['player_drop_multiple_10'],
            0: 1 - bp['player_drop_multiple_30'] - bp['player_drop_multiple_20'] - bp['player_drop_multiple_10']
        },

        'matchup_seed': {
            True: bp['matchup_seed_True'],
            False: 1 - bp['matchup_seed_True']
        },

        'matchup_drop': {
            1:bp['matchup_drop_1'],
            2: bp['matchup_drop_2'],
            3:  bp['matchup_drop_3'],
            0: 1 - bp['matchup_drop_1'] - bp['matchup_drop_2'] - bp['matchup_drop_3']
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
            'kmeans_pred_trunc_new': bp['covar_type_kmeans_pred_trunc_new'],
            'no_covar': bp['covar_type_no_covar'],
            'team_points_trunc': 1 - bp['covar_type_no_covar'] - bp['covar_type_kmeans_pred_trunc_new']
        }, 

        'max_team_type': {
            'player_points': bp['max_team_type_player_points'],
            'vegas_points': 1 - bp['max_team_type_player_points']
        },

        'min_player_same_team': {
            2: bp['min_player_same_team_2'],
            3: bp['min_player_same_team_3'],
            'Auto': 1 - bp['min_player_same_team_2'] - bp['min_player_same_team_3']
        }, 

        'min_players_opp_team': {
            1: bp['min_players_opp_team_1'],
            2: bp['min_players_opp_team_2'],
            'Auto': 1 - bp['min_players_opp_team_1'] - bp['min_players_opp_team_2']
        }, 

        'num_top_players': {
            2: bp['num_top_players_2'],
            3: bp['num_top_players_3'],
            5: 1 - bp['num_top_players_2'] - bp['num_top_players_3']
        },

        'ownership_vers': {
            'mil_only': bp['ownership_vers_mil_only'],
            'mil_times_standard_ln': bp['ownership_vers_mil_times_standard_ln'],
           'mil_div_standard_ln': bp['ownership_vers_mil_div_standard_ln'],
            'standard_ln': 1 -  bp['ownership_vers_mil_only'] -  bp['ownership_vers_mil_times_standard_ln'] - bp['ownership_vers_mil_div_standard_ln']
        },

        'qb_min_iter': {
            0: 1- bp['qb_min_iter_2'] - bp['qb_min_iter_9'],
            2: bp['qb_min_iter_2'],
            9: bp['qb_min_iter_9']
        },

        'qb_set_max_team': {
            True: bp['qb_set_max_team_True'],
            False: 1 - bp['qb_set_max_team_True']
        },

        'qb_solo_start': {
            True: bp['qb_solo_start_True'],
            False: 1 - bp['qb_solo_start_True']
        },

        'qb_stack_wt': {
            1: bp['qb_stack_wt_1'],
            2: bp['qb_stack_wt_2'],
            4: bp['qb_stack_wt_4'],
            3: 1 - bp['qb_stack_wt_1'] - bp['qb_stack_wt_2'] - bp['qb_stack_wt_4']
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
            0.9: bp['own_neg_frac_0.9'],
            1: 1 - bp['own_neg_frac_0.8'] - bp['own_neg_frac_0.9']
        },

        'max_salary_remain': {
            200: bp['max_salary_remain_200'],
            500: bp['max_salary_remain_500'],
            1000: 1 - bp['max_salary_remain_200'] - bp['max_salary_remain_500'] - bp['max_salary_remain_1500'],
            1500: bp['max_salary_remain_1500']
        },

        'num_iters': {
            150: 1 - bp['num_iters_50'] - bp['num_iters_100'],
            100: bp['num_iters_100'],
            50: bp['num_iters_50']
        },  

        'num_avg_pts': {
            10:  1 - bp['num_avg_pts_3'] - bp['num_avg_pts_7'] - bp['num_avg_pts_5'],
            7: bp['num_avg_pts_7'],
            5: bp['num_avg_pts_5'],
            3: bp['num_avg_pts_3']
            },

        'use_unique_players': {
            True: bp['use_unique_players_True'],
            False: 1 - bp['use_unique_players_True']
        },
    }

    for k,_ in d.items():
        for k2,v2 in d[k].items():
            d[k][k2] = np.round(v2,2)

    return d



def run_weekly_sim(d, week, year, salary_cap, pos_require_start, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups):

    rs = RunSim(dm, week, year, salary_cap, pos_require_start, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups)
    params = rs.generate_param_list(d)
    winnings, player_results = rs.run_multiple_lineups(params, calc_winnings=True, parallelize=False)

    return winnings, player_results


def objective(bayes_params):

    print('\n')

    pred_vers = bayes_params['pred_vers']
    reg_ens_vers = bayes_params['reg_ens_vers']
    std_dev_type = bayes_params['std_dev_type']
    million_ens_vers = bayes_params['million_ens_vers']

    d = convert_param_options(bayes_params)
    print('Reg Vers:', reg_ens_vers, 'Std Dev Type:', std_dev_type, 'Million Vers:', million_ens_vers)
    print(d)
    
    output = Parallel(n_jobs=-1, verbose=0)(
                                delayed(run_weekly_sim)(d, week, year, salary_cap, pos_require_start, pred_vers, reg_ens_vers, 
                                                   million_ens_vers, std_dev_type, total_lineups) for
                                week, year in zip(set_weeks, set_years)
                                )

    total_winnings = []
    player_results = pd.DataFrame()
    for w, l in output:
        total_winnings.append(w)
        player_results = pd.concat([player_results, l])

    total_winnings = list(np.round(np.array(total_winnings) * 13 / total_lineups,1))

    
    print('Unadjusted Winnings:', {i+1: t for i,t in zip(range(len(total_winnings)), total_winnings)})
    rs = RunSim(dm, 1, 2022, salary_cap, pos_require_start, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups)
    total_winnings = rs.adjust_high_winnings(total_winnings, max_adjust=15000)

    mean_loss = -np.mean(total_winnings)
    median_loss = -np.percentile(total_winnings, 80)
    var_loss = np.var(total_winnings)
    loss = mean_loss + median_loss
    print('Total Winnings:', np.sum(total_winnings), 'Mean Loss:', np.round(mean_loss,1), 
          'Median Loss (80th perc):', np.round(median_loss,1), ' Current Loss:', np.round(loss,1), 'Variance Loss:', np.round(np.sqrt(var_loss),1))

    return {'loss': loss, 'loss_variance': var_loss, 'status': 'ok'}

#%%

#================
# Set Param spaces
#================

param_spacing = 0.05

init_space = {

      'pred_vers': hp.choice('pred_vers', ['sera0_rsq0_mse1_brier1_matt1_bayes']),
         
        'reg_ens_vers':  hp.choice('reg_ens_vers', [
                                                   'random_full_stack_sera0_rsq0_mse1_include2_kfold3',
                                                   'random_full_stack_team_stats_sera0_rsq0_mse1_include2_kfold3',
                                                    'random_kbest_sera0_rsq0_mse1_include2_kfold3',
                                                    'random_kbest_team_stats_sera0_rsq0_mse1_include2_kfold3',
                                                     ]),

        'std_dev_type':  hp.choice('std_dev_type', [
                                                    'spline_pred_class80_q80_matt0_brier1_kfold3',
                                                    'spline_pred_class80_matt0_brier1_kfold3',
                                                    'spline_pred_q80_matt0_brier1_kfold3',
                                                    'spline_class80_q80_matt0_brier1_kfold3',
                                                    ]),

        'million_ens_vers':  hp.choice('million_ens_vers', [
                                                            'random_kbest_matt0_brier1_include2_kfold3',
                                                            'random_kbest_team_stats_matt0_brier1_include2_kfold3',
                                                            'random_full_stack_matt0_brier1_include2_kfold3',
                                                            'random_full_stack_team_stats_matt0_brier1_include2_kfold3',
                                                            ]),

        'ownership_vers_mil_only': hp.quniform('ownership_vers_mil_only', 0, 0.4, param_spacing),
        'ownership_vers_mil_times_standard_ln': hp.quniform('ownership_vers_mil_times_standard_ln', 0, 0.3, param_spacing),
        'ownership_vers_mil_div_standard_ln': hp.quniform('ownership_vers_mil_div_standard_ln', 0, 0.3, param_spacing),

        'adjust_pos_counts_True': hp.quniform('adjust_pos_counts_True', 0, 1, param_spacing),

        'player_drop_multiple_30': hp.quniform('player_drop_multiple_30', 0, 0.3, param_spacing),
        'player_drop_multiple_20': hp.quniform('player_drop_multiple_20', 0, 0.3, param_spacing),
        'player_drop_multiple_10': hp.quniform('player_drop_multiple_10', 0, 0.4, param_spacing),

        'matchup_seed_True': hp.quniform('matchup_seed_True', 0, 1, param_spacing),

        'matchup_drop_1': hp.quniform('matchup_drop_1', 0, 0.4, param_spacing),
        'matchup_drop_2': hp.quniform('matchup_drop_2', 0, 0.4, param_spacing),
        'matchup_drop_3': hp.quniform('matchup_drop_3', 0, 0.2, param_spacing),

        'top_n_choices_1': hp.quniform('top_n_choices_1', 0, 0.5, param_spacing),
        'top_n_choices_2': hp.quniform('top_n_choices_2', 0, 0.5, param_spacing),

        'full_model_weight_5': hp.quniform('full_model_weight_5', 0, 1, param_spacing),

        'covar_type_kmeans_pred_trunc_new': hp.quniform('covar_type_kmeans_pred_trunc_new', 0, 0.4, param_spacing),
        'covar_type_no_covar': hp.quniform('covar_type_no_covar', 0, 0.6, param_spacing),

        'max_team_type_player_points': hp.quniform('max_team_type_player_points', 0, 1, param_spacing),

        'min_player_same_team_2': hp.quniform('min_player_same_team_2', 0, 0.5, param_spacing),
        'min_player_same_team_3': hp.quniform('min_player_same_team_3', 0, 0.5, param_spacing),

        'min_players_opp_team_1': hp.quniform('min_players_opp_team_1', 0, 0.5, param_spacing),
        'min_players_opp_team_2': hp.quniform('min_players_opp_team_2', 0, 0.5, param_spacing),

        'num_top_players_2': hp.quniform('num_top_players_2', 0, 0.5, param_spacing),
        'num_top_players_3': hp.quniform('num_top_players_3', 0, 0.5, param_spacing),

        'qb_min_iter_9': hp.quniform('qb_min_iter_9', 0, 0.5, param_spacing),
        'qb_min_iter_2': hp.quniform('qb_min_iter_2', 0, 0.5, param_spacing),

        'qb_set_max_team_True': hp.quniform('qb_set_max_team_True', 0, 1, param_spacing),

        'qb_solo_start_True': hp.quniform('qb_solo_start_True', 0, 1, param_spacing),

        'qb_stack_wt_1': hp.quniform('qb_stack_wt_1', 0, 0.2, param_spacing),
        'qb_stack_wt_2': hp.quniform('qb_stack_wt_2', 0, 0.4, param_spacing),
        'qb_stack_wt_4': hp.quniform('qb_stack_wt_4', 0, 0.4, param_spacing),

        'static_top_players_True': hp.quniform('static_top_players_True', 0, 1, param_spacing),

        'use_ownership_0.9': hp.quniform('use_ownership_0.9', 0, 0.5, param_spacing),
        'use_ownership_0.8': hp.quniform('use_ownership_0.8', 0, 0.5, param_spacing),

        'own_neg_frac_0.8': hp.quniform('own_neg_frac_0.8', 0, 0.3, param_spacing),
        'own_neg_frac_0.9': hp.quniform('own_neg_frac_0.9', 0, 0.3, param_spacing),

        'max_salary_remain_200': hp.quniform('max_salary_remain_200', 0, 0.2, param_spacing),
        'max_salary_remain_500': hp.quniform('max_salary_remain_500', 0, 0.6, param_spacing),
        'max_salary_remain_1500': hp.quniform('max_salary_remain_1500', 0, 0.2, param_spacing),

        'num_iters_100': hp.quniform('num_iters_100', 0, 0.5, param_spacing),
        'num_iters_50': hp.quniform('num_iters_50', 0, 0.5, param_spacing),

        'num_avg_pts_3': hp.quniform('num_avg_pts_3', 0, 0.2, param_spacing),
        'num_avg_pts_7': hp.quniform('num_avg_pts_7', 0, 0.5, param_spacing),
        'num_avg_pts_5': hp.quniform('num_avg_pts_5', 0, 0.3, param_spacing),

        'use_unique_players_True': hp.quniform('use_unique_players_True', 0, 1, param_spacing),

}


full_space = init_space.copy()

#%%

# class BayesTrain:
#     def __init__(self, trial_name, save_path, warm_start_evals=10, full_evals=50):

#         self.trial_name = trial_name
#         self.warm_start_evals = warm_start_evals
#         self.full_evals = full_evals
#         self.save_path = save_path
#         self.warm_start_exists = self.check_warm_start_exists()
#         self.full_trial_exists = self.check_full_trial_exists()

#     def check_warm_start_exists(self):
#         if os.path.exists(self.save_path+f'warm_start_{self.trial_name}.p'):
#             self.warm_start_exists = True

#         else:
#             self.warm_start_exists = False

#     def check_full_trial_exists(self):
#         if os.path.exists(self.save_path+f'full_space_{self.trial_name}.p'):
#             self.full_trial_exists = True

#         else:
#             self.full_trial_exists = False

#     def run_bayes_opt(self):
#         if self.full_trial_exists:
#             self.trials  = load_pickle(self.save_path, f'full_space_{se.ftrial_name}')
#             save_pickle(trials, self.save_path, f'full_space_{self.trial_name}')

#         elif self.warm_start_exists and not self.full_trial_exists:
#             self.trials  = load_pickle(self.save_path, f'warm_start_{self.trial_name}')

#         else:
#             trials = Trials()
#             save_pickle(trials, save_path, f'warm_start_{trial_name}')
#%%
from wakepy import keep        

trial_name = 'mean_perc80_adjust15000_full_2022_2023_13lineups_rounded_1trial_lossvar_newplayerdrop'
# trial_name = 'test'

with keep.running() as m:

    for i in range(15,16):

        if os.path.exists(save_path+f'full_space_{trial_name}.p'):

            trials  = load_pickle(save_path, f'full_space_{trial_name}')
            print('Loading Full Space')

            print('Running Full Space')
            best = fmin(objective, space=full_space, algo=tpe.suggest, trials=trials, max_evals=i*30)
            print(space_eval(full_space, best))
            save_pickle(trials, save_path, f'full_space_{trial_name}')
        
        elif os.path.exists(save_path+f'warm_start_{trial_name}.p'):
            trials  = load_pickle(save_path, f'warm_start_{trial_name}')
            print('Loading Warm Start')

            print('Running Full Space')
            best = fmin(objective, space=full_space, algo=tpe.suggest, trials=trials, max_evals=i*30)
            print(space_eval(full_space, best))
            save_pickle(trials, save_path, f'full_space_{trial_name}')
        
        else:
            trials = Trials()
            print('Running Warm Start')
            best = fmin(objective, space=init_space, algo=tpe.suggest, trials=trials, max_evals=i*30)
            save_pickle(trials, save_path, f'warm_start_{trial_name}')

    

#%%
trial_name = 'mean_perc80_adjust15000_full_2022_2023_13lineups_rounded_1trial_lossvar_newplayerdrop'
results = results_to_df(save_path, f'full_space_{trial_name}')
dm.delete_from_db('Results', 'Entry_Optimize_Bayes', f"trial_name='{trial_name}'", create_backup=False)
dm.write_to_db(results, 'Results', 'Entry_Optimize_Bayes', 'replace')

#%%

dm.read(f"SELECT * FROM Entry_Optimize_Bayes", 'Results')

#%%
trial_name = 'mean_perc80_adjust15000_full_2022_2023_13lineups_rounded_1trial_lossvar_newplayerdrop'
# trial_name = 'adjust5000_mean_median_week1to17_2022_million_own_pct_matchupdropnew'
show_trial_best_params('full_space_'+trial_name, 7)

# %%
