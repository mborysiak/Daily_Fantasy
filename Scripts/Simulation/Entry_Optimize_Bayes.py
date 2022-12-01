#%%
from zSim_Helper_Covar import *

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral
from joblib import Parallel, delayed
import pickle
import gzip

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#===============
# Settings and User Inputs
#===============
# set the model version
set_weeks = [
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
]

set_years = [
      2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022
]

save_path = "c:/Users/mborysia/Documents/Github/Daily_Fantasy/Model_Outputs/2022/Bayes_Sim_Opt/"

pred_versions = len(set_weeks)*['sera1_rsq0_brier1_matt1_lowsample_perc']

ensemble_versions =[
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    ]

std_dev_types = [
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                ]

sim_types = len(set_weeks) * ['ownership_ln_pos_fix_flip']
    
salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
adjust_winnings = False
set_max_team = None


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


def create_params_list(d, lineups_per_param, week, year, pred_vers, ensemble_vers, std_dev_type):
    params = []
    for i in range(int(30/lineups_per_param)):
        cur_params = []
        for param, param_options in d.items():
            param_vars = list(param_options.keys())
            param_prob = list(param_options.values())
            cur_params.append(np.random.choice(param_vars, p=param_prob))

        cur_params.extend([lineups_per_param, week, year, pred_vers, ensemble_vers, std_dev_type])
        params.append(cur_params)
    return params

#%%

# def sim_winnings(adjust_select, player_drop_multiplier, matchup_drop, top_n_choices, 
#                 full_model_rel_weight, covar_type, min_players_same_team, 
#                 min_players_opp_team, num_top_players, qb_min_iter, qb_set_max_team, qb_solo_start,
#                 static_top_players, use_ownership, own_neg_frac, max_salary_remain, 
#                 num_iters, lineups_per_param, week, year, pred_vers, ensemble_vers, std_dev_type
#                 ):
    
#     prizes = get_prizes(week, year)
#     player_teams, matchups = pull_matchups(week, year, pred_vers)
#     points = pull_points(week, year)

#     try: min_players_opp_team = int(min_players_opp_team)
#     except: pass

#     try: min_players_same_team = float(min_players_same_team)
#     except: pass

#     if covar_type=='no_covar': use_covar=False
#     else: use_covar=True

#     sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
#                                 pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
#                                 covar_type=covar_type, full_model_rel_weight=full_model_rel_weight, 
#                                 use_covar=use_covar, use_ownership=use_ownership, 
#                                 salary_remain_max=max_salary_remain)

#     winnings = []
#     for _ in range(int(6/lineups_per_param)):
        
#         total_add = []
#         to_drop_selected = []
#         for t in range(lineups_per_param):

#             to_add = []
#             to_drop = []
#             to_drop.extend(to_drop_selected)
            
#             # if matchup_drop > 0: to_drop = rand_drop_teams(matchups, matchup_drop)
#             # else: to_drop = []
#             # to_drop.extend(to_drop_selected)

#             for i in range(9):
#                 results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, 
#                                         min_players_opp_team_input=min_players_opp_team, 
#                                         adjust_select=adjust_select, num_matchup_drop=matchup_drop,
#                                         own_neg_frac=own_neg_frac, n_top_players=num_top_players,
#                                         static_top_players=static_top_players, qb_min_iter=qb_min_iter,
#                                         qb_set_max_team=qb_set_max_team, qb_solo_start=qb_solo_start)
                
#                 prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
#                 try: 
#                     selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
#                     to_add.append(selected_player)
#                 except: 
#                     pass
                
#             prize_money = calc_winnings(to_add, points, prizes)
#             winnings.append(prize_money)

#             total_add.extend(to_add)
#             to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier, lineups_per_param)

#     return np.mean(winnings)


# def objective(params):
    
#     iter_cats = zip(set_weeks, set_years, pred_versions, ensemble_versions, std_dev_types)
#     input_args = []
#     for week, year, pred_vers, ensemble_vers, std_dev_type in iter_cats:
#         input_args.append([params['adjust_pos_counts'], 
#                            params['player_drop_multiple'], 
#                            params['matchup_drop'], 
#                            params['top_n_choices'], 
#                            params['full_model_weight'], 
#                            params['covar_type'], 
#                            params['min_player_same_team'], 
#                            params['min_players_opp_team'], 
#                            params['num_top_players'], 
#                            params['qb_min_iter'], 
#                            params['qb_set_max_team'], 
#                            params['qb_solo_start'], 
#                            params['static_top_players'], 
#                            params['use_ownership'], 
#                            params['own_neg_frac'], 
#                            params['max_salary_remain'], 
#                            params['num_iters'], 
#                            params['lineups_per_param'],
#                            week, year, pred_vers, ensemble_vers, std_dev_type])

#     # for p1, p2, p3, p4, p5, p6,p7, p8, p9, p10, p11, p12,p13, p14, p15, p16, p17, p18,p19, p20, p21, p22, p23 in input_args[:3]:

#     #     winnings = sim_winnings(p1, p2, p3, p4, p5, p6,
#     #                 p7, p8, p9, p10, p11, p12,
#     #                 p13, p14, p15, p16, p17, p18,
#     #                 p19, p20, p21, p22, p23)
#     #     print(winnings)
                
#     winnings = Parallel(n_jobs=-1, verbose=0)(delayed(sim_winnings)(p1, p2, p3, p4, p5, p6,
#                                                                    p7, p8, p9, p10, p11, p12,
#                                                                    p13, p14, p15, p16, p17, p18,
#                                                                    p19, p20, p21, p22, p23) for 
#                                                                    p1, p2, p3, p4, p5, p6,
#                                                                    p7, p8, p9, p10, p11, p12,
#                                                                    p13, p14, p15, p16, p17, p18,
#                                                                    p19, p20, p21, p22, p23 in input_args)
    
#     print(input_args[0])
#     print({'Week'+str(i+1): w for i, w in enumerate(winnings)}) 
#     print('Total Winnings:', np.sum(winnings))
    
#     return -np.mean(winnings)-np.percentile(winnings, 50)
        
# #%%
# from hyperopt import fmin, tpe, hp, space_eval, Trials

# space = {
#         'adjust_pos_counts_True': hp.choice('adjust_pos_counts_True', [True, False]),
#         'player_drop_multiple': hp.choice('player_drop_multiple',  np.arange(0, 5, dtype=int)),
#         'matchup_drop': hp.choice('matchup_drop', np.arange(0, 3, dtype=int)),
#         'top_n_choices': hp.choice('top_n_choices',  np.arange(0, 4, dtype=int)),
#         'full_model_weight': hp.choice('full_model_weight', [0.2, 1, 5]),
#         'covar_type': hp.choice('covar_type', ['no_covar', 'team_points_trunc']),
#         'min_player_same_team': hp.choice('min_player_same_team', ['Auto', 2, 3]),
#         'min_players_opp_team': hp.choice('min_players_opp_team', ['Auto', 1, 2]),
#         'num_top_players': hp.choice('num_top_players', np.arange(2, 6, dtype=int)),
#         'qb_min_iter': hp.choice('qb_min_iter', [0, 1, 9]),
#         'qb_set_max_team': hp.choice('qb_set_max_team', [True, False]),
#         'qb_solo_start': hp.choice('qb_solo_start', [True, False]),
#         'static_top_players': hp.choice('static_top_players', [True, False]),
#         'use_ownership': hp.uniform('use_ownership', 0, 1),
#         'own_neg_frac': hp.uniform('own_neg_frac', 0, 1),
#         'max_salary_remain': hp.choice('max_salary_remain', np.arange(200, 2100, 100, dtype=int)),
#         'num_iters': hp.choice('num_iters', np.arange(50, 175, 25, dtype=int)),
#         'lineups_per_param': hp.choice('lineups_per_param', [2,3])
# }
# trials = Trials()
# fmin_result = fmin(objective, space, trials=trials, algo=tpe.suggest, max_evals=100)
# print(space_eval(space, fmin_result))

# # %%
# fmin_result = {'adjust_pos_counts': 0.74564620653125,
#  'covar_type': 0,
#  'full_model_weight': 2,
#  'lineups_per_param': 0,
#  'matchup_drop': 0,
#  'max_salary_remain': 4,
#  'min_player_same_team': 0,
#  'min_players_opp_team': 0,
#  'num_iters': 5,
#  'num_top_players': 3,
#  'own_neg_frac': 0.7065411384909316,
#  'player_drop_multiple': 0.9807781565825981,
#  'qb_min_iter': 1,
#  'qb_set_max_team': 1,
#  'qb_solo_start': 1,
#  'static_top_players': 1,
#  'top_n_choices': 0.18001757324562861,
#  'use_ownership': 0.6085929357842961}
 
# fmin_result_2 = {'adjust_pos_counts': True,
#  'covar_type': 'team_points_trunc',
#  'full_model_weight': 5,
#  'lineups_per_param': 2,
#  'matchup_drop': 2,
#  'max_salary_remain': 600,
#  'min_player_same_team': 'Auto',
#  'min_players_opp_team': 2,
#  'num_iters': 50,
#  'num_top_players': 2,
#  'own_neg_frac': 0.8994698832136082,
#  'player_drop_multiple': 1,
#  'qb_min_iter': 1,
#  'qb_set_max_team': True,
#  'qb_solo_start': False,
#  'static_top_players': True,
#  'top_n_choices': 0,
#  'use_ownership': 0.6761495505970899}

# %%

def sim_winnings(adjust_select, player_drop_multiplier, matchup_drop, top_n_choices, 
                full_model_rel_weight, covar_type, min_players_same_team, 
                min_players_opp_team, num_top_players, qb_min_iter, qb_set_max_team, qb_solo_start,
                static_top_players, use_ownership, own_neg_frac, max_salary_remain, 
                num_iters, lineups_per_param, week, year, pred_vers, ensemble_vers, std_dev_type
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
                                covar_type=covar_type, full_model_rel_weight=full_model_rel_weight, 
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


def objective(bayes_params):
    
    print({k: np.round(v,3) for k,v in bayes_params.items()})
    print('\n')

    lineups_per_param = bayes_params['lineups_per_param']

    d = {
        'adjust_pos_counts': {
            True: bayes_params['adjust_pos_counts_True'],
            False: 1 - bayes_params['adjust_pos_counts_True']
        },

        'player_drop_multiple': {
            4: bayes_params['player_drop_multiple_4'],
            2: bayes_params['player_drop_multiple_2'],
            0: 1 - bayes_params['player_drop_multiple_4'] - bayes_params['player_drop_multiple_2']
        },

        'matchup_drop': {
            1: bayes_params['matchup_drop_1'],
            2: bayes_params['matchup_drop_2'],
            0: 1 - bayes_params['matchup_drop_1'] - bayes_params['matchup_drop_2']
        }, 

        'top_n_choices': {
            1: bayes_params['top_n_choices_1'],
            2: bayes_params['top_n_choices_2'],
            0: 1 - bayes_params['top_n_choices_1'] - bayes_params['top_n_choices_2']
        },

        'full_model_weight': {
            5: bayes_params['full_model_weight_5'],
            0.2: 1 -  bayes_params['full_model_weight_5']
        }, 

        'covar_type': {
            'no_covar': bayes_params['covar_type_no_covar'],
            'team_points_trunc': 1 -  bayes_params['covar_type_no_covar']
        }, 

        'min_player_same_team': {
            2: bayes_params['min_player_same_team_2'],
            3: bayes_params['min_player_same_team_3'],
            'Auto': 1 - bayes_params['min_player_same_team_2'] - bayes_params['min_player_same_team_3']
        }, 

        'min_player_opp_team': {
            1: bayes_params['min_player_opp_team_1'],
            2: bayes_params['min_player_opp_team_2'],
            'Auto': 1 - bayes_params['min_player_opp_team_1'] - bayes_params['min_player_opp_team_2']
        }, 

        'num_top_players': {
            2: bayes_params['num_top_players_2'],
            3: bayes_params['num_top_players_3'],
            5: 1 - bayes_params['num_top_players_2'] - bayes_params['num_top_players_3']
        },

        'qb_min_iter': {
            0: bayes_params['qb_min_iter_0'],
            9: 1 - bayes_params['qb_min_iter_0']
        },

        'qb_set_max_team': {
            True: bayes_params['qb_set_max_team_True'],
            False: 1 - bayes_params['qb_set_max_team_True']
        },

        'qb_solo_start': {
            True: bayes_params['qb_solo_start_True'],
            False: 1 - bayes_params['qb_solo_start_True']
        },

        'static_top_players': {
            True: bayes_params['static_top_players_True'],
            False: 1 - bayes_params['static_top_players_True']
        },

        'use_ownership': {
            0.9: bayes_params['use_ownership_0.9'],
            0.8: bayes_params['use_ownership_0.8'],
            1: 1 - bayes_params['use_ownership_0.9'] - bayes_params['use_ownership_0.8']
        },

        'own_neg_frac': {
            0.8: bayes_params['own_neg_frac_0.8'],
            1: 1 - bayes_params['own_neg_frac_0.8'],
        },

        'max_salary_remain': {
            200: bayes_params['max_salary_remain_200'],
            500: bayes_params['max_salary_remain_500'],
            1000: bayes_params['max_salary_remain_1000'],
            1500: 1 - bayes_params['max_salary_remain_200'] - bayes_params['max_salary_remain_500'] - bayes_params['max_salary_remain_1000'],
        },

        'num_iters': {
            100: bayes_params['num_iters_100'],
            50: 1 - bayes_params['num_iters_100'],
        },
    }

    
    total_winnings = []
    iter_cats = zip(set_weeks, set_years, pred_versions, ensemble_versions, std_dev_types)
    for week, year, pred_vers, ensemble_vers, std_dev_type in iter_cats:
        params = create_params_list(d, lineups_per_param, week, year, pred_vers, ensemble_vers, std_dev_type)

        # for adj, pdm, md, tn, fmw, ct, mpst, mpot, ntp, qmi, qsmt, qss, stp, uo, onf, msr, ni, lpp, week, year, pred_vers, ensemble_vers, std_dev_type in params:
        #     winnings = sim_winnings(adj, pdm, md, tn, fmw, ct, mpst, mpot, ntp, qmi, qsmt, qss, stp, uo, onf, msr, ni, lpp, week, year, pred_vers, ensemble_vers, std_dev_type)
                
        winnings = Parallel(n_jobs=-1, verbose=0)(delayed(sim_winnings)(adj, pdm, md, tn, fmw, ct, mpst, mpot, ntp, 
                                                                        qmi, qsmt, qss, stp, uo, onf, msr, ni,lpp,
                                                                        week, year, pred_vers, ensemble_vers, std_dev_type) for 
                                                                        adj, pdm, md, tn, fmw, ct, mpst, mpot, ntp, 
                                                                        qmi, qsmt, qss, stp, uo, onf, msr, ni,lpp,
                                                                        week, year, pred_vers, ensemble_vers, std_dev_type in params)
        
        winnings = [item for sublist in winnings for item in sublist]
        winnings = avg_winnings_contest(winnings)
        winnings = int(np.sum(winnings))
        total_winnings.append(winnings)

        
        print(f'Week {week} Winnings: {winnings}') 
        print(f'Total Cumulative Winnings: {int(np.sum(total_winnings))}\n')
    
    return -np.mean(total_winnings)-np.percentile(total_winnings, 50)
        
#%%
from hyperopt import fmin, tpe, hp, space_eval, Trials

full_space = {
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


init_space = {
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

        'use_ownership_0.9': hp.uniform('use_ownership_0.9', 0.3, 0.6),
        'use_ownership_0.8': hp.uniform('use_ownership_0.8', 0, 0.1),

        'own_neg_frac_0.8': hp.uniform('own_neg_frac_0.8', 0, 0.2),

        'max_salary_remain_200': hp.uniform('max_salary_remain_200', 0.2, 0.3),
        'max_salary_remain_500': hp.uniform('max_salary_remain_500', 0.2, 0.3),
        'max_salary_remain_1000': hp.uniform('max_salary_remain_1000', 0.2, 0.3),

        'num_iters_100': hp.uniform('num_iters_100', 0.8, 1),

        'lineups_per_param': hp.choice('lineups_per_param', [2, 3])
}

import os

save_path = "c:/Users/mborysia/Documents/Github/Daily_Fantasy/Model_Outputs/Bayes_Sim_Opt/"
if os.path.exists(save_path):
    # load saved initial parameters if they exist
    trials  = pickle.load(open(save_path, "rb"))
else:
    trials = Trials()

    # fill the trials object with 5 evals of the first restriction
    best = fmin(
        objective, space=init_space, algo=tpe.suggest, trials=trials, max_evals=10
    )

    # save for later
    pickle.dump(trials, open(save_path, 'wb'))

# run 10 evals after the 'warm start'
best = fmin(
    objective, space=full_space, algo=tpe.suggest, trials=trials, max_evals=25
)

print(space_eval(full_space, best))

pickle.dump(trials, open("c:/Users/mborysia/Documents/Github/Daily_Fantasy/Model_Outputs/Bayes_Sim_Opt/full_trials.p", "wb"))
# %%


save_pickle(trials, save_path, 'warm_start_week12')

