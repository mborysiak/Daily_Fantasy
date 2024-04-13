#%%
from zSim_Helper_Covar import *

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral
from joblib import Parallel, delayed
from wakepy import keep

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
conn = dm.db_connect('Simulation')


#%%
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

# set_weeks=[14]
# set_years=[2022]

model_vers = {'million_ens_vers': 'random_full_stack_team_stats_matt0_brier1_include2_kfold3',
 'pred_vers': 'sera0_rsq0_mse1_brier1_matt1_bayes',
 'reg_ens_vers': 'random_full_stack_sera0_rsq0_mse1_include2_kfold3',
 'std_dev_type': 'spline_class80_q80_matt0_brier1_kfold3'}


pred_vers = model_vers['pred_vers']
reg_ens_vers = model_vers['reg_ens_vers']
million_ens_vers = model_vers['million_ens_vers']
std_dev_type = model_vers['std_dev_type']

d = {'adjust_pos_counts': {False: 0.7, True: 0.3},
 'covar_type': {'kmeans_pred_trunc': 0.0,
                'kmeans_pred_trunc_new': 0.0,
                'no_covar': 0.3,
                'team_points_trunc': 0.7},
 'full_model_weight': {0.2: 0.5, 5: 0.5},
 'matchup_drop': {0: 0.7, 1: 0.1, 2: 0.2, 3: 0.0},
 'matchup_seed': {0: 0.3, 1: 0.7},
 'max_salary_remain': {200: 0.0, 500: 0.6, 1000: 0.4, 1500: 0.0},
 'max_team_type': {'player_points': 0.5, 'vegas_points': 0.5},
 'min_player_same_team': {2: 0.2, 3: 0.2, 'Auto': 0.6},
 'min_players_opp_team': {1: 0.1, 2: 0.2, 'Auto': 0.7},
 'num_avg_pts': {1: 0.0, 2: 0.0, 3: 0.0, 5: 0.0, 7: 0.3, 10: 0.7},
 'num_iters': {50: 0.0, 100: 0.0, 150: 1.0},
 'num_top_players': {2: 0.4, 3: 0.6, 5: 0.0},
 'own_neg_frac': {0.8: 0.0, 0.9: 0.0, 1: 1.0},
 'ownership_vers': {'mil_div_standard_ln': 0.0,
                    'mil_only': 0.0,
                    'mil_times_standard_ln': 0.3,
                    'standard_ln': 0.7},
 'player_drop_multiple': {0: 0.4, 2: 0.2, 4: 0.4, 10: 0.0, 20: 0.0, 30: 0.0},
 'qb_min_iter': {0: 0.6, 2: 0.4, 4: 0.0, 9: 0.0},
 'qb_set_max_team': {0: 0.7, 1: 0.3},
 'qb_solo_start': {False: 1.0, True: 0.0},
 'qb_stack_wt': {1: 0.0, 2: 0.0, 3: 0.7, 4: 0.3},
 'static_top_players': {False: 0.3, True: 0.7},
 'top_n_choices': {0: 1.0, 1: 0.0, 2: 0.0},
 'use_ownership': {0.7: 0.0, 0.8: 0.2, 0.9: 0.0, 1: 0.8},
 'use_unique_players': {0: 1.0, 1: 0.0}}

salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
adjust_winnings = False
set_max_team = None

entry_type = 'millions_only'
if entry_type == 'millions_playaction': total_lineups = 30
elif entry_type == 'millions_only': total_lineups = 13

rs = RunSim(dm, 1, 2022, salary_cap, pos_require_start, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups)

max_trial_num = dm.read("SELECT max(trial_num) FROM Entry_Optimize_Params_Detail", 'Results').values[0][0]
trial_num = max_trial_num + 1

def run_weekly_sim(d, week, year, salary_cap, pos_require_start, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups):

    rs = RunSim(dm, week, year, salary_cap, pos_require_start, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups)
    params = rs.generate_param_list(d)
    winnings, player_results, winnings_list = rs.run_multiple_lineups(params, calc_winnings=True, parallelize=False)
    print(player_results)

    return winnings, player_results, params, winnings_list


def objective(param_options, pred_vers, reg_ens_vers, std_dev_type, million_ens_vers, total_lineups, set_weeks, 
              set_years, salary_cap, pos_require_start):
    
    output = Parallel(n_jobs=-1, verbose=0)(
                                delayed(run_weekly_sim)(param_options, week, year, salary_cap, pos_require_start, pred_vers, reg_ens_vers, 
                                                        million_ens_vers, std_dev_type, total_lineups) for
                                week, year in zip(set_weeks, set_years)
                                )

    total_winnings = []
    player_results = pd.DataFrame()
    param_output = []
    winnings_list = []
    for w, pl, par, wl in output:
        total_winnings.append(w)
        player_results = pd.concat([player_results, pl])
        param_output.append(par)
        winnings_list.append(wl)

    total_winnings = list(np.round(np.array(total_winnings) * 13 / total_lineups,1))

    print('Total Winnings:', np.sum(total_winnings))
    win_dict = {f'{week}_{year}': t for week, year ,t in zip(set_weeks, set_years, total_winnings)}
    print_df = pd.DataFrame(win_dict, index=[0]).T.reset_index()
    print_df.columns = ['date', 'winnings']
    print(print_df)

    return total_winnings, player_results, param_output, winnings_list


def format_output_results(weekly_winnings, set_weeks, set_years, pred_vers, reg_ens_vers, std_dev_type, 
                       million_ens_vers, trial_num, repeat_num, entry_type):
    output_results = []
    for week_win_amt, week, year in zip(weekly_winnings, set_weeks, set_years):
        output_results.append([week, year, pred_vers, reg_ens_vers, std_dev_type, million_ens_vers, week_win_amt])

    # save out the high level results of the overall week
    output_results = pd.DataFrame(output_results, columns=['week', 'year', 'pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers', 'avg_winnings'])
    output_results['ownership_vers'] = 'variable'
    output_results['trial_num'] = trial_num
    output_results['repeat_num'] = repeat_num
    output_results['entry_type'] = entry_type

    return output_results


def format_player_results(player_results, total_lineups, repeat_num, trial_num):
    player_results = player_results.assign(repeat_num=repeat_num, trial_num=trial_num)
    player_results['unique_lineup_num'] = player_results.lineup_num + (player_results.repeat_num * total_lineups)
    player_results = player_results[['player','fantasy_pts', 'lineup_num', 'repeat_num', 'trial_num', 'unique_lineup_num', 'week', 'year']]
    return player_results

def detailed_param_output(rs, param_output, winnings_list, set_weeks, set_years, trial_num, repeat_num):
    rs.col_ordering = [c for c in rs.col_ordering if c != 'lineup_num']
    cols = rs.col_ordering
    cols.append('lineup_num')

    param_output_df = pd.DataFrame()
    for w, yr, po in zip(set_weeks, set_years, param_output):
        param_output_cur = pd.DataFrame(po, columns=cols).assign(week=w, year=yr, trial_num=trial_num, repeat_num=repeat_num)
        param_output_df = pd.concat([param_output_df, param_output_cur], axis=0)

    winning_list_long = [w for ws in winnings_list for w in ws]
    param_output_df['winnings'] = winning_list_long

    return param_output_df

def param_set_output(d):
            output = pd.DataFrame(d).T
            output = pd.melt(output.reset_index(), id_vars='index').dropna().sort_values(by='index')
            lpp = pd.DataFrame({'index': ['lineups_per_param'],
                                'variable': [1],
                                'value': 1})
            output = pd.concat([output, lpp], axis=0)
            output['trial_num'] = trial_num
            output.columns = ['param', 'param_option', 'option_value', 'trial_num']

            return output


print('Trial #', trial_num, '\n==============\n')
with keep.running() as m:        

    for repeat_num in range(10):

        print('Repeat #', repeat_num, '\n==============\n')

        weekly_winnings, player_results, param_output, winning_list = objective(d, pred_vers, reg_ens_vers, std_dev_type, million_ens_vers, total_lineups, 
                                                                        set_weeks, set_years, salary_cap, pos_require_start)


        output_results = format_output_results(weekly_winnings, set_weeks, set_years, pred_vers, reg_ens_vers, std_dev_type,
                                               million_ens_vers, trial_num, repeat_num, entry_type)
        dm.write_to_db(output_results, 'Results', 'Entry_Optimize_Results', 'append')

        player_results_out = format_player_results(player_results, total_lineups, repeat_num, trial_num)
        dm.write_to_db(player_results_out, 'Results', 'Entry_Optimize_Lineups', 'append')

        param_output_df = detailed_param_output(rs, param_output, winning_list, set_weeks, set_years, trial_num, repeat_num)
        dm.write_to_db(param_output_df, 'Results', 'Entry_Optimize_Params_Detail', 'append')

    # save out the initial params that were set for randomization
    output = param_set_output(d)
    dm.write_to_db(output, 'Results', 'Entry_Optimize_Params', 'append')


    
#%%





#%%

to_delete_num=520
df = dm.read(f"SELECT * FROM Entry_Optimize_Lineups WHERE trial_num!={to_delete_num}", 'Results')
dm.write_to_db(df, 'Results', 'Entry_Optimize_Lineups', 'replace')

df = dm.read(f"SELECT * FROM Entry_Optimize_Params WHERE trial_num!={to_delete_num}", 'Results')
dm.write_to_db(df, 'Results', 'Entry_Optimize_Params', 'replace')

df = dm.read(f"SELECT * FROM Entry_Optimize_Params_Detail WHERE trial_num!={to_delete_num}", 'Results')
dm.write_to_db(df, 'Results', 'Entry_Optimize_Params_Detail', 'replace')

df = dm.read(f"SELECT * FROM Entry_Optimize_Results WHERE trial_num!={to_delete_num}", 'Results')
dm.write_to_db(df, 'Results', 'Entry_Optimize_Results', 'replace')


#%%

df = dm.read(f"SELECT * FROM Entry_Optimize_Params", 'Results')
add_on = pd.DataFrame({'trial_num': range(df.trial_num.max()+1)})
add_on = add_on.assign(param='use_unique_players', param_option=False, option_value=1)
add_on = add_on[df.columns]

df = pd.concat([df, add_on], axis=0)
df = df.sort_values(by='trial_num')

# # df.loc[(df.trial_num.isin([84])) & (df.param=='num_iters'), ['param_option', 'option_value']] = [100, 1]
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Params', 'replace', create_backup=True)

#%%


df = dm.read(f"SELECT * FROM Entry_Optimize_Params_Detail", 'Results')
df['use_unique_players'] = False
# df.loc[df.trial_num.isin([84]), 'num_iters'] = 100
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Params_Detail', 'replace')

#%%

# df = dm.read(f"SELECT * FROM Entry_Optimize_Results", 'Results')

# df['rk'] = np.where(df.reg_ens_vers.str.contains('random_kbest'), 'random_kbest_', 'random_')
# df['matthews'] = np.where(df.std_dev_type.str.contains('matt1'), 'matt1_', 'matt0_')
# df['brier'] = np.where(df.std_dev_type.str.contains('brier1'), 'brier1_', 'brier0_')
# df['include'] = np.where(df.reg_ens_vers.str.contains('include2'), 'include2_', 'include0_')
# df['kfold'] = np.where(df.reg_ens_vers.str.contains('kfold3'), 'kfold3', 'kfold0')

# df.loc[(df.trial_num > 162) & (df.million_ens_vers.isnull()), 'million_ens_vers'] = (
#     df.loc[(df.trial_num > 162) & (df.million_ens_vers.isnull()), 'rk'] +
#     df.loc[(df.trial_num > 162) & (df.million_ens_vers.isnull()), 'matthews'] +
#     df.loc[(df.trial_num > 162) & (df.million_ens_vers.isnull()), 'brier'] +
#     df.loc[(df.trial_num > 162) & (df.million_ens_vers.isnull()), 'include'] +
#     df.loc[(df.trial_num > 162) & (df.million_ens_vers.isnull()), 'kfold']
# )

# df = df.drop(['rk', 'matthews', 'brier', 'include', 'kfold'], axis=1)
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Results', 'replace')

#%%

# df = dm.read(f"SELECT * FROM Entry_Optimize_Results", 'Results')
# df.loc[(df.trial_num==286) & (df.million_ens_vers == 'kbest_matt0_brier1_include2_kfold3'), 'trial_num'] = 292
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Results', 'replace')

# # %%
# df = dm.read(f"SELECT * FROM Entry_Optimize_Params_Detail", 'Results')
# idx = df[df.trial_num==286].iloc[4800:].index
# df.loc[df.index.isin(idx), 'trial_num'] = 292
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Params_Detail', 'replace')

# # %%

# df = dm.read(f"SELECT * FROM Entry_Optimize_Params", 'Results')
# idx = df.loc[df.trial_num==286].iloc[62:].index
# df.loc[df.index.isin(idx), 'trial_num'] = 292
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Params', 'replace')

# # %%

# df = dm.read(f"SELECT * FROM Entry_Optimize_Lineups", 'Results')
# idx = df[df.trial_num==286].iloc[43200:].index
# df.loc[df.index.isin(idx), 'trial_num'] = 292
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Lineups', 'replace')

# %%
# df = dm.read(f"SELECT * FROM Entry_Optimize_Results", 'Results')
# df['entry_type'] = 'millions_playaction'
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Results', 'replace')
# %%
