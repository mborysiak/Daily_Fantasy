#%%
from zSim_Helper_Covar import *

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral
from joblib import Parallel, delayed
from wakepy import keep
import pprint
import gc
import psutil

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
conn = dm.db_connect('Simulation')

def get_top_hyperparams(num_rank, model_notes):

    top_settings = dm.read(f'''SELECT *
                            FROM Entry_Optimize_Hyperparams 
                            JOIN (
                                    SELECT max(date_run) date_run, model_notes
                                    FROM Entry_Optimize_Hyperparams
                                    WHERE model_notes='{model_notes}'
                                    GROUP BY model_notes
                                    ) USING (model_notes, date_run)
                            WHERE param_rank = {num_rank}''', 'SimParamsNew')
    pred_params = {}
    for k, v in top_settings.items():
        if k in ['pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers']:
            pred_params[k] = v.values[0]

    other_params = {}
    for k,v in top_settings.items():
        if k not in ['pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers', 'entry', 
                        'winnings_pred', 'ownership_vers', 'entry_type', 'param_rank', 'model_notes', 
                        'date_run', 'lineups_per_param']:
            
            if 'covar_type' in k:
                p = 'covar_type'
                opt = k.split('covar_type_')[-1]
            elif 'ownership_vers_variable' in k:
                p = 'ownership_vers_variable'
                opt = k.split('ownership_vers_variable_')[-1]
            elif 'overlap_constraint' in k:
                p = 'overlap_constraint'
                opt = k.split('overlap_constraint_')[-1]
            elif 'ownership_vers' in k:
                p = 'ownership_vers'
                opt = k.split('ownership_vers_')[-1]
            elif 'max_team_type' in k:
                p = 'max_team_type'
                opt = k.split('max_team_type_')[-1]
            else:
                p = '_'.join(k.split('_')[:-1])
                opt = k.split('_')[-1]
            
            if p not in other_params.keys(): other_params[p] = {}
            if p in ('adjust_pos_counts', 'static_top_players', 'qb_set_max_max', 'qb_solo_start', 'use_unique_players'):
                if opt=='0': opt=False
                if opt=='1': opt=True
            else:
                try: opt = int(opt)
                except: 
                    try: opt = float(opt)
                    except: pass

            other_params[p][opt] = np.round(float(v),2)

    del other_params['lineups_per_param']
        
    return pred_params, other_params

def pull_best_params(best_trial):
    params = {}
    params_opt = dm.read(f'''SELECT * 
                             FROM Entry_Optimize_Params
                             WHERE trial_num = {best_trial}''', 'ResultsNew')
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
                       WHERE trial_num = {best_trial}''', 'ResultsNew')
    return vers

entry_type = 'random_sample'
total_lineups = 20
sample_lineups = 20

trial_repeat = 121
model_notes = f'Trial {trial_repeat} Auto Flex, Sample {sample_lineups}, Total {total_lineups}'
num_rank = None
manual_adjust = True

if trial_repeat is not None:
    d = pull_best_params(trial_repeat)
    pprint.pprint(d)

    model_vers = pull_params_version(trial_repeat).drop('trial_num', axis=1).to_dict()
    model_vers = {k: v[0] for k,v in model_vers.items()}
    pprint.pprint(model_vers)

# num_rank = None
# # model_notes = 'param_test_v1'
# model_notes = 'Trial 68 Repeat'

# if num_rank is not None: 
#     model_vers, d = get_top_hyperparams(num_rank, model_notes)
#     manual_adjust = False
# else:

#     manual_adjust = True

if manual_adjust:
    model_vers = {'million_ens_vers': 'random_full_stack_newp_matt0_brier1_include2_kfold3',
                'pred_vers': 'sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_higherkb',
                'reg_ens_vers': 'random_full_stack_newp_sera0_rsq0_mse1_include2_kfold3',
                'std_dev_type': 'spline_pred_class80_q80_matt0_brier1_kfold3',
    }
    d = {'covar_type': {'kmeans_pred_trunc': 0.0,
                'kmeans_pred_trunc_new': 0.0,
                'no_covar': 0.3,
                'team_points_trunc': 0.7,
                'team_points_trunc_avgproj': 0.0},
 'full_model_rel_weight': {0.2: 0.3, 5: 0.7},
 'lineups_per_param': {1: 1.0},
 'max_overlap': {3: 0.0, 5: 0.0, 7: 0.0, 8: 1.0, 9: 0.0, 11: 0.0, 13: 0.0},
 'max_salary_remain': {500: 0.0, 1000: 1.0},
 'max_teams_lineup': {4: 0.0, 5: 0.0, 6: 0.0, 8: 1.0},
 'min_opp_team': {0: 1.0, 1: 0.0, 2: 0.0},
 'num_avg_pts': {10: 0.0, 25: 0.0, 50: 0.0, 100: 0.0, 200: 0.0, 500: 1.0},
 'num_iters': {1: 1.0},
 'num_options': {50: 0.0, 200: 0.0, 500: 0.0, 1000: 0.0, 2000: 1.0},
 'overlap_constraint': {'standard': 1.0},
 'ownership_vers': {'mil_div_standard_ln': 0.0,
                    'mil_only': 0.0,
                    'mil_times_standard_ln': 0.5,
                    'standard_ln': 0.5},
 'ownership_vers_variable': {0: 1.0, 1: 0.0},
 'pos_or_neg': {1: 1.0},
 'prev_def_wt': {1: 1.0},
 'prev_qb_wt': {1: 1.0, 2: 0.0, 3: 0.0, 5: 0.0, 7: 0.0},
 'qb_te_stack': {0: 0.4, 1: 0.6},
 'qb_wr_stack': {0: 0, 1: 1, 2: 0.0},
 'rb_flex_pct': {0.3: 0.7, 0.4: 0.3},
 'use_ownership': {0: 0.3, 1: 0.7},
 'wr_flex_pct': {0.5: 0, 0.6: 0.7, 'auto': 0.3}}

try: del d['num']
except: pass

try: del d['lineups_per_param']
except: pass

print('Num Rank:', num_rank)
print('Model Notes:', model_notes)
pprint.pprint(model_vers)
pprint.pprint(d)
for k,v in d.items():
    print(k, np.sum(list(v.values())))

#%%
#===============
# Settings and User Inputs
#===============

# set the model version
set_weeks = [
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
]

set_years = [
      2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022,
      2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023,
      2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024
]

# set_weeks=[14]
# set_years=[2022]

pred_vers = model_vers['pred_vers']
reg_ens_vers = model_vers['reg_ens_vers']
million_ens_vers = model_vers['million_ens_vers']
std_dev_type = model_vers['std_dev_type']


salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
adjust_winnings = False
set_max_team = None

rs = RunSim(db_path, 1, 2022, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups)

max_trial_num = dm.read("SELECT max(trial_num) FROM Entry_Optimize_Params_Detail", 'ResultsNew').values[0][0]
trial_num = max_trial_num + 1


def run_weekly_sim(d, week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups):
    try:
        rs = RunSim(db_path, week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups)
        params = rs.generate_param_list(d)
        winnings, player_results, winnings_list = rs.run_multiple_lineups(params, calc_winnings=True, parallelize=False)
        return winnings, player_results, params, winnings_list

    finally:
        # Clean up large objects
        del rs, params
        gc.collect()


def random_sample_winnings(w, pl, par, wl, total_lineups, sample_lineups):
    
    true_mask = sample_lineups*[True]
    false_mask = (total_lineups-sample_lineups)*[False]
    mask = true_mask + false_mask
    np.random.shuffle(mask)

    wl = list(np.array(wl)[mask])

    lineups_num = pl.lineup_num.unique()[mask]
    pl = pl[pl.lineup_num.isin(lineups_num)]

    par = np.array(par)[mask]
    par = list([list(p) for p in par])

    w = np.sum(wl)
    
    return w, pl, par, wl

def objective(param_options, pred_vers, reg_ens_vers, std_dev_type, million_ens_vers, total_lineups, sample_lineups, set_weeks, 
              set_years):

    try: 

        output = Parallel(n_jobs=12, verbose=0, max_nbytes='0.5G')(
                                    delayed(run_weekly_sim)(param_options, week, year, pred_vers, reg_ens_vers, 
                                                            million_ens_vers, std_dev_type, total_lineups) for
                                    week, year in zip(set_weeks, set_years)
                                    )
        total_winnings = []
        player_results = pd.DataFrame()
        param_output = []
        winnings_list = []
        for w, pl, par, wl in output:
            if entry_type == 'random_sample': 
                w, pl, par, wl = random_sample_winnings(w, pl, par, wl, total_lineups, sample_lineups)
            total_winnings.append(w)
            player_results = pd.concat([player_results, pl])
            param_output.append(par)
            winnings_list.append(wl)

        total_winnings = list(np.round(np.array(total_winnings) * 13 / sample_lineups,1))

        print('Total Winnings:', np.sum(total_winnings))
        win_dict = {f'{week}_{year}': t for week, year ,t in zip(set_weeks, set_years, total_winnings)}
        print_df = pd.DataFrame(win_dict, index=[0]).T.reset_index()
        print_df.columns = ['date', 'winnings']
        print(print_df)

        return total_winnings, player_results, param_output, winnings_list
    
    finally:
        gc.collect()


def format_output_results(weekly_winnings, set_weeks, set_years, pred_vers, reg_ens_vers, std_dev_type, 
                       million_ens_vers, trial_num, repeat_num, entry_type, num_rank, model_notes, manual_adjust):
    output_results = []
    for week_win_amt, week, year in zip(weekly_winnings, set_weeks, set_years):
        output_results.append([week, year, pred_vers, reg_ens_vers, std_dev_type, million_ens_vers, week_win_amt])

    # save out the high level results of the overall week
    output_results = pd.DataFrame(output_results, columns=['week', 'year', 'pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers', 'avg_winnings'])
    output_results['ownership_vers'] = 'variable'
    output_results['trial_num'] = trial_num
    output_results['repeat_num'] = repeat_num
    output_results['entry_type'] = entry_type
    output_results['num_rank'] = num_rank
    output_results['model_notes'] = model_notes
    output_results['manual_adjust'] = manual_adjust

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

        weekly_winnings, player_results, param_output, winning_list = objective(d, pred_vers, reg_ens_vers, std_dev_type, million_ens_vers, 
                                                                                total_lineups, sample_lineups, set_weeks, set_years)

        output_results = format_output_results(weekly_winnings, set_weeks, set_years, pred_vers, reg_ens_vers, std_dev_type,
                                               million_ens_vers, trial_num, repeat_num, entry_type, num_rank, model_notes, manual_adjust)
        dm.write_to_db(output_results, 'ResultsNew', 'Entry_Optimize_Results', 'append')

        player_results_out = format_player_results(player_results, total_lineups, repeat_num, trial_num)
        dm.write_to_db(player_results_out, 'ResultsNew', 'Entry_Optimize_Lineups', 'append')

        param_output_df = detailed_param_output(rs, param_output, winning_list, set_weeks, set_years, trial_num, repeat_num)
        param_output_df.ownership_vers = param_output_df.ownership_vers.astype('str')
        dm.write_to_db(param_output_df, 'ResultsNew', 'Entry_Optimize_Params_Detail', 'append')

        del weekly_winnings, player_results, param_output, winning_list, output_results, player_results_out, param_output_df
        gc.collect()

    # save out the initial params that were set for randomization
    output = param_set_output(d)
    dm.write_to_db(output, 'ResultsNew', 'Entry_Optimize_Params', 'append')


#%%
prev_qb_wt = 1
prev_def_wt = 1
max_overlap = 8

constraint_div = float(max_overlap+int(prev_qb_wt/2)+(prev_def_wt/2))
constraint_minus = float(max_overlap+(prev_qb_wt-1)+(prev_def_wt-1))
constraint_plus = float(max_overlap+(prev_qb_wt)+(prev_def_wt))
constraint_standard = float(max_overlap)

print('Divide By')
print('if qb, other players:', constraint_div-prev_qb_wt)
print('if not qb, other players:', constraint_div)

print('Minus One')
print('if qb, other players:', constraint_minus-prev_qb_wt)
print('if not qb, other players:', constraint_minus)

print('Plus Weights')
print('if qb, other players:', constraint_plus-prev_qb_wt)
print('if not qb, other players:', constraint_plus)

print('Standard')
print('if qb, other players:', constraint_standard)
print('if not qb, other players:', constraint_standard)

#%%

to_delete_num=130

dm.delete_from_db('ResultsNew', 'Entry_Optimize_Results', f'trial_num={to_delete_num}', create_backup=False)
dm.delete_from_db('ResultsNew', 'Entry_Optimize_Lineups', f'trial_num={to_delete_num}', create_backup=False)
dm.delete_from_db('ResultsNew', 'Entry_Optimize_Params', f'trial_num={to_delete_num}', create_backup=False)
dm.delete_from_db('ResultsNew', 'Entry_Optimize_Params_Detail', f'trial_num={to_delete_num}', create_backup=False)

#%%

df = dm.read(f"SELECT * FROM Entry_Optimize_Params", 'ResultsNew')
add_on = pd.DataFrame({'trial_num': range(df.trial_num.max()+1)})
add_on = add_on.assign(param='overlap_constraint', param_option='standard', option_value=1)
# add_on = add_on.assign(param='rb_flex_pct', param_option=0.4, option_value=1)
# add_on = add_on.assign(param='wr_flex_pct', param_option=0.5, option_value=1)
# add_on = add_on.assign(param='rb_min_sal', param_option=3000, option_value=1)

add_on = add_on[df.columns]

df = pd.concat([df, add_on], axis=0)
df = df.sort_values(by='trial_num')

# df.loc[(df.trial_num.isin([57, 58, 59])) & (df.param=='overlap_constraint'), ['param_option', 'option_value']] = ['plus_wts', 1]
# df.loc[(df.trial_num>59) & (df.param=='overlap_constraint'), ['param_option', 'option_value']] = ['minus_one', 1]
# dm.write_to_db(df, 'ResultsNew', 'Entry_Optimize_Params', 'replace', create_backup=True)

#%%


df = dm.read(f"SELECT * FROM Entry_Optimize_Params_Detail", 'ResultsNew')
df['use_ownership'] = 1
df['overlap_constraint'] = 'standard'
# df['wr_flex_pct'] = 0.5
# df['rb_min_sal'] = 3000

df.loc[(df.trial_num.isin([57, 58, 59])), 'overlap_constraint'] = 'plus_wts'
df.loc[(df.trial_num>59), 'overlap_constraint'] = 'minus_one'
df
# dm.write_to_db(df, 'ResultsNew', 'Entry_Optimize_Params_Detail', 'replace')

#%%


# num_rank = 5
# model_notes = 'all_weeks_non8_times_3'

# df = dm.read(f"SELECT * FROM Entry_Optimize_Results", 'Results')
# df = df.assign(num_rank=None, model_notes=None, manual_adjust=None)
# df.loc[df.trial_num==553, ['num_rank', 'model_notes', 'manual_adjust']] = [5, 'all_weeks_non8_times_3', False]
# df.loc[df.trial_num==552, ['num_rank', 'model_notes', 'manual_adjust']] = [5114, 'all_weeks_non8_times_3', False]
# df.loc[df.trial_num==551, ['num_rank', 'model_notes', 'manual_adjust']] = [1159, 'all_weeks_non8_times_3', False]
# df.loc[df.trial_num==550, ['num_rank', 'model_notes', 'manual_adjust']] = [509, 'all_weeks_non8_times_3', False]
# df.loc[df.trial_num==549, ['num_rank', 'model_notes', 'manual_adjust']] = [677, 'all_weeks_non8_times_3', False]
# df.loc[df.trial_num==548, ['num_rank', 'model_notes', 'manual_adjust']] = [0, 'all_weeks_non8_times_3', False]
# df.loc[df.trial_num==546, ['num_rank', 'model_notes', 'manual_adjust']] = [12885, 'all_weeks_non8_times_3', False]
# df.loc[df.trial_num==545, ['num_rank', 'model_notes', 'manual_adjust']] = [12647, 'all_weeks_non8_times_3', False]
# df.loc[df.trial_num==544, ['num_rank', 'model_notes', 'manual_adjust']] = [161, 'all_weeks_non8_times_3', False]
# df.loc[df.trial_num==543, ['num_rank', 'model_notes', 'manual_adjust']] = [154, 'all_weeks_non8_times_3', False]

# dm.write_to_db(df, 'Results', 'Entry_Optimize_Results', 'replace')


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
