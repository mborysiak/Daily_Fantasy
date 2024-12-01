#%%
from zSim_Helper_Covar import *
import pprint
import copy
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

year=2024
week=13

salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
num_lineups = 20
set_max_team = None

pred_vers = 'sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_higherkb'

#%%

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


def clean_lineup_list(lineups_list, player_data):
    lineups = pd.DataFrame(lineups_list).T
    lineups = pd.melt(lineups)
    lineups.columns = ['TeamNum', 'player']
    lineups = pd.merge(lineups, player_data[['player', 'pos']], on=['player'])
    lineups = lineups.rename(columns={'pos':'Position', 'player': 'Player'})
    lineups = lineups.sort_values(by='TeamNum').reset_index(drop=True)
    return lineups

def create_database_output(my_team):

    my_team = my_team.sort_values(by=['Position', 'Player']).reset_index(drop=True)

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
    dk_output = dk_output.sort_values(by=['Position', 'player']).reset_index(drop=True)
    pos_map_rev = {v: k for k,v in pos_map.items()}
    dk_output.Position = dk_output.Position.map(pos_map_rev)

    dk_output_ids = dk_output[['Position', 'player_id']].T.reset_index(drop=True)
    dk_output_players = dk_output[['Position', 'player']].T.reset_index(drop=True)
    dk_output = pd.concat([dk_output_players, dk_output_ids], axis=1)

    dk_output.columns = range(dk_output.shape[1])
    dk_output = pd.DataFrame(dk_output.iloc[1,:]).T

    dk_output['year'] = year
    dk_output['week'] = week

    return dk_output



#%%

lineups_list = []
lineup_info = pd.DataFrame()
best_trials = [121, 109]
for bt in tuple(best_trials):

    opt_params = pull_best_params(bt)
    try:del opt_params['lineups_per_param']
    except:pass
    pprint.pprint(opt_params)

    best_vers = pull_params_version(bt)
    pred_vers = best_vers.pred_vers.values[0]
    reg_ens_vers = best_vers.reg_ens_vers.values[0]
    million_ens_vers = best_vers.million_ens_vers.values[0]
    std_dev_type = best_vers.std_dev_type.values[0]
    pprint.pprint(best_vers.values)


    # run all the lineups
    rs = RunSim(db_path, week, year,pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, num_lineups)
    params = rs.generate_param_list(opt_params)
    lineups_list_cur = rs.run_multiple_lineups(params, calc_winnings=False, parallelize=False, n_jobs=-1, verbose=0)
    lineups_list.extend(lineups_list_cur)

    lineup_info_cur = pd.DataFrame([i+1 for i in range(num_lineups)], columns=['lineup_num'])
    lineup_info_cur['trial_num'] = bt
    lineup_info = pd.concat([lineup_info, lineup_info_cur], axis=0).reset_index(drop=True)

# get the player data
rs_pd = RunSim(db_path, week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, 1)
player_data_param = copy.deepcopy(opt_params)
player_data_param = rs.generate_param_list(player_data_param)
sim, _ = rs.setup_sim(player_data_param[0])
player_data = sim.player_data.copy()

lineups = clean_lineup_list(lineups_list, player_data)

dk_output = pd.DataFrame()
for t in lineups.TeamNum.unique():
    if len(lineups[lineups.TeamNum==t]) == 9:
        dk_output_cur = create_database_output(lineups[lineups.TeamNum==t])
        dk_output = pd.concat([dk_output, dk_output_cur], axis=0)
    else:
        print('Incomplete Lineup')

dk_output = dk_output.reset_index(drop=True)
dk_output = pd.concat([dk_output, lineup_info], axis=1)

print('Number Duplicates', dk_output.shape[0]-dk_output.drop_duplicates(subset=[0, 1, 2, 3, 4, 5, 6, 7, 8]).shape[0])
dk_output

#%%

dm.delete_from_db('Simulation', 'Automated_Lineups', f'year={year} AND week={week}', create_backup=False)
dm.write_to_db(dk_output, 'Simulation', 'Automated_Lineups', 'append')

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

for k, v in opt_params.items():
    run_params_dict[k] = str(v)

run_params_df = pd.DataFrame(run_params_dict, index=[0])

dm_app = DataManage('c:/Users/borys/OneDrive/Documents/Github/Daily_Fantasy_App/app/')
dm_app.write_to_db(run_params_df, 'Simulation_App', 'Run_Params', 'replace')

for t in ['Predicted_Ownership', 'Mean_Ownership', 'Gambling_Lines', 'Vegas_Points', 
          'Salaries', 'Player_Teams', 'Player_Ids', 'Automated_Lineups']:

    df = dm.read(f"SELECT * FROM {t} WHERE year={year} and week={week}", 'Simulation')
    if t == 'Automated_Lineups':
        df = df[df.trial_num== best_trials[1]].reset_index(drop=True)
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

for t in ['ETR_Projections_DK']:
    df = dm.read(f'''SELECT * 
                    FROM {t}
                    WHERE week={week}
                        AND year={year}
                        ''', 'Pre_PlayerData')
    dm_app.write_to_db(df, 'Simulation_App', t, 'replace')

# %%
