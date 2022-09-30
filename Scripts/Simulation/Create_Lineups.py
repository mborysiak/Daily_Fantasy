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
week=3

pred_vers = 'sera1_rsq0_brier2_matt1_lowsample_perc'
ensemble_vers = 'no_weight_yes_kbest_randsample_sera10_rsq1_include2'
std_dev_type = 'pred_spline_class80'

sim_type = 'ownership_ln_pos'
use_ownership=True

salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
num_iters = 200
TOTAL_LINEUPS = 4
    
min_players_same_team = 'Auto'
set_max_team = None

adjust_select = False
use_covar = False
covar_type = 'no_covar'

#%%

def rand_drop_selected(total_add, drop_multiplier):
    to_drop = []
    total_selections = dict(Counter(total_add))
    for k, v in total_selections.items():
        prob_drop = (v * drop_multiplier) / TOTAL_LINEUPS
        drop_val = np.random.uniform() * prob_drop
        if  drop_val >= 0.5:
            to_drop.append(k)
    return to_drop


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

from itertools import product

def dict_configs(d):
    for vcomb in product(*d.values()):
        yield dict(zip(d.keys(), vcomb))

G = {
    'full_model_rel_weight': [1, 5],
    'drop_player_multiple': [0], 
    'top_n_choices': [0, 2, 4],
    'drop_team_frac': [0, 0.1],
    'adjust_pos_counts': [False],
    'use_ownership': [True]
    }

params = []
for config in dict_configs(G):
    params.append(list(config.values()))

#%%
def sim_winnings(player_drop_multiplier, team_drop_frac, full_model_rel_weight, use_ownership):
    
    if covar_type=='team_points': use_covar=True
    elif covar_type=='no_covar': use_covar=False

    sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                             pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                             covar_type=covar_type, full_model_rel_weight=full_model_rel_weight, 
                             use_covar=use_covar, use_ownership=use_ownership)

    total_add = []
    to_drop_selected = []
    lineups = []
    for _ in range(10):
        
        to_add = []
        to_drop = rand_drop_teams(unique_teams, team_drop_frac)
        to_drop.extend(to_drop_selected)

        for i in range(9):
            results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, adjust_select)
            prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
            selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
            to_add.append(selected_player)
        lineups.append(to_add)

        total_add.extend(to_add)
        to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier)


    return lineups, sim.player_data

def clean_lineup_list(lineups_list):
    lineups = pd.DataFrame(lineups_list).T
    lineups = pd.melt(lineups)
    lineups.columns = ['TeamNum', 'player']
    lineups = pd.merge(lineups, player_data[['player', 'pos']], on=['player'])
    lineups = lineups.rename(columns={'pos':'Position', 'player': 'Player'})
    lineups = lineups.sort_values(by='TeamNum').reset_index(drop=True)
    return lineups

def create_database_output(my_team):

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

    dm.write_to_db(dk_output, 'Simulation', 'Automated_Lineups', 'append')

#%%

dm.delete_from_db('Simulation', 'Automated_Lineups', f'year={year} AND week={week}')

for dpm, dtf, fmrw, uo in params:
    lineups_list, player_data = sim_winnings(dpm, dtf, fmrw, uo)
    lineups = clean_lineup_list(lineups_list)

    for i in lineups.TeamNum.unique():
        create_database_output(lineups[lineups.TeamNum==i])
# %%
