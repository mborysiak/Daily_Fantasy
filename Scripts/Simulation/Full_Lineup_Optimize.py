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

week = 16
year = 2021
salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
num_iters = 100


sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters)
min_players_same_team = 'Auto'
set_max_team = None

# %%
to_add = []
to_drop = []
for i in range(8):
    results, max_team_cnt = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team)
    to_add.append(results.loc[i, 'player'])
    print(to_add)

# %%
