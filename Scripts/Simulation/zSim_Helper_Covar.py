#%%

# core packages
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp

from ff.db_operations import DataManage   
import ff.general as ffgeneral 

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

week=13
set_year=2021
flex_pos = ['RB', 'WR', 'TE']

def join_salary(df):

    # add salaries to the dataframe and set index to player
    salaries = dm.read(f'''SELECT player, salary
                            FROM Salaries
                            WHERE year={set_year}
                                AND league={week} ''', 'Simulation')

    df = pd.merge(df, salaries, how='left', left_on='player', right_on='player')
    df.salary = df.salary.fillna(10000)

    return df


def get_top_players_from_team(df):
    df = df[df.pos!='DEF']
    df = df.sort_values(by=['team', 'pred_fp_per_game'], ascending=[True, False])
    df['player_rank'] = df.groupby('team').cumcount()
    df = df.loc[df.player_rank <= 5, ['player', 'team']].reset_index(drop=True)

    return df


def get_predictions(data, covar, num_options):

    import scipy.stats as ss
    dist = ss.multivariate_normal(mean=data.pred_fp_per_game.values, 
                                cov=covar.drop('player', axis=1).values, allow_singular=True)
    predictions = pd.DataFrame(dist.rvs(num_options)).T
    predictions = pd.concat([data[['player', 'pos', 'team', 'salary']], predictions], axis=1)

    return predictions


def _df_shuffle(df):
    '''
    Input: A dataframe to be shuffled, row-by-row indepedently.
    Return: The same dataframe whose columns have been shuffled for each row.
    '''
    # store the index before converting to numpy
    idx = df.index
    df = df.values

    # shuffle each row separately, inplace, and convert o df
    _ = [np.random.shuffle(i) for i in df]

    return pd.DataFrame(df, index=idx)


def create_flex(df, flex_pos):
        
    # filter to flex positions and create labels
    flex = df[df.pos.isin(flex_pos)]
    flex_labels = flex[['player','pos', 'team', 'salary']]

    # shuffle the data
    flex = _df_shuffle(flex.drop(['pos', 'team', 'salary', 'player'], axis=1))
    # flex.columns = [str(c) for c in flex.columns]
    
    # concat the flex labels with data and add the position
    flex = pd.concat([flex_labels, flex], axis=1)
    flex.loc[:, 'pos'] = 'FLEX'

    # concat the flex data to the full data
    df = pd.concat([df, flex], axis=0).reset_index(drop=True)

    return df

def add_players(df, to_add, pos_require):
    h_player_add = {}
    cur_loop = []

    df_add = df[df.player.isin(to_add)]
    for player, pos in df_add[['player', 'pos']].values:

        if player not in cur_loop and pos_require[pos] > 0:
            h_player_add[f'{player}_{pos}'] = -1
            pos_require[pos] -= 1
            cur_loop.append(player)
        
        else:
            df = df[~((df.player==player) & (df.pos==pos))]

    df = df.reset_index(drop=True)

    return df, h_player_add, pos_require


def drop_players(df, to_drop):
    return df[~df.player.isin(to_drop)].reset_index(drop=True)


def player_matrix_mapping(df):
    idx_player_map = {}
    player_idx_map = {}
    for i, row in df.iterrows():
        idx_player_map[i] = {
            'player': row.player,
            'team': row.team,
            'pos': row.pos,
            'salary': row.salary
        }

        player_idx_map[f'{row.player}_{row.pos}'] = i

    return idx_player_map, player_idx_map

def position_matrix_mapping(pos_require):
    position_map = {}
    i = 0
    for k, _ in pos_require.items():
        position_map[k] = i
        i+=1

    return position_map

def team_matrix_mapping(df):
    
    team_map = {}
    unique_teams = df.team.unique()
    for i, team in enumerate(unique_teams):
        team_map[team] = i

    return team_map

def create_A_position(position_map, player_map):

    num_positions = len(position_map)
    num_players = len(player_map)
    A_positions = np.zeros(shape=(num_positions, num_players))

    for i in range(num_players):
        cur_pos = player_map[i]['pos']
        row_idx = position_map[cur_pos]
        A_positions[row_idx, i] = 1

    return A_positions

def create_b_matrix(pos_require):
    return np.array(list(pos_require.values())).reshape(-1,1)

def create_G_salaries(df):
    return df.salary.values.reshape(1, len(df))

def create_h_salaries(salary_cap):
    return np.array(salary_cap).reshape(1, 1)



def get_max_team(labels, c_points):
    team_pts = pd.concat([labels, c_points], axis=1)
    team_pts = team_pts[~team_pts.pos.isin(['DEF', 'FLEX'])]
    team_pts = pd.merge(team_pts, top_team_players, on=['player', 'team'])
    team_pts = team_pts.drop(['salary', 'player', 'pos'], axis=1)
    team_pts = team_pts.groupby('team').sum()
    best_team = team_pts.idxmin().values[0]

    return best_team


def create_G_team(team_map, player_map):
    
    num_players = len(player_map)
    num_teams = len(team_map)
    G_teams = np.zeros(shape=(num_teams, num_players))

    for i in range(num_players):

        cur_team = player_map[i]['team']
        cur_pos = player_map[i]['pos']
        if cur_pos != 'DEF':
            G_teams[team_map[cur_team], i] = -1

    return G_teams


def create_h_teams(team_map, set_max_team, min_players):
    if set_max_team is None: 
        max_team = get_max_team(labels, c_points)
    else:
        max_team = set_max_team

    h_teams = np.full(shape=(len(team_map), 1), fill_value=0)
    h_teams[team_map[max_team]] = -min_players

    return h_teams

def create_G_players(player_map):

    num_players = len(player_map)
    G_players = np.zeros(shape=(num_players, num_players))
    np.fill_diagonal(G_players, -1)

    return G_players

def create_h_players(player_map, h_player_add):
    num_players = len(player_map)
    h_players = np.zeros(shape=(num_players, 1))

    for player, val in h_player_add.items():
        h_players[player_map[player]] = val

    return h_players

def sample_c_points(data, max_entries):

    labels = data[['player', 'pos', 'team', 'salary']]
    current_points = -1 * data.iloc[:, np.random.choice(range(4, max_entries+4))]

    return labels, current_points

# create empty dataframe to store player point distributions
data = dm.read('''SELECT * FROM Covar_Means''', 'Simulation')
covar = dm.read('''SELECT * FROM Covar_Matrix''', 'Simulation')
top_team_players = get_top_players_from_team(data)

min_players_same_team = 3
num_options = 500

# set league information, included position requirements, number of teams, and salary cap
pos_require = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DEF': 1}
to_add = []
to_drop = []
salary_cap = 50000
set_max_team = None

data = join_salary(data)

player_selections = {}
for p in data.player:
    player_selections[p] = 0

for i in range(500):

    if i % 100 == 0:

        predictions = get_predictions(data, covar, num_options)
        predictions = create_flex(predictions, flex_pos)

        pos_require_chk = copy.deepcopy(pos_require)
        predictions, h_player_add, open_pos_require = add_players(predictions, to_add, pos_require_chk)
        predictions = drop_players(predictions, to_drop)
        remaining_pos_cnt = np.sum(list(open_pos_require.values()))

        if i == 0:
            position_map = position_matrix_mapping(pos_require)
            idx_player_map, player_idx_map = player_matrix_mapping(predictions)
            team_map = team_matrix_mapping(predictions)

            A_position = create_A_position(position_map, idx_player_map)
            b_position = create_b_matrix(pos_require)

            G_salaries = create_G_salaries(predictions)
            h_salaries = create_h_salaries(salary_cap)

            G_players = create_G_players(player_idx_map)
            h_players = create_h_players(player_idx_map, h_player_add)

            G_teams = create_G_team(team_map, idx_player_map)
        
    # generate the c matrix with the point values to be optimized
    labels, c_points = sample_c_points(predictions, num_options)
    
    if remaining_pos_cnt > min_players_same_team:
    
        h_teams = create_h_teams(team_map, set_max_team, min_players_same_team)
        G = np.concatenate([G_salaries, G_teams, G_players])
        h = np.concatenate([h_salaries, h_teams, h_players])

    else:
        G = np.concatenate([G_salaries, G_players])
        h = np.concatenate([h_salaries, h_players])

    G = matrix(G, tc='d')
    h = matrix(h, tc='d')
    b = matrix(b_position, tc='d')
    A = matrix(A_position, tc='d')    
    c = matrix(c_points, tc='d')

    # solve the integer LP problem
    (status, x) = ilp(c, G, h, A=A, b=b, B=set(range(0, len(c_points))))

    # find all LP results chosen and equal to 1
    x = np.array(x)[:, 0]==1
    names = predictions.player.values[x]

    if len(names) != len(np.unique(names)):
        pass
    else:
        for n in names:
            player_selections[n] += 1

results = pd.DataFrame(player_selections,index=['SelectionCounts']).T
results.sort_values(by='SelectionCounts', ascending=False).iloc[:25]

#%%

df = predictions.copy()

# set league information, included position requirements, number of teams, and salary cap
pos_require = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DEF': 1}
to_add = ['Tom Brady', 'Chris Godwin', 'Mike Evans', 'Brandon Aiyuk', 'Tee Higgins']




        


# %%


# %%
