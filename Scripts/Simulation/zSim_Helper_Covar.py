#%%

# core packages
from os import stat
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from collections import Counter

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp

from ff.db_operations import DataManage   
import ff.general as ffgeneral 

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

class FootballSimulation:

    def __init__(self, dm, week, set_year, salary_cap, pos_require_start, num_iters):

        self.week = week
        self.set_year = set_year
        self.pos_require_start = pos_require_start
        self.num_iters = num_iters
        self.salary_cap = salary_cap

        # pull in the player data (means, team, position) and covariance matrix
        player_data = dm.read('''SELECT * FROM Covar_Means''', 'Simulation')
        self.covar = dm.read('''SELECT * FROM Covar_Matrix''', 'Simulation')

        # extract the top players for each team
        self.top_team_players = self.get_top_players_from_team(player_data)

        # join in salary data to player data
        self.player_data = self.join_salary(player_data)

    def join_salary(self, df):

        # add salaries to the dataframe and set index to player
        salaries = dm.read(f'''SELECT player, salary
                                FROM Salaries
                                WHERE year={self.set_year}
                                    AND league={self.week} ''', 'Simulation')

        df = pd.merge(df, salaries, how='left', left_on='player', right_on='player')
        df.salary = df.salary.fillna(10000)

        return df


    @staticmethod
    def get_top_players_from_team(df, top_players=5):
                
        df = df[df.pos!='DEF']
        df = df.sort_values(by=['team', 'pred_fp_per_game'], ascending=[True, False])
        df['player_rank'] = df.groupby('team').cumcount()
        df = df.loc[df.player_rank <= top_players-1, ['player', 'team']].reset_index(drop=True)

        return df


    def get_predictions(self, num_options=500):

        import scipy.stats as ss
        dist = ss.multivariate_normal(mean=self.player_data.pred_fp_per_game.values, 
                                      cov=self.covar.drop('player', axis=1).values, 
                                      allow_singular=True)
        predictions = pd.DataFrame(dist.rvs(num_options)).T
        labels = self.player_data[['player', 'pos', 'team', 'salary']]
        predictions = pd.concat([labels, predictions], axis=1)

        return predictions

    def init_select_cnts(self):
        
        player_selections = {}
        for p in self.player_data.player:
            player_selections[p] = 0
        return player_selections


    def add_flex(self, open_pos_require):

        cur_pos_require = copy.deepcopy(self.pos_require_start)

        chk_flex = [p for p,v in open_pos_require.items() if v == -1]
        if len(chk_flex) == 0:
            flex_pos = np.random.choice(['RB', 'WR', 'TE'], p=[0.38, 0.49, 0.13])
        else:
            flex_pos = chk_flex[0]

        cur_pos_require[flex_pos] += 1

        return cur_pos_require


    def add_players(self, to_add):
        
        h_player_add = {}
        open_pos_require = copy.deepcopy(self.pos_require_start)
        df_add = self.player_data[self.player_data.player.isin(to_add)]
        for player, pos in df_add[['player', 'pos']].values:
            h_player_add[f'{player}'] = -1
            open_pos_require[pos] -= 1

        return h_player_add, open_pos_require


    def get_current_team_cnts(self, to_add):

        added_teams = self.player_data.loc[self.player_data.player.isin(to_add), 
                                           ['player', 'team']].drop_duplicates()
        added_teams = list(added_teams.team)

        if len(added_teams) > 0:
            from collections import Counter
            cnts = Counter(added_teams).most_common()
            max_added_team_cnts = cnts[0][1]
        else:
            max_added_team_cnts = 0

        return added_teams, max_added_team_cnts


    @staticmethod
    def drop_players(df, to_drop):
        return df[~df.player.isin(to_drop)].reset_index(drop=True)


    @staticmethod
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

            player_idx_map[f'{row.player}'] = i

        return idx_player_map, player_idx_map


    @staticmethod
    def position_matrix_mapping(pos_require):
        position_map = {}
        i = 0
        for k, _ in pos_require.items():
            position_map[k] = i
            i+=1

        return position_map


    @staticmethod
    def team_matrix_mapping(df):
        
        team_map = {}
        unique_teams = df.team.unique()
        for i, team in enumerate(unique_teams):
            team_map[team] = i

        return team_map

    @staticmethod
    def create_A_position(position_map, player_map):

        num_positions = len(position_map)
        num_players = len(player_map)
        A_positions = np.zeros(shape=(num_positions, num_players))

        for i in range(num_players):
            cur_pos = player_map[i]['pos']
            row_idx = position_map[cur_pos]
            A_positions[row_idx, i] = 1

        return A_positions

    @staticmethod
    def create_b_matrix(pos_require):
        return np.array(list(pos_require.values())).reshape(-1,1)

    @staticmethod
    def create_G_salaries(df):
        return df.salary.values.reshape(1, len(df))

    def create_h_salaries(self):
        return np.array(self.salary_cap).reshape(1, 1)

    @staticmethod
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


    
    def get_max_team(self, labels, c_points, added_teams):

        team_pts = pd.concat([labels, c_points], axis=1)
        team_pts = team_pts[~team_pts.pos.isin(['DEF'])]
        team_pts = pd.merge(team_pts, self.top_team_players, on=['player', 'team'])
        team_pts = team_pts.drop(['salary', 'player', 'pos'], axis=1)
        team_pts = team_pts.groupby('team').sum()

        best_teams = team_pts.apply(lambda x: pd.Series(x.nsmallest(3).index))
        best_teams = [b[0] for b in best_teams.values]
        best_teams.extend(added_teams)

        best_team = np.random.choice(best_teams)

        return best_team


    def create_h_teams(self, team_map, added_teams, set_max_team, min_players):

        if set_max_team is None: 
            max_team = self.get_max_team(self.labels, self.c_points, added_teams)
        else:
            max_team = set_max_team

        h_teams = np.full(shape=(len(team_map), 1), fill_value=0)
        h_teams[team_map[max_team]] = -min_players

        return h_teams, max_team

    @staticmethod
    def create_G_players(player_map):

        num_players = len(player_map)
        G_players = np.zeros(shape=(num_players, num_players))
        np.fill_diagonal(G_players, -1)

        return G_players

    @staticmethod
    def create_h_players(player_map, h_player_add):
        num_players = len(player_map)
        h_players = np.zeros(shape=(num_players, 1))

        for player, val in h_player_add.items():
            h_players[player_map[player]] = val

        return h_players

    @staticmethod
    def sample_c_points(data, max_entries):

        labels = data[['player', 'pos', 'team', 'salary']]
        current_points = -1 * data.iloc[:, np.random.choice(range(4, max_entries+4))]

        return labels, current_points

    @staticmethod
    def solve_ilp(c, G, h, A, b):
            
        # solve the integer LP problem
        (status, x) = ilp(c, G, h, A=A, b=b, B=set(range(0, len(c))))

        return status, x


    @staticmethod
    def tally_player_selections(predictions, player_selections, x):
            
            # find all LP results chosen and equal to 1
            x = np.array(x)[:, 0]==1
            names = predictions.player.values[x]

            # add up the player selections
            if len(names) != len(np.unique(names)):
                pass
            else:
                for n in names:
                    player_selections[n] += 1

            return player_selections

    
    def final_results(self, player_selections):
        results = pd.DataFrame(player_selections, index=['SelectionCounts']).T
        results = results.sort_values(by='SelectionCounts', ascending=False).iloc[:25]
        salaries = self.player_data[['player', 'salary']].set_index('player')
        results = pd.merge(results, salaries, left_index=True, right_index=True)
        results = results.reset_index().rename(columns={'index': 'player'})
        results.SelectionCounts = 100*np.round(results.SelectionCounts / self.num_iters, 3)
        return results

    @staticmethod
    def final_team_cnts(max_team_cnt):
        max_team_cnt = Counter(max_team_cnt)
        df = pd.DataFrame(dict(max_team_cnt), index=[0]).T.reset_index()
        df.columns=['team', 'high_score_team']
        df = df.sort_values(by='high_score_team', ascending=False).reset_index(drop=True)

        return df

    def run_sim(self, to_add, to_drop, min_players_same_team, set_max_team):
        
        # can set as argument, but static set for now
        num_options=500
        player_selections = self.init_select_cnts()
        max_team_cnt = []

        for i in range(self.num_iters):

            if i % 100 == 0:

                # pull out current add players and added teams
                h_player_add, open_pos_require = self.add_players(to_add)
                remaining_pos_cnt = np.sum(list(open_pos_require.values()))
                added_teams, max_added_team_cnt = self.get_current_team_cnts(to_add)

                # append the flex position to position requirements
                cur_pos_require = self.add_flex(open_pos_require)

                # get predictions and remove to drop players
                predictions = self.get_predictions(num_options=num_options)
                predictions = self.drop_players(predictions, to_drop)

                if i == 0:
                    position_map = self.position_matrix_mapping(cur_pos_require)
                    idx_player_map, player_idx_map = self.player_matrix_mapping(predictions)
                    team_map = self.team_matrix_mapping(predictions)

                    A_position = self.create_A_position(position_map, idx_player_map)
                    b_position = self.create_b_matrix(cur_pos_require)

                    G_salaries = self.create_G_salaries(predictions)
                    h_salaries = self.create_h_salaries()

                    G_players = self.create_G_players(player_idx_map)
                    h_players = self.create_h_players(player_idx_map, h_player_add)

                    G_teams = self.create_G_team(team_map, idx_player_map)
                
            # generate the c matrix with the point values to be optimized
            self.labels, self.c_points = self.sample_c_points(predictions, num_options)
            
            if remaining_pos_cnt > min_players_same_team and max_added_team_cnt < min_players_same_team:
            
                h_teams, max_team = self.create_h_teams(team_map, added_teams, set_max_team, min_players_same_team)
                max_team_cnt.append(max_team)
                G = np.concatenate([G_salaries, G_teams, G_players])
                h = np.concatenate([h_salaries, h_teams, h_players])

            else:
                G = np.concatenate([G_salaries, G_players])
                h = np.concatenate([h_salaries, h_players])

            G = matrix(G, tc='d')
            h = matrix(h, tc='d')
            b = matrix(b_position, tc='d')
            A = matrix(A_position, tc='d')    
            c = matrix(self.c_points, tc='d')

            if len(to_add) < 9:
                status, x = self.solve_ilp(c, G, h, A, b)
                player_selections = self.tally_player_selections(predictions, player_selections, x)
            
        results = self.final_results(player_selections)
        team_cnts = self.final_team_cnts(max_team_cnt)

        return results, team_cnts



# %%

week = 14
year = 2021
salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
num_iters = 500

sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters)


min_players_same_team = 3
set_max_team = 'KC'
to_add = ['Patrick Mahomes', 'Tyreek Hill', 'Clyde Edwards Helaire', 'Antonio Gibson',
          'CLE', 'Austin Hooper', 'Jeff Wilson', 'Brandin Cooks', 'Cole Beasley']
to_drop = ['Jared Cook']
results, max_team_cnt = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team)
results

# %%

max_team_cnt
# %%
