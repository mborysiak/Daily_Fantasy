#%%

# core packages
import pandas as pd
import numpy as np
import copy
from collections import Counter

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp

import cvxopt
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'

class FootballSimulation:

    def __init__(self, dm, week, set_year, salary_cap, pos_require_start, num_iters, 
                 pred_vers='standard', ensemble_vers='no_weight', std_dev_type='spline',
                 covar_type='team_points', full_model_rel_weight=1,
                 use_covar=True, teams=()):

        self.week = week
        self.set_year = set_year
        self.pos_require_start = pos_require_start
        self.num_iters = num_iters
        self.salary_cap = salary_cap
        self.dm = dm
        self.pred_vers = pred_vers
        self.ensemble_vers = ensemble_vers
        self.std_dev_type = std_dev_type
        self.covar_type = covar_type
        self.full_model_rel_weight = full_model_rel_weight
        self.use_covar = use_covar
        self.teams = teams

        player_data = self.get_covar_means()
        self.covar = self.pull_covar()

        # join in salary data to player data
        self.player_data = self.join_salary(player_data)

    def get_covar_means(self):
        # pull in the player data (means, team, position) and covariance matrix
        player_data = self.dm.read(f'''SELECT * 
                                       FROM Covar_Means
                                       WHERE week={self.week}
                                             AND year={self.set_year}
                                             AND pred_vers='{self.pred_vers}'
                                             AND ensemble_vers='{self.ensemble_vers}'
                                             AND std_dev_type='{self.std_dev_type}'
                                             AND covar_type='{self.covar_type}' 
                                             AND full_model_rel_weight={self.full_model_rel_weight}
                                             AND team IN {self.teams}''', 
                                             'Simulation')
        return player_data

    def pull_covar(self):
        covar = self.dm.read(f'''SELECT player, player_two, covar
                                 FROM Covar_Matrix
                                 WHERE week={self.week}
                                       AND year={self.set_year}
                                       AND pred_vers='{self.pred_vers}'
                                       AND ensemble_vers='{self.ensemble_vers}'
                                       AND std_dev_type='{self.std_dev_type}'
                                       AND covar_type='{self.covar_type}'
                                       AND full_model_rel_weight={self.full_model_rel_weight} ''', 
                                       'Simulation')
        covar = pd.pivot_table(covar, index='player', columns='player_two').reset_index()
        covar.columns = [c[1] if i!=0 else 'player' for i, c in enumerate(covar.columns)]
        return covar


    def get_model_predictions(self):
        df = self.dm.read(f'''SELECT * 
                         FROM Model_Predictions
                         WHERE week={self.week}
                               AND year={self.set_year}
                               AND version='{self.pred_vers}'
                               AND ensemble_vers='{self.ensemble_vers}'
                               AND std_dev_type='{self.std_dev_type}'
                               AND pos !='K'
                               AND pos IS NOT NULL
                               AND player!='Ryan Griffin'
                                ''', 'Simulation')
        df['weighting'] = 1
        df.loc[df.model_type=='full_model', 'weighting'] = self.full_model_rel_weight

        score_cols = ['pred_fp_per_game', 'std_dev', 'min_score', 'max_score']
        for c in score_cols: df[c] = df[c] * df.weighting

        # Groupby and aggregate with namedAgg [1]:
        df = df.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                                'std_dev': 'sum',
                                                                'weighting': 'sum',
                                                                'min_score': 'sum',
                                                                'max_score': 'sum'})

        for c in score_cols: df[c] = df[c] / df.weighting
        df.loc[df.pos=='Defense', 'pos'] = 'DEF'
        teams = self.dm.read("SELECT * FROM Player_Teams", 'Simulation')
        df = pd.merge(df, teams, on=['player'])

        drop_teams = self.get_drop_teams()
        df = df[~df.team.isin(drop_teams)].reset_index(drop=True)

        return df.drop('weighting', axis=1)



    def join_salary(self, df):

        # add salaries to the dataframe and set index to player
        salaries = self.dm .read(f'''SELECT player, pos, salary
                                     FROM Salaries
                                     WHERE year={self.set_year}
                                           AND league={self.week} ''', 'Simulation')

        df = pd.merge(df, salaries, how='left', left_on='player', right_on='player')
        df.salary = df.salary.fillna(10000)

        return df


    @staticmethod
    def bounded_multivariate_dist(means, cov_matrix, dimenions_bounds, size):

        from numpy.random import multivariate_normal

        ndims = means.shape[0]
        return_samples = np.empty([0,ndims])
        local_size = size

        # generate new samples while the needed size is not reached
        while not return_samples.shape[0] == size:
            samples = multivariate_normal(means, cov_matrix, size=size*10)

            for dim, bounds in enumerate(dimenions_bounds):
                if not np.isnan(bounds[0]): # bounds[0] is the lower bound
                    samples = samples[(samples[:,dim] > bounds[0])]  # samples[:,dim] is the column of the dim

                if not np.isnan(bounds[1]): # bounds[1] is the upper bound
                    samples = samples[(samples[:,dim] < bounds[1])]   # samples[:,dim] is the column of the dim

            return_samples = np.vstack([return_samples, samples])

            local_size = size - return_samples.shape[0]
            if local_size < 0:
                return_samples = return_samples[np.random.choice(return_samples.shape[0], size, replace=False), :]

        return return_samples


    @staticmethod
    def trunc_normal(mean_val, sdev, min_sc, max_sc, num_samples=500):

        import scipy.stats as stats

        # create truncated distribution
        lower_bound = (min_sc - mean_val) / sdev, 
        upper_bound = (max_sc - mean_val) / sdev
        trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc=mean_val, scale=sdev)
        
        estimates = trunc_dist.rvs(num_samples)

        return estimates


    def trunc_normal_dist(self, col, num_options=500):
        if col=='pred_ownership': df = self.ownership_data
        elif col=='pred_fp_per_game': df = self.player_data

        pred_list = []
        for mean_val, sdev, min_sc, max_sc in df[[col, 'std_dev', 'min_score', 'max_score']].values:
            pred_list.append(self.trunc_normal(mean_val, sdev, min_sc, max_sc, num_options))
        
        return pd.DataFrame(pred_list)


    def covar_dist(self, num_options=500):

        dists = pd.DataFrame()
        min_max = self.get_model_predictions()[['player', 'min_score', 'max_score']]

        means_sample = self.player_data.copy()
        if len(means_sample) > 0:
            covar_sample = self.covar.loc[self.covar.player.isin(means_sample.player), means_sample.player]
            min_max_sample = pd.merge(means_sample[['player']], min_max, on='player', how='left')

            mean_vals = means_sample.pred_fp_per_game.values
            covar_vals = covar_sample.values
            bound_vals = min_max_sample[['min_score', 'max_score']].values

            results = self.bounded_multivariate_dist(mean_vals, covar_vals, bound_vals, size=num_options)
            results = pd.DataFrame(results, columns=means_sample.player).T
            dists = pd.concat([dists, results], axis=0)

        dists = dists.reset_index().rename(columns={'index': 'player'})
        dists = pd.merge(self.player_data[['player']], dists, on='player').drop('player', axis=1)

        return dists.reset_index(drop=True)


    def get_predictions(self, col, ownership=False, num_options=500):

        if self.use_covar and not ownership: 
            predictions = self.covar_dist(num_options)
        else: 
            predictions = self.trunc_normal_dist(col, num_options)

        labels = self.player_data[['player', 'pos', 'team', 'salary']]
        predictions = pd.concat([labels, predictions], axis=1)

        return predictions

    def init_select_cnts(self):
        
        player_selections = {}
        for p in self.player_data.player:
            player_selections[p] = 0
        return player_selections


    def add_players(self, to_add):
        
        h_player_add = {}
        open_pos_require = copy.deepcopy(self.pos_require_start)
        df_add = self.player_data[self.player_data.player.isin(to_add)]
        for player, pos in df_add[['player', 'pos']].values:
            h_player_add[f'{player}'] = -1
            open_pos_require[pos] -= 1

        return h_player_add, open_pos_require


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

    
    def final_results(self, player_selections, success_trials):
        results = pd.DataFrame(player_selections, index=['SelectionCounts']).T
        results = results.sort_values(by='SelectionCounts', ascending=False).iloc[:29]
        salaries = self.player_data[['player', 'salary']].set_index('player')
        results = pd.merge(results, salaries, left_index=True, right_index=True)
        results = results.reset_index().rename(columns={'index': 'player'})
        results.SelectionCounts = 100*np.round(results.SelectionCounts / success_trials, 3)
        return results


    def run_sim(self, to_add, to_drop, min_players_same_team_input, set_max_team, min_players_opp_team_input=0, adjust_select=False):
        
        # can set as argument, but static set for now
        num_options=500
        player_selections = self.init_select_cnts()
        max_team_cnt = []
        success_trials = 0
        for i in range(self.num_iters):
            
            if i ==0:
                # pull out current add players and added teams
                h_player_add, open_pos_require = self.add_players(to_add)
                remaining_pos_cnt = np.sum(list(open_pos_require.values()))
                added_teams, max_added_team_cnt = self.get_current_team_cnts(to_add)

            # append the flex position to position requirements
            cur_pos_require = self.add_flex(open_pos_require)
            b_position = self.create_b_matrix(cur_pos_require)

            if i % 100 == 0:
                # get predictions and remove to drop players
                predictions = self.get_predictions(col='pred_fp_per_game', num_options=num_options)
                predictions = self.drop_players(predictions, to_drop)

            if i == 0:

                position_map = self.position_matrix_mapping(cur_pos_require)
                idx_player_map, player_idx_map = self.player_matrix_mapping(predictions)
                team_map = self.team_matrix_mapping(predictions)

                A_position = self.create_A_position(position_map, idx_player_map)

                G_salaries = self.create_G_salaries(predictions)
                h_salaries = self.create_h_salaries()
                
                G_players = self.create_G_players(player_idx_map)
                h_players = self.create_h_players(player_idx_map, h_player_add)

                G_teams = self.create_G_team(team_map, idx_player_map)
        
            # generate the c matrix with the point values to be optimized
            self.labels, self.c_points = self.sample_c_points(predictions, num_options)
            
            G = np.concatenate([G_salaries, G_players])
            h = np.concatenate([h_salaries, h_players])

            G = matrix(G, tc='d')
            h = matrix(h, tc='d')
            b = matrix(b_position, tc='d')
            A = matrix(A_position, tc='d')    
            c = matrix(self.c_points, tc='d')

            if len(to_add) < 6:
                status, x = self.solve_ilp(c, G, h, A, b)
                if status=='optimal':
                    player_selections = self.tally_player_selections(predictions, player_selections, x)
                    success_trials += 1

        results = self.final_results(player_selections, success_trials)
        if adjust_select:
            results = self.adjust_select_perc(results, open_pos_require)

        team_cnts = self.final_team_cnts(max_team_cnt)

        return results, team_cnts



#%%

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

pred_vers = 'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate'
ens_vers = 'no_weight_yes_kbest_randsample_sera10_rsq1_include2'
std_dev_type = 'pred_spline_class80_matt1_brier1_calibrate'
use_covar=True
use_ownership=False

week = 5
year = 2022
salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
num_iters = 100

sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                         ensemble_vers=ens_vers, pred_vers=pred_vers, std_dev_type=std_dev_type,
                         full_model_rel_weight=0.2, covar_type='team_points_trunc', use_covar=use_covar, 
                         teams=('BUF', 'PIT'))
min_players_same_team = 'Auto'
min_players_opp_team = 'Auto'
set_max_team = None
to_add = []
to_drop = []

results, max_team_cnt = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, 
                                    min_players_opp_team, adjust_select=False)

print(max_team_cnt)
results
# %%
