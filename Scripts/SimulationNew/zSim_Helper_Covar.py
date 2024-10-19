#%%
from itertools import combinations
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
from cvxopt.glpk import ilp
from scipy.stats import beta
import random
import sqlite3

class FullLineupSim:
    def __init__(self, week, year, conn, pred_vers, reg_ens_vers, million_ens_vers,
                 std_dev_type, full_model_rel_weight, use_covar, covar_type, print_results):

        self.week = week
        self.set_year = year
        self.pred_vers = pred_vers
        self.reg_ens_vers = reg_ens_vers
        self.million_ens_vers = million_ens_vers
        self.std_dev_type = std_dev_type
        self.conn = conn
        self.full_model_rel_weight = full_model_rel_weight
        self.use_covar = use_covar
        self.salary_cap = 50000
        self.covar_type = covar_type
        self.print_results = print_results
        
        
        if self.use_covar: 
            player_data = self.get_covar_means()
            self.covar = self.pull_covar()
        else:
            player_data = self.get_model_predictions()

        self.player_data = self.join_salary(player_data)
        self.player_data = self.join_opp(self.player_data)
        self.opponents = self.get_matchups()


    def get_model_predictions(self):
        df = pd.read_sql_query(f'''SELECT * 
                                    FROM Model_Predictions
                                    WHERE week={self.week}
                                        AND year={self.set_year}
                                        AND pred_vers='{self.pred_vers}'
                                        AND reg_ens_vers='{self.reg_ens_vers}'
                                        AND std_dev_type='{self.std_dev_type}'
                                        AND pos !='K'
                                        AND pos IS NOT NULL
                                        AND player!='Ryan Griffin'
                    
                                ''', self.conn)
        df['weighting'] = 1
        df.loc[df.model_type=='full_model', 'weighting'] = self.full_model_rel_weight

        score_cols = ['pred_fp_per_game', 'std_dev', 'min_score', 'max_score']
        for c in score_cols: 
            df[c] = df[c] * df.weighting

        # Groupby and aggregate with namedAgg [1]:
        df = df.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                                'std_dev': 'sum',
                                                                'weighting': 'sum',
                                                                'min_score': 'sum',
                                                                'max_score': 'sum'})

        for c in score_cols: df[c] = df[c] / df.weighting
        df.loc[df.pos=='Defense', 'pos'] = 'DEF'
        teams = pd.read_sql_query(f'''SELECT player, team 
                                        FROM Player_Teams 
                                        WHERE week={self.week} 
                                            AND year={self.set_year}
                                    ''', 
                                    self.conn)
        df = pd.merge(df, teams, on=['player'])

        drop_teams = self.get_drop_teams()
        df = df[~df.team.isin(drop_teams)].reset_index(drop=True)

        return df.drop('weighting', axis=1)
    
    
            
    def get_covar_means(self):
        # pull in the player data (means, team, position) and covariance matrix
        player_data = pd.read_sql_query(f'''SELECT * 
                                            FROM Covar_Means
                                            WHERE week={self.week}
                                                    AND year={self.set_year}
                                                    AND pred_vers='{self.pred_vers}'
                                                    AND reg_ens_vers='{self.reg_ens_vers}'
                                                    AND std_dev_type='{self.std_dev_type}'
                                                    AND covar_type='{self.covar_type}' 
                                                    AND full_model_rel_weight={self.full_model_rel_weight}''', 
                                             self.conn)
        return player_data

    def pull_covar(self):
        covar = pd.read_sql_query(f'''SELECT player, player_two, covar
                                      FROM Covar_Matrix
                                      WHERE week={self.week}
                                            AND year={self.set_year}
                                            AND pred_vers='{self.pred_vers}'
                                            AND reg_ens_vers='{self.reg_ens_vers}'
                                            AND std_dev_type='{self.std_dev_type}'
                                            AND covar_type='{self.covar_type}'
                                            AND full_model_rel_weight={self.full_model_rel_weight} ''', 
                                       self.conn)
        covar = pd.pivot_table(covar, index='player', columns='player_two').reset_index().fillna(0)
        covar.columns = [c[1] if i!=0 else 'player' for i, c in enumerate(covar.columns)]
        return covar
    

    def join_salary(self, df):

        # add salaries to the dataframe and set index to player
        salaries = pd.read_sql_query(f'''SELECT player, salary
                                        FROM Salaries
                                        WHERE year={self.set_year}
                                            AND week={self.week} ''', 
                                        self.conn)

        df = pd.merge(df, salaries, how='left', left_on='player', right_on='player')
        df.salary = df.salary.fillna(10000)

        return df
    
    def join_opp(self, df):
        matchups = pd.DataFrame(self.get_matchups(), index=[0]).T.reset_index()
        matchups.columns = ['team', 'opp']
        df = pd.merge(df, matchups, on='team')
        return df

    def get_drop_teams(self):

        df = pd.read_sql_query(f'''SELECT away_team, home_team, gametime 
                                    FROM Gambling_Lines 
                                    WHERE week={self.week} 
                                            and year={self.set_year} 
                    ''', self.conn)
        df.gametime = pd.to_datetime(df.gametime)
        df['day_of_week'] = df.gametime.apply(lambda x: x.weekday())
        df['hour_in_day'] = df.gametime.apply(lambda x: x.hour)
        df = df[(df.day_of_week!=6) | (df.hour_in_day > 16) | (df.hour_in_day < 11)]
        drop_teams = list(df.away_team.values)
        drop_teams.extend(list(df.home_team.values))

        return drop_teams
    
    def get_matchups(self):
        df = pd.read_sql_query(f'''SELECT away_team, home_team
                                   FROM Gambling_Lines 
                                   WHERE week={self.week} 
                                         and year={self.set_year} 
                    ''', self.conn)

        df = df[(df.away_team.isin(self.player_data.team.unique())) & \
                (df.home_team.isin(self.player_data.team.unique()))]

        matchups = {}
        for away, home in df.values:
            matchups[away] = home
            matchups[home] = away

        return matchups
    
    def join_ownership_pred(self, df):

        # add salaries to the dataframe and set index to player
        ownership = pd.read_sql_query(f'''SELECT player, pred_ownership, std_dev+0.01 as std_dev, min_score, max_score
                                          FROM Predicted_Ownership
                                          WHERE year={self.set_year}
                                                AND week={self.week}
                                                AND ownership_vers='{self.ownership_vers}'
                                                AND pred_vers='{self.pred_vers}'
                                                AND million_ens_vers='{self.million_ens_vers}' ''', 
                                        self.conn)

        if self.use_covar: df = df.drop(['pred_fp_per_game'], axis=1)
        else: df = df.drop(['pred_fp_per_game', 'std_dev', 'min_score','max_score'], axis=1)

        df = pd.merge(df, ownership, how='left', on='player')

        # df['min_score'] = df.pred_ownership.min()-0.01

        df.pred_ownership = df.pred_ownership.fillna(df.pred_ownership.mean())
        df.std_dev = df.std_dev.fillna(df.std_dev.mean())
        df.min_score = df.min_score.fillna(df.min_score.min())

        df.loc[df.max_score < 0.05, 'max_score'] = 0.05
        df.loc[df.pred_ownership < df.min_score, 'pred_ownership'] = df.loc[df.pred_ownership < df.min_score, 'max_score'] / 3
        df.loc[df.max_score.isnull(), 'max_score'] = abs(df.loc[df.max_score.isnull(), 'pred_ownership']) * 2

        return df

    
    def get_predictions(self, col, ownership=False, num_options=500):

        labels = self.player_data[['player', 'pos', 'team', 'salary']]

        if self.use_covar and not ownership: 
            predictions = self.covar_dist(num_options)
            predictions = pd.concat([labels, predictions], axis=1)

        else: 
            predictions = self.trunc_normal_dist(col, num_options)
            predictions = pd.concat([labels, predictions], axis=1)

        return predictions
    
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
        elif col=='vegas_points': df = self.vegas_points

        pred_list = []
        for mean_val, sdev, min_sc, max_sc in df[[col, 'std_dev', 'min_score', 'max_score']].values:
            pred_list.append(self.trunc_normal(mean_val, sdev, min_sc, max_sc, num_options))

        return pd.DataFrame(pred_list)
    
    
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
    def unique_matchup_pairs(matchups):
        import itertools
        sorted_matchups = [sorted(m) for m in matchups.items()]
        return list(m for m, _ in itertools.groupby(sorted_matchups))
    
    
    def covar_dist(self, num_options=500):

        unique_matchups = self.unique_matchup_pairs(self.opponents)
        min_max = self.get_model_predictions()[['player', 'min_score', 'max_score']]
        
        dists = pd.DataFrame()
        for matchup in unique_matchups:

            means_sample = self.player_data[self.player_data.team.isin(matchup)].reset_index(drop=True)
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

    
    def create_h_ownership(self):
        mean_own, std_own = pd.read_sql_query(f'''SELECT ownership_mean, ownership_std
                                                  FROM Mean_Ownership
                                                  WHERE week={self.week}
                                                        AND year={self.set_year}
                                                        AND ownership_vers='{self.ownership_vers}'
                                                        AND pred_vers='{self.pred_vers}'
                                                        AND million_ens_vers='{self.million_ens_vers}'
                                        ''', self.conn).values[0]

        mean_own = self.pos_or_neg * np.random.normal(mean_own, std_own, size=1).reshape(1, 1)
        return mean_own
    


    def run_sim(self, conn, to_add, to_drop, num_iters, ownership_vers, num_options, num_avg_pts, pos_or_neg,
                qb_wr_stack, qb_te_stack, min_opp_team, max_teams_lineup,max_salary_remain,
                max_overlap, prev_qb_wt, previous_lineups):
        
        self.conn = conn
        self.prepare_data(ownership_vers, num_options, pos_or_neg)
        
        player_counts = {player: 0 for player in self.player_data['player']}
        successful_iterations = 0
        
        for _ in range(num_iters):

            iters_run = 0
            success = False
            
            while not success and iters_run < 5:

                self.set_position_counts()
                iters_run += 1
                cur_pred_fps = self.sample_points(num_options, num_avg_pts)
                cur_ownership = self.sample_ownership(num_options, num_avg_pts)
                c, A, b, G, h = self.setup_optimization_problem(to_add, to_drop, cur_pred_fps, cur_ownership, qb_wr_stack, qb_te_stack, 
                                                                min_opp_team, max_teams_lineup, max_salary_remain,
                                                                max_overlap, prev_qb_wt, previous_lineups)
                
                status, x = self.solve_optimization(c, G, h, A, b)
                
                if status != 'optimal':
                    print(f"Optimization failed in iteration {_ + 1}. Status: {status}")
                    continue

                selected_players = self.process_results(x)
                
                for player in selected_players:
                    player_counts[player] += 1
                
                successful_iterations += 1
                success = True

        if successful_iterations == 0:
            print("All optimization attempts failed.")
            return None, None
        
        player_percentages = self.calculate_player_perc(player_counts, successful_iterations)
        
        return selected_players, player_percentages
    
    def sample_points(self, num_options, num_avg_pts):
        current_points = self.pred_fps.iloc[:, np.random.choice(range(4, num_options+4), size=num_avg_pts)].mean(axis=1)
        return current_points
    
    def sample_ownership(self, num_options, num_avg_pts):
        current_ownership = self.ownerships.iloc[:, np.random.choice(range(4, num_options+4), size=num_avg_pts)].mean(axis=1)
        return current_ownership
        
    def prepare_data(self, ownership_vers, num_options, pos_or_neg):
        
        self.df = self.player_data.copy()
        self.pos_or_neg = pos_or_neg
        self.ownership_vers = ownership_vers
        self.ownership_data = self.join_ownership_pred(self.df)
        self.min_ownership = self.create_h_ownership()[0][0]

        self.players = self.player_data['player'].values
        self.salaries = self.player_data['salary'].values
        self.positions = self.player_data['pos'].values
        self.teams = self.player_data['team'].values
        self.n_players = len(self.players)
        self.unique_teams = list(set(self.teams))
        self.n_teams = len(self.unique_teams)

        self.pred_fps = self.get_predictions('pred_fp_per_game', num_options=num_options+1)   

        self.ownerships = self.get_predictions('pred_ownership', ownership=True, num_options=num_options+1)

    def set_position_counts(self):
        flex_pos = np.random.choice(['RB', 'WR', 'TE'], p=[0.37, 0.51, 0.12])
        self.position_counts = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
        self.position_counts[flex_pos] += 1
        self.num_pos = float(np.sum(list(self.position_counts.values())))

    def setup_optimization_problem(self, to_add, to_drop, cur_pred_fps, cur_ownership, qb_wr_stack, qb_te_stack, min_opp_team, 
                                   max_teams_lineup, max_salary_remain, max_overlap, prev_qb_wt, previous_lineups):
        
        n_variables = self.n_players + len(self.unique_teams)
        
        c = self.create_objective_function(cur_pred_fps)
        A, b = self.create_equality_constraints()
        G, h = self.create_inequality_constraints(to_add, to_drop, cur_ownership, n_variables, qb_wr_stack, qb_te_stack, 
                                                  min_opp_team, max_teams_lineup, max_salary_remain,
                                                  max_overlap, prev_qb_wt, previous_lineups)
        return c, A, b, G, h

    def create_objective_function(self, cur_pred_fps):
        return matrix(list(-cur_pred_fps) + [0] * len(self.unique_teams), tc='d')

    def create_equality_constraints(self):
        A = []
        b = []

        # Constraint: Exact number of players
        A.append([1.0] * self.n_players + [0.0] * len(self.unique_teams))
        b.append(self.num_pos)

        # Constraints for exact number of players per position
        for pos, count in self.position_counts.items():
            constraint = [1.0 if p == pos else 0.0 for p in self.positions] + [0.0] * len(self.unique_teams)
            A.append(constraint)
            b.append(float(count))

        return matrix(A, tc='d'), matrix(b, tc='d')

    def create_inequality_constraints(self, to_add, to_drop, cur_ownership, n_variables, qb_wr_stack, qb_te_stack, 
                                      min_opp_team, max_teams_lineup, max_salary_remain, max_overlap, prev_qb_wt,
                                      previous_lineups):
        G = []
        h = []

        self.add_salary_constraint(G, h, max_salary_remain)
        self.add_forced_players_constraint(G, h, n_variables, to_add)
        self.add_excluded_players_constraint(G, h, n_variables, to_drop)
        self.add_qb_dst_constraint(G, h, n_variables)

        if len(to_add) <= 7:
            self.add_ownership_constraint(G, h, cur_ownership)
            self.add_stacking_constraints(G, h, n_variables, qb_wr_stack, qb_te_stack)
            self.add_opposing_team_constraint(G, h, n_variables, min_opp_team)
            self.add_max_teams_constraint(G, h, n_variables, max_teams_lineup)
            self.add_overlap_constraint(G, h, n_variables, max_overlap, previous_lineups, prev_qb_wt)
            
        return matrix(np.array(G), tc='d'), matrix(h, tc='d')
    

    def add_salary_constraint(self, G, h, max_salary_remain):
        G.append(list(self.salaries) + [0] * len(self.unique_teams))
        h.append(float(self.salary_cap))

        # New constraint: Total salary >= SALARY_CAP - max_salary_remain
        G.append([-s for s in self.salaries] + [0] * len(self.unique_teams))
        h.append(-float(self.salary_cap - max_salary_remain))

    def add_forced_players_constraint(self, G, h, n_variables, to_add):
        for player in to_add:
            if player in self.players:
                index = list(self.players).index(player)
                constraint = [0] * n_variables
                constraint[index] = -1  # Force this player to be selected
                G.append(constraint)
                h.append(-1.0)  # Must be less than or equal to -1, forcing selection

    
    def add_excluded_players_constraint(self, G, h, n_variables, to_drop):
        for player in to_drop:
            if player in self.players:
                index = list(self.players).index(player)
                constraint = [0] * n_variables
                constraint[index] = 1  # Prevent this player from being selected
                G.append(constraint)
                h.append(0.0)  # Must be less than or equal to 0, forcing non-selection

    def add_stacking_constraints(self, G, h, n_variables, qb_wr_stack, qb_te_stack):
        for team in self.unique_teams:
            self.add_qb_wr_stack(G, h, n_variables, team, qb_wr_stack)
            self.add_qb_te_stack(G, h, n_variables, team, qb_te_stack)

    def add_qb_wr_stack(self, G, h, n_variables, team, qb_wr_stack):
        qb_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'QB' and t == team]
        wr_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'WR' and t == team]
        
        if qb_indices and wr_indices:
            constraint = [0] * n_variables
            for qb_index in qb_indices:
                constraint[qb_index] = qb_wr_stack
            for wr_index in wr_indices:
                constraint[wr_index] = -1
            G.append(constraint)
            h.append(0.0)

    def add_qb_te_stack(self, G, h, n_variables, team, qb_te_stack):
        qb_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'QB' and t == team]
        te_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'TE' and t == team]
        
        if qb_indices and te_indices:
            constraint = [0] * n_variables
            for qb_index in qb_indices:
                constraint[qb_index] = qb_te_stack
            for te_index in te_indices:
                constraint[te_index] = -1
            G.append(constraint)
            h.append(0.0)

    def add_qb_dst_constraint(self, G, h, n_variables):
        for team in self.unique_teams:
            qb_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'QB' and t == team]
            opposing_team = self.opponents[team]
            opposing_dst_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'DEF' and t == opposing_team]
            
            if qb_indices and opposing_dst_indices:
                constraint = [0] * n_variables
                for qb_index in qb_indices:
                    constraint[qb_index] = 1
                for dst_index in opposing_dst_indices:
                    constraint[dst_index] = 1
                G.append(constraint)
                h.append(1.0)

    def add_opposing_team_constraint(self, G, h, n_variables, min_opp_team):
        for team in self.unique_teams:
            qb_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if pos == 'QB' and t == team]
            opposing_team = self.opponents[team]
            opposing_player_indices = [i for i, (pos, t) in enumerate(zip(self.positions, self.teams)) if t == opposing_team and pos != 'DEF']
            
            if qb_indices and opposing_player_indices:
                constraint = [0] * n_variables
                for qb_index in qb_indices:
                    constraint[qb_index] = min_opp_team
                for opp_index in opposing_player_indices:
                    constraint[opp_index] = -1
                G.append(constraint)
                h.append(0.0)

    def add_max_teams_constraint(self, G, h, n_variables, max_teams_lineup):
        for i, team in enumerate(self.unique_teams):
            team_indices = [i for i, t in enumerate(self.teams) if t == team]
            
            constraint = [0] * n_variables
            for idx in team_indices:
                constraint[idx] = -1
            constraint[self.n_players + i] = 1
            G.append(constraint)
            h.append(0.0)

            constraint = [0] * n_variables
            for idx in team_indices:
                constraint[idx] = 1
            constraint[self.n_players + i] = -6
            G.append(constraint)
            h.append(0.0)

        G.append([0] * self.n_players + [1] * len(self.unique_teams))
        h.append(float(max_teams_lineup))

    def add_overlap_constraint(self, G, h, n_variables, max_overlap, previous_lineups, prev_qb_wt):
        for prev_lineup in previous_lineups:
            constraint = [0] * n_variables
            for i, (player, position) in enumerate(zip(self.players, self.positions)):
                if player in prev_lineup:
                    if position == 'QB':
                        constraint[i] = prev_qb_wt
                    else:
                        constraint[i] = 1
            G.append(constraint)
            h.append(float(max_overlap))

    def add_ownership_constraint(self, G, h, cur_ownership):
        G.append([-o for o in cur_ownership] + [0] * len(self.unique_teams))
        h.append(-self.min_ownership)

    def solve_optimization(self, c, G, h, A, b):
        return ilp(c, G, h, A.T, b, B=set(range(len(c))))

    def process_results(self, x):
        selected_indices = [i for i in range(self.n_players) if x[i] > 0.5]
        selected_players = [self.players[i] for i in selected_indices]
        selected_salaries = [self.salaries[i] for i in selected_indices]
        selected_teams = [self.teams[i] for i in selected_indices]
        selected_positions = [self.positions[i] for i in selected_indices]

        if self.print_results:
            self.print_out_results(selected_players, selected_salaries, 
                                   selected_teams, selected_positions)

        return selected_players
    
    def calculate_player_perc(self, player_counts, successful_iterations):
        # Convert counts to percentages
        player_percentages = {player: np.round((count / successful_iterations) * 100,1)
                            for player, count in player_counts.items() if count > 0}
        
        player_percentages = pd.DataFrame(player_percentages, index=[0]).T.reset_index()
        player_percentages.columns = ['player', 'SelectionCounts']
        player_percentages = pd.merge(player_percentages, self.player_data[['player', 'salary']], on='player')
        player_percentages = player_percentages.sort_values(by='SelectionCounts', ascending=False).reset_index(drop=True)
        return player_percentages

    def print_out_results(self, selected_players, selected_salaries,
                      selected_teams, selected_positions):
        print("Selected lineup:")
        print(selected_players)
        for player, salary, team, pos in zip(selected_players, selected_salaries, 
                                                            selected_teams, selected_positions):
            print(f"{player} ({team}, {pos}): Salary ${salary}")
        print(f"Total Salary: ${sum(selected_salaries)}")

        self.print_stacks(selected_players, selected_teams, selected_positions)
        self.print_constraint_verification(selected_players, selected_salaries, selected_positions, selected_teams)

    def print_stacks(self, selected_players, selected_teams, selected_positions):
        for team in set(selected_teams):
            qb = next((player for player, t, pos in zip(selected_players, selected_teams, selected_positions) if t == team and pos == 'QB'), None)
            wrs = [player for player, t, pos in zip(selected_players, selected_teams, selected_positions) if t == team and pos == 'WR']
            tes = [player for player, t, pos in zip(selected_players, selected_teams, selected_positions) if t == team and pos == 'TE']

            if qb and wrs:
                print(f"\nQB-WR Stack for {team}:")
                print(f"QB: {qb}")
                print(f"WRs: {', '.join(wrs)}")

            if qb and tes:
                print(f"\nQB-TE Stack for {team}:")
                print(f"QB: {qb}")
                print(f"TEs: {', '.join(tes)}")

    def print_constraint_verification(self, selected_players, selected_salaries, selected_positions, selected_teams):
        print("\nConstraint Verification:")
        print(f"Total players: {len(selected_players)} (should be {self.num_pos})")
        for pos, count in self.position_counts.items():
            print(f"{pos}s: {sum(1 for p in selected_positions if p == pos)} (should be {count})")
        print(f"Total Salary: ${sum(selected_salaries)} (should be <= {self.salary_cap})")
            
        # Display stack information
        selected_teams_set = set(selected_teams)
        print(f"\nNumber of teams in lineup: {len(selected_teams_set)} (should be exactly {max_teams_lineup})")
        print(f"Teams in lineup: {', '.join(selected_teams_set)}")
        for team in selected_teams_set:
            team_players = [player for player, t in zip(selected_players, selected_teams) if t == team]
            print(f"{team}: {len(team_players)} players")


# week = 6
# year = 2024
# pred_vers = 'sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_higherkb'
# reg_ens_vers = 'random_full_stack_newp_sera0_rsq0_mse1_include2_kfold3'
# std_dev_type = 'spline_class80_q80_matt0_brier1_kfold3'
# million_ens_vers = 'random_full_stack_newp_matt0_brier1_include2_kfold3'
# full_model_rel_weight = 0.2
# use_covar = False
# covar_type = 'no_covar'
# print_results = True

# conn = sqlite3.connect("c:/Users/mborysia/OneDrive - Starbucks/Documents/Github/Simulation/Simulation.sqlite3")
# sim = FullLineupSim(week, year, conn, pred_vers, reg_ens_vers, million_ens_vers,
#                     std_dev_type, full_model_rel_weight, use_covar, covar_type, print_results)

# previous_lineups = [
#  ['Darnell Mooney',
#   'Drake London',
#   'IND',
#   'Jalen Tolbert',
#   'James Conner',
#   'Josh Jacobs',
#   'Kirk Cousins',
#   'Kyle Pitts',
#   'Najee Harris'],

# ]

# # Configurable parameters
# ownership_vers = 'standard_ln'
# pos_or_neg = 1
# qb_wr_stack = 2
# qb_te_stack = 0
# min_opp_team = 0
# max_salary_remain = 500
# max_teams_lineup = 6  # Maximum number of teams in the lineup
# max_overlap = 7  # Maximum number of players that can overlap with any previous lineup

# to_add = []
# to_drop = []

# num_iters = 1
# num_options = 50
# num_avg_pts = 50

# prev_qb_wt = 7

# last_lineup, player_cnts = sim.run_sim(conn, to_add, to_drop, num_iters, ownership_vers, num_options, num_avg_pts,
#                                        pos_or_neg, qb_wr_stack, qb_te_stack, min_opp_team, max_teams_lineup,
#                                        max_salary_remain, max_overlap, prev_qb_wt, previous_lineups)

# last_lineup, player_cnts


#%%

from joblib import Parallel, delayed

class RunSim:

    def __init__(self, db_path, week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups):
        
        if '.sqlite3' not in db_path: self.db_path = f'{db_path}/Simulation.sqlite3'
        else: self.db_path = db_path

        self.week = week
        self.year = year
        self.pred_vers = pred_vers
        self.reg_ens_vers = reg_ens_vers
        self.million_ens_vers = million_ens_vers
        self.std_dev_type = std_dev_type
        self.total_lineups = total_lineups
        self.col_ordering = ['full_model_rel_weight', 'covar_type', 'num_iters', 'ownership_vers_variable', 
                             'ownership_vers',  'num_options', 'num_avg_pts',
                             'pos_or_neg', 'qb_wr_stack', 'qb_te_stack', 'min_opp_team', 'max_teams_lineup',
                             'max_salary_remain', 'max_overlap', 'prev_qb_wt']
        
        try:
            stats_conn = sqlite3.connect(f'{db_path}/FastR.sqlite3', timeout=60)
            results_conn = sqlite3.connect(f'{db_path}/DK_Results.sqlite3', timeout=60)
            self.player_stats = self.pull_past_points(stats_conn, week, year)
            self.prizes = self.get_past_prizes(results_conn, week, year)
        except:
            print('No Stats or DK Results')

    def create_conn(self):
        return sqlite3.connect(self.db_path, timeout=60)
        
    @staticmethod
    def get_past_stats(stats_conn, pos, week, year):
        if pos=='Defense': colname='defTeam'
        else: colname='player'
        return pd.read_sql_query(f'''SELECT {colname} AS player, fantasy_pts
                                     FROM {pos}_Stats
                                     WHERE week={week}
                                            AND season={year}''', stats_conn)



    def pull_past_points(self, stats_conn, week, year):

        points = pd.DataFrame()
        for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
            points = pd.concat([points, self.get_past_stats(stats_conn, pos, week, year)])

        return points

    def calc_winnings(self, to_add):
        results = pd.DataFrame(to_add, columns=['player'])
        results = pd.merge(results, self.player_stats, on='player')
        total_pts = results.fantasy_pts.sum()
        idx_match = np.argmin(abs(self.prizes.Points - total_pts))
        prize_money = self.prizes.loc[idx_match, 'prize']

        return prize_money, results

    @staticmethod
    def get_past_prizes(results_conn, week, year):
        prizes = pd.read_sql_query(f'''SELECT Rank, Points, prize
                                        FROM Contest_Results
                                        WHERE week={week}
                                            AND year={year}
                                            AND Contest='Million' ''', results_conn)
        return prizes
    
    @staticmethod
    def adjust_high_winnings(tw, max_adjust=10000):
        tw = np.array(tw)
        tw[tw>max_adjust] = max_adjust
        return list(tw)
    
    def generate_param_list(self, d):
    
        extra_keys = [k for k,_ in d.items() if k not in self.col_ordering]
        missing_keys = [k for k in self.col_ordering if k not in d.keys()]

        if len(extra_keys) > 0:
            raise ValueError(f'Extra keys: {extra_keys}')
        if len(missing_keys) > 0:
            raise ValueError(f'Missing keys: {missing_keys}')

        d = {k: d[k] for k in self.col_ordering}
        params = []
        for i in range(self.total_lineups):
            cur_params = []
            for k, param_options in d.items():
                if k == 'ownership_vers' and ownership_var==1:
                    cur_params.append(param_options)
                else:
                    param_vars = list(param_options.keys())
                    param_prob = list(param_options.values())
                    cur_choice = np.random.choice(param_vars, p=param_prob)
                    cur_params.append(cur_choice)
                    if k=='ownership_vers_variable':
                        ownership_var = cur_choice

            cur_params.append(i)
            params.append(cur_params)

        return params
    
    def setup_sim(self, params):

        p = {k: v for k,v in zip(self.col_ordering, params)}
        print(p)
    
        if p['covar_type']=='no_covar': p['use_covar']=False
        else: p['use_covar']=True

        conn = self.create_conn()
        sim = FullLineupSim(self.week, self.year, conn, self.pred_vers, self.reg_ens_vers, 
                            self.million_ens_vers, self.std_dev_type, p['full_model_rel_weight'], p['use_covar'],
                            p['covar_type'], print_results=False)
        conn.close()
        return sim, p
    
    
    def run_single_iter(self, sim, p, to_add, to_drop_selected, previous_lineups):

        if p['ownership_vers_variable']==1:
            own_opt, own_prob = list(p['ownership_vers'].keys()), list(p['ownership_vers'].values())
            own_vers = np.random.choice(own_opt, p=own_prob)
        else:
            own_vers = p['ownership_vers']

        to_drop = []
        to_drop.extend(to_drop_selected)

        conn = self.create_conn()
        last_lineup = None
        i = 0
        while last_lineup is None and i < 10:
            last_lineup, player_cnts = sim.run_sim(conn, to_add, to_drop, p['num_iters'], own_vers, p['num_options'], p['num_avg_pts'],
                                                p['pos_or_neg'], p['qb_wr_stack'], p['qb_te_stack'], p['min_opp_team'], p['max_teams_lineup'],
                                                p['max_salary_remain'], p['max_overlap'], p['prev_qb_wt'], previous_lineups)
            i += 1
        conn.close() 
        player_cnts = player_cnts[~player_cnts.player.isin(to_add)].reset_index(drop=True)
        
        return last_lineup, player_cnts
                    
    def run_full_lineup(self, params, to_add, to_drop, previous_lineups):

        sim, p = self.setup_sim(params)

        if p['num_iters'] == 1:
            to_add, _ = self.run_single_iter(sim, p, to_add, to_drop, previous_lineups)

        else:
            i = 0  # Initialize the iteration counter
            while len(to_add) < 9 and i < 18:  # Use a while loop to control iterations and break if necessary
                _, results = self.run_single_iter(sim, p, to_add, to_drop, previous_lineups)
                prob = results.loc[:p['top_n_choices'], 'SelectionCounts'] / results.loc[:p['top_n_choices'], 'SelectionCounts'].sum()
            
                try: 
                    selected_player = np.random.choice(results.loc[:p['top_n_choices'], 'player'], p=prob)
                    to_add.append(selected_player)
                except: 
                    pass
                i += 1  # Increment the iteration counter    

        return to_add
    
    def run_multiple_lineups(self, params, calc_winnings=False, parallelize=False, n_jobs=-1, verbose=0):
        
        existing_players =[]
        if parallelize:
            all_lineups = Parallel(n_jobs=n_jobs, verbose=verbose)(
                                delayed(self.run_full_lineup)(cur_param, existing_players)
                                for cur_param in params
                                )
                                
        else:
            all_lineups = []
            for cur_params in params:
                to_add = self.run_full_lineup(cur_params, to_add=[], to_drop=[], previous_lineups=all_lineups)
                all_lineups.append(to_add)

        if calc_winnings:
            total_winnings = 0
            winnings_list = []
            player_results = pd.DataFrame()
            for i, lineup in enumerate(all_lineups):
  
                winnings, player_results_cur = self.calc_winnings(lineup)
                total_winnings += winnings
                winnings_list.append(winnings)

                player_results_cur = player_results_cur.assign(lineup_num=i, week=self.week, year=self.year)
                player_results = pd.concat([player_results, player_results_cur])

            return total_winnings, player_results, winnings_list
        
        else: 
            pass

        return all_lineups


# week = 6
# year = 2024
# total_lineups = 50

# model_vers = {
#             'million_ens_vers': 'random_full_stack_newp_matt0_brier1_include2_kfold3',
#             'pred_vers': 'sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_higherkb',
#             'reg_ens_vers': 'random_full_stack_newp_sera0_rsq0_mse1_include2_kfold3',
#             'std_dev_type': 'spline_class80_q80_matt0_brier1_kfold3'
#             }


# d = {
#      'covar_type': {'kmeans_pred_trunc': 0.0,
#                     'kmeans_pred_trunc_new': 0.0,
#                     'no_covar': 0.3,
#                     'team_points_trunc': 0.7,
#                     'team_points_trunc_avgproj': 0.0},
#     'full_model_rel_weight': {0.2: 0.6, 5: 0.4},
#     'pos_or_neg': {1: 1},
#     'qb_wr_stack': {0: 0.1, 1: 0.5, 2: 0.4},
#     'qb_te_stack': {0: 0.7, 1: 0.3},
#     'num_options': {50: 1},
#     'num_avg_pts': {50: 1},
#     'num_iters': {1: 1},
#     'ownership_vers_variable': {0: 1.0, 1: 0},
#     'ownership_vers': {'mil_div_standard_ln': 0,
#                         'mil_only': 0.3,
#                         'mil_times_standard_ln': 0.3,
#                         'standard_ln': 0.4},
#     'max_teams_lineup': {8: 0.4, 6: 0.4, 4: 0.2},
#     'max_salary_remain': {500: 1},
#     'max_overlap': {7: 1}, 
#     'min_opp_team': {0: 0.3, 1: 0.5, 2: 0.2},
#     'prev_qb_wt': {1: 1}
#     }

# print(f'Running week {week} for year {year}')

# pred_vers = model_vers['pred_vers']
# reg_ens_vers = model_vers['reg_ens_vers']
# million_ens_vers = model_vers['million_ens_vers']
# std_dev_type = model_vers['std_dev_type']

# path = 'C:/Users/mborysia/OneDrive - Starbucks/Documents/Github/Simulation/'
# rs = RunSim(path, week, year, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups)
# params = rs.generate_param_list(d)


#%%
# sim, p = rs.setup_sim(params[0])

# to_add = []
# to_drop = []
# previous_lineups = []
# rs.run_single_iter(sim, p, to_add, to_drop,previous_lineups)

#%%

# rs.run_multiple_lineups(params)
