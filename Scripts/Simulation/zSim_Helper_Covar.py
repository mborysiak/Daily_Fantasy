#%%

# core packages
import pandas as pd
import numpy as np
import copy
from collections import Counter
import contextlib

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp

import cvxopt
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'

class FootballSimulation:

    def __init__(self, dm, week, set_year, salary_cap, pos_require_start, num_iters, 
                 pred_vers='standard', ensemble_vers='no_weight', std_dev_type='spline',
                 covar_type='team_points', full_model_rel_weight=1, matchup_seed=False,
                 use_covar=True, use_ownership=0, salary_remain_max=None, db_name='Simulation'):

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
        self.use_ownership = use_ownership
        self.salary_remain_max = salary_remain_max  
        self.boot = False
        self.db_name = db_name

        if self.use_covar and 'boot' not in std_dev_type: 
            player_data = self.get_covar_means()
            self.covar = self.pull_covar()
            
        elif 'boot' in std_dev_type:
            player_data, self.points_dist = self.get_boot_data()
            self.boot = True
            self.use_covar = False

        else: 
            player_data = self.get_model_predictions()

        # join in salary data to player data
        self.player_data = self.join_salary(player_data)

        # pull in the vegas points
        self.vegas_points = self.pull_vegas_points()

        if matchup_seed: self.matchup_seed = np.random.randint(20000)
        else: self.matchup_seed = None



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
                                             AND full_model_rel_weight={self.full_model_rel_weight}''', 
                                             self.db_name)
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
                                       self.db_name)
        covar = pd.pivot_table(covar, index='player', columns='player_two').reset_index().fillna(0)
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
                                ''', self.db_name)
        df['weighting'] = 1
        df.loc[df.model_type=='full_model', 'weighting'] = self.full_model_rel_weight

        score_cols = ['pred_fp_per_game', 'std_dev', 'min_score', 'max_score']
        for c in score_cols: 
            print(df.dtypes)
            df[c] = df[c].astype('float')
            print(df.dtypes)
            df[c] = df[c] * df.weighting

        # Groupby and aggregate with namedAgg [1]:
        df = df.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                                'std_dev': 'sum',
                                                                'weighting': 'sum',
                                                                'min_score': 'sum',
                                                                'max_score': 'sum'})

        for c in score_cols: df[c] = df[c] / df.weighting
        df.loc[df.pos=='Defense', 'pos'] = 'DEF'
        teams = self.dm.read(f"SELECT player, team FROM Player_Teams WHERE week={self.week} AND year={self.set_year}", self.db_name)
        df = pd.merge(df, teams, on=['player'])

        drop_teams = self.get_drop_teams()
        df = df[~df.team.isin(drop_teams)].reset_index(drop=True)

        return df.drop('weighting', axis=1)
    
    def get_boot_data(self):
        df = self.dm.read(f'''SELECT *
                              FROM Model_Predictions_Boot
                              WHERE week={self.week}
                                    AND year={self.set_year}
                                    AND version='{self.pred_vers}'
                                    AND ensemble_vers='{self.ensemble_vers}'
                                    AND std_dev_type='{self.std_dev_type}'
                                    AND pos !='K'
                                    AND pos IS NOT NULL
                                    AND player!='Ryan Griffin'
                                        ''', self.db_name)
        
        df['weighting'] = 1
        df.loc[df.model_type=='full_model', 'weighting'] = self.full_model_rel_weight

        score_cols = []
        agg_dict = {'weighting': 'sum'}
        for c in df.columns:
            if c not in ('player', 'team', 'week', 'year', 'dk_salary', 'pos', 'version', 'ensemble_vers', 'std_dev_type', 'model_type', 'weighting'):
                df[c] = df[c] * df.weighting
                df = df.rename(columns={c: int(c)})
                score_cols.append(int(c))
                agg_dict[int(c)] = 'sum'
                
        # Groupby and aggregate with namedAgg [1]:
        df = df.groupby(['player', 'pos'], as_index=False).agg(agg_dict)
        
        for c in score_cols:
            df[c] = df[c] / df.weighting
        points = df.drop(['pos', 'weighting'], axis=1)

        df = df[['player', 'pos']].assign(pred_fp_per_game=df[score_cols].mean(axis=1))
        teams = self.dm.read(f"SELECT player, team FROM Player_Teams WHERE week={self.week} AND year={self.set_year}", self.db_name)
        df = pd.merge(df, teams, on=['player'])

        drop_teams = self.get_drop_teams()
        df = df[~df.team.isin(drop_teams)].reset_index(drop=True)
        
        df.loc[df.pos=='Defense', 'pos'] = 'DEF'
        
        return df, points


    def get_drop_teams(self):

        df = self.dm.read(f'''SELECT away_team, home_team, gametime 
                              FROM Gambling_Lines 
                              WHERE week={self.week} 
                                    and year={self.set_year} 
                    ''', self.db_name)
        df.gametime = pd.to_datetime(df.gametime)
        df['day_of_week'] = df.gametime.apply(lambda x: x.weekday())
        df['hour_in_day'] = df.gametime.apply(lambda x: x.hour)
        df = df[(df.day_of_week!=6) | (df.hour_in_day > 16) | (df.hour_in_day < 11)]
        drop_teams = list(df.away_team.values)
        drop_teams.extend(list(df.home_team.values))

        return drop_teams


    def join_salary(self, df):

        # add salaries to the dataframe and set index to player
        salaries = self.dm .read(f'''SELECT player, salary
                                     FROM Salaries
                                     WHERE year={self.set_year}
                                           AND week={self.week} ''', self.db_name)

        df = pd.merge(df, salaries, how='left', left_on='player', right_on='player')
        df.salary = df.salary.fillna(10000)

        return df

    def join_ownership_pred(self, df):

        # add salaries to the dataframe and set index to player
        ownership = self.dm.read(f'''SELECT player, pred_ownership, std_dev+0.01 as std_dev, min_score, max_score
                                      FROM Predicted_Ownership
                                      WHERE year={self.set_year}
                                            AND week={self.week}
                                            AND ownership_vers='{self.ownership_vers}' ''', self.db_name)

        if self.use_covar: df = df.drop(['pred_fp_per_game'], axis=1)
        elif self.boot: df = df.copy()
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

    def pull_vegas_points(self):

        # add salaries to the dataframe and set index to player
        df = self.dm.read(f'''SELECT team, implied_points_for vegas_points, std_dev, min_score, max_score
                              FROM Vegas_Points
                              WHERE year={self.set_year}
                                    AND week={self.week}
                                    ''', self.db_name)
        df = df[df.team.isin(self.player_data.team.unique())].reset_index(drop=True)
        return df

    def get_matchups(self):
        df = self.dm.read(f'''SELECT away_team, home_team
                              FROM Gambling_Lines 
                              WHERE week={self.week} 
                                    and year={self.set_year} 
                    ''', self.db_name)

        df = df[(df.away_team.isin(self.player_data.team.unique())) & \
                (df.home_team.isin(self.player_data.team.unique()))]

        matchups = {}
        for away, home in df.values:
            matchups[away] = home
            matchups[home] = away

        return matchups

    @staticmethod
    def get_min_players(min_players_same_team_input):
        if min_players_same_team_input=='Auto': 
                min_players_same_team= np.random.choice([-1, 2, 3, 4], p=[0.1, 0.55, 0.3, 0.05])
        else:
            min_players_same_team = min_players_same_team_input

        min_players_same_team += 1

        return min_players_same_team

    @staticmethod
    def get_min_players_opp_team(min_players_opp_team_input):
        if min_players_opp_team_input=='Auto': 

            min_players_opp_team = np.random.choice([0, 1, 2], p=[0.25, 0.4, 0.35])
        else:
            min_players_opp_team = min_players_opp_team_input

        return min_players_opp_team

    @staticmethod
    def get_top_players_from_team(df, top_players=5):
        
        df = df[df.pos.isin(['QB', 'WR', 'TE'])]
        df = df.sort_values(by=['team', 'pred_fp_per_game'], ascending=[True, False])
        df['player_rank'] = df.groupby('team').cumcount()
        df = df.loc[df.player_rank <= top_players-1, ['player', 'team', 'pred_fp_per_game']].reset_index(drop=True)

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
        elif col=='vegas_points': df = self.vegas_points

        pred_list = []
        for mean_val, sdev, min_sc, max_sc in df[[col, 'std_dev', 'min_score', 'max_score']].values:
            pred_list.append(self.trunc_normal(mean_val, sdev, min_sc, max_sc, num_options))

        return pd.DataFrame(pred_list)

    # def covar_dist(self, num_options=500):
    #     import scipy.stats as ss
    #     dist = ss.multivariate_normal(mean=self.player_data.pred_fp_per_game.values, 
    #                                   cov=self.covar.drop('player', axis=1).values, 
    #                                   allow_singular=True)
    #     predictions = pd.DataFrame(dist.rvs(num_options)).T
    #     return predictions

    @staticmethod
    def unique_matchup_pairs(matchups):
        import itertools
        sorted_matchups = [sorted(m) for m in matchups.items()]
        return list(m for m, _ in itertools.groupby(sorted_matchups))

    def covar_dist(self, num_options=500):

        unique_matchups = self.unique_matchup_pairs(self.matchups)
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
    
    @staticmethod
    def _df_shuffle(df):
        '''
        Input: A dataframe to be shuffled, row-by-row indepedently.
        Return: The same dataframe whose columns have been shuffled for each row.
        '''
        # store the index before converting to numpy
        idx = df.player
        df = df.drop('player', axis=1).values

        # shuffle each row separately, inplace, and convert o df
        _ = [np.random.shuffle(i) for i in df]

        return pd.DataFrame(df, index=idx).reset_index()


    def get_predictions(self, col, ownership=False, num_options=500):

        labels = self.player_data[['player', 'pos', 'team', 'salary']]

        if self.use_covar and not ownership: 
            predictions = self.covar_dist(num_options)
            predictions = pd.concat([labels, predictions], axis=1)

        elif self.boot:
            points_dist = self._df_shuffle(self.points_dist)
            idx_choices = ['player']
            idx_choices.extend(np.random.choice(range(1000), size=num_options))
            predictions = pd.merge(labels, points_dist[idx_choices], on=['player'])
            
            cols = ['player', 'pos', 'team', 'salary']
            cols.extend(range(num_options))
            predictions.columns = cols

        else: 
            predictions = self.trunc_normal_dist(col, num_options)
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
            flex_pos = np.random.choice(['RB', 'WR', 'TE'], p=[0.35, 0.55, 0.1])
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

        added_teams = self.player_data.loc[(self.player_data.player.isin(to_add)) & \
                                           (self.player_data.pos.isin(['QB', 'WR', 'TE'])), 
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

    def samples_G_ownership(self, data, max_entries):

        current_ownership = self.pos_or_neg * data.iloc[:, np.random.choice(range(4, max_entries+4))].values
        current_ownership = current_ownership.reshape(1, len(current_ownership))
        return current_ownership

    
    def create_h_ownership(self):
        mean_own, std_own = self.dm.read(f'''SELECT ownership_mean, ownership_std
                                             FROM Mean_Ownership
                                              WHERE week={self.week}
                                                    AND year={self.set_year}
                                                    AND ownership_vers='{self.ownership_vers}'
                                        ''', self.db_name).values[0]

        mean_own = self.pos_or_neg * np.random.normal(mean_own, std_own, size=1).reshape(1, 1)
        return mean_own

    @staticmethod
    def create_G_team(team_map, player_map, adjust_select):

        pos_wt = {
            'QB': -2,
            'WR': -1,
            'TE': -1,
            'RB': -0.5,
            'DEF': 0
        }
        
        num_players = len(player_map)
        num_teams = len(team_map)
        G_teams = np.zeros(shape=(num_teams, num_players))

        for i in range(num_players):

            cur_team = player_map[i]['team']
            cur_pos = player_map[i]['pos']
            G_teams[team_map[cur_team], i] = pos_wt[cur_pos]

        return G_teams

    def max_team_player_points(self, labels, c_points):
        team_pts = pd.concat([labels, -c_points], axis=1)
        team_pts.columns = ['player', 'pos', 'team', 'salary', 'pred_fp_per_game']
        
        if self.static_top_players: 
            team_pts = pd.merge(team_pts, self.top_team_players.drop('pred_fp_per_game', axis=1), on=['player', 'team'])
        else:
            team_pts = self.get_top_players_from_team(team_pts, top_players=self.n_top_players)
        
        team_pts = team_pts.groupby('team').agg({'pred_fp_per_game': 'sum'})
        best_teams = team_pts.apply(lambda x: pd.Series(x.nlargest(3).index))
        best_teams = [b[0] for b in best_teams.values]
        
        return best_teams

    def max_team_vegas_points(self):
        col_idx = np.random.choice(range(1,self.cur_vegas_pts.shape[1]), size=1)[0]
        best_teams = self.cur_vegas_pts.iloc[:, [0,col_idx]]
        best_teams = best_teams.sort_values(by=col_idx-1, ascending=False)
        best_teams = list(best_teams.team[:3].values)
        return best_teams

    def get_max_team(self, labels, c_points, added_teams):

        if self.max_team_type=='player_points':
            best_teams = self.max_team_player_points(labels, c_points)
        
        elif self.max_team_type=='vegas_points':
            best_teams = self.max_team_vegas_points()
        
        elif self.max_team_type=='all':
            best_teams = self.max_team_player_points(labels, c_points)
            best_teams_vp = self.max_team_vegas_points()
            best_teams.extend(best_teams_vp)

        best_teams.extend(added_teams)

        best_team = np.random.choice(best_teams)

        return best_team


    def create_h_teams(self, team_map, added_teams, set_max_team, min_players, opp_players):

        if set_max_team is None: 
            max_team = self.get_max_team(self.labels, self.c_points, added_teams)
        else:
            max_team = set_max_team

        h_teams = np.full(shape=(len(team_map), 1), fill_value=0.0)
        h_teams[team_map[max_team]] = -min_players

        # use a bring back player
        if opp_players > 0:
            max_team_opponent = self.matchups[max_team]
            h_teams[team_map[max_team_opponent]] = -opp_players/2
        
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
        results = results.sort_values(by='SelectionCounts', ascending=False).iloc[:59]
        salaries = self.player_data[['player', 'salary']].set_index('player')
        results = pd.merge(results, salaries, left_index=True, right_index=True)
        results = results.reset_index().rename(columns={'index': 'player'})
        results.SelectionCounts = 100*np.round(results.SelectionCounts / success_trials, 3)
        return results

    @staticmethod
    def final_team_cnts(max_team_cnt):
        max_team_cnt = Counter(max_team_cnt)
        df = pd.DataFrame(dict(max_team_cnt), index=[0]).T.reset_index()
        df.columns=['team', 'high_score_team']
        df = df.sort_values(by='high_score_team', ascending=False).reset_index(drop=True)

        return df

    def adjust_select_perc(self, df, open_pos_require):
        df = pd.merge(df, self.player_data[['player', 'pos']], on='player')

        pos_require_df = pd.DataFrame(open_pos_require, index=[0]).T.reset_index()
        pos_require_df.columns = ['pos', 'num_required']
        pos_require_df.num_required = pos_require_df.num_required + [0, 0.5, 0.5, 0.5, 1]
        # pos_require_df.num_required = pos_require_df.num_required + [-1, 2, 0, 1, 2]
        
        df = pd.merge(df, pos_require_df, on='pos')
        df.loc[df.SelectionCounts < self.num_iters, 'SelectionCounts'] = \
            df.loc[df.SelectionCounts < self.num_iters, 'SelectionCounts'] / \
                (df.loc[df.SelectionCounts < self.num_iters, 'num_required']+1)
        df = df.sort_values(by='SelectionCounts', ascending=False).reset_index(drop=True)
        df = df.drop(['pos', 'num_required'], axis=1)
        return df

    @staticmethod
    @contextlib.contextmanager
    def temp_seed(seed):
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(state)

    def player_matchup_drop(self, to_drop, to_add, num_matchup_drop):

        with self.temp_seed(self.matchup_seed):
            drop_teams = np.random.choice(list(self.matchups.keys()), num_matchup_drop, replace=False)

        lineup_teams = self.player_data.loc[self.player_data.player.isin(to_add), 'team'].unique()
        
        matchup_to_drop = []
        for t in drop_teams:
            if t not in lineup_teams and self.matchups[t] not in lineup_teams: 
                matchup_to_drop.extend([t, self.matchups[t]])

        to_drop.extend(self.player_data.loc[self.player_data.team.isin(matchup_to_drop), 'player'].values)

        return to_drop, matchup_to_drop


    def run_sim(self, to_add, to_drop, min_players_same_team_input, set_max_team, 
                min_players_opp_team_input=0, adjust_select=False, max_team_type='player_points',
                num_matchup_drop=0, own_neg_frac=1, n_top_players=5, ownership_vers='standard_ln',
                static_top_players=True, qb_min_iter=9, qb_set_max_team=False, qb_solo_start=True):
        
        # can set as argument, but static set for now
        num_options=250
        player_selections = self.init_select_cnts()
        max_team_cnt = []
        success_trials = 0

        self.ownership_vers = ownership_vers
        if self.use_ownership > 0:
            self.ownership_data = self.join_ownership_pred(self.player_data)
            self.use_own_frac = self.use_ownership
       
        self.matchups = self.get_matchups()
        if num_matchup_drop > 0:
            to_drop, drop_matchup_teams = self.player_matchup_drop(to_drop, to_add, num_matchup_drop)
        else:
            drop_matchup_teams = []
            
        # decide whether to use above or below ownership threshold
        self.pos_or_neg = np.random.choice([-1, 1], p=[own_neg_frac, 1-own_neg_frac])

        # randomly decide to use threshold or not
        self.use_ownership = np.random.choice([True, False], p=[self.use_own_frac, 1-self.use_own_frac])

        # randomly decide max_team_type
        self.max_team_type = max_team_type

        # extract the top players for each team
        self.n_top_players = n_top_players
        self.static_top_players = static_top_players
        self.top_team_players = self.get_top_players_from_team(self.player_data, top_players=self.n_top_players)

        for i in range(self.num_iters):

            min_players_same_team = self.get_min_players(min_players_same_team_input)
            min_player_opp_team = self.get_min_players_opp_team(min_players_opp_team_input)
            
            if i ==0:
                # pull out current add players and added teams
                h_player_add, open_pos_require = self.add_players(to_add)
                remaining_pos_cnt = np.sum(list(open_pos_require.values()))
                added_teams, max_added_team_cnt = self.get_current_team_cnts(to_add)

            # append the flex position to position requirements
            cur_pos_require = self.add_flex(open_pos_require)
            b_position = self.create_b_matrix(cur_pos_require)

            if i % 25 == 0:

                # remove QB defense opponent from dataset
                if open_pos_require['QB'] == 0:
                    qb_team = self.player_data.loc[(self.player_data.player.isin(to_add)) & \
                                                   (self.player_data.pos=='QB'), 'team'].values[0]
                    qb_def_team = self.matchups[qb_team]
                    if qb_def_team not in to_add: 
                        to_drop.append(qb_def_team)
                
                # get predictions and remove to drop players
                predictions = self.get_predictions(col='pred_fp_per_game', num_options=num_options)
                predictions = self.drop_players(predictions, to_drop)

                if self.use_ownership:
                    try:
                        # get ownership and remove to drop players
                        ownership = self.get_predictions(col='pred_ownership', ownership=True, num_options=num_options)
                        ownership = self.drop_players(ownership, to_drop)
                    except:
                        self.use_ownership = False
                        print('Ownership failed')

                # create vegas points if needed
                if self.max_team_type in ['all', 'vegas_points']:
                    self.cur_vegas_pts = pd.concat([self.vegas_points.team, self.trunc_normal_dist('vegas_points')], axis=1)
                    if len(drop_matchup_teams) > 0:
                        self.cur_vegas_pts = self.cur_vegas_pts[~self.cur_vegas_pts.team.isin(drop_matchup_teams)].reset_index(drop=True)

            if i == 0:

                position_map = self.position_matrix_mapping(cur_pos_require)
                idx_player_map, player_idx_map = self.player_matrix_mapping(predictions)
                team_map = self.team_matrix_mapping(predictions)

                A_position = self.create_A_position(position_map, idx_player_map)

                G_salaries = self.create_G_salaries(predictions)
                h_salaries = self.create_h_salaries()
                
                G_players = self.create_G_players(player_idx_map)
                h_players = self.create_h_players(player_idx_map, h_player_add)

                G_teams = self.create_G_team(team_map, idx_player_map, adjust_select)
        
            # generate the c matrix with the point values to be optimized
            self.labels, self.c_points = self.sample_c_points(predictions, num_options)
            
            if remaining_pos_cnt > (min_players_same_team+min_player_opp_team-1) and max_added_team_cnt < min_players_same_team:
             
                if open_pos_require['QB'] == 0 and qb_set_max_team:
                    set_max_team = self.player_data.loc[(self.player_data.player.isin(to_add)) & \
                                                        (self.player_data.pos=='QB'), 'team'].values[0]

                if open_pos_require['QB'] == 1 and qb_solo_start:
                    min_players_same_team = -1
                    min_player_opp_team = -1
               
                h_teams, max_team = self.create_h_teams(team_map, added_teams, set_max_team, min_players_same_team, min_player_opp_team)
                max_team_cnt.append(max_team)
                G = np.concatenate([G_salaries, G_teams, G_players])
                h = np.concatenate([h_salaries, h_teams, h_players])

            else:
                G = np.concatenate([G_salaries, G_players])
                h = np.concatenate([h_salaries, h_players])

            if self.use_ownership:

                G_ownership = self.samples_G_ownership(ownership, num_options)
                G = np.concatenate([G, G_ownership])

                h_ownership = self.create_h_ownership()
                h = np.concatenate([h, h_ownership])

            if self.salary_remain_max is not None:
                G_salaries_min = -self.create_G_salaries(predictions)
                h_salaries_min = -(self.create_h_salaries()-self.salary_remain_max)
                
                G = np.concatenate([G, G_salaries_min])
                h = np.concatenate([h, h_salaries_min])

            G = matrix(G, tc='d')
            h = matrix(h, tc='d')
            b = matrix(b_position, tc='d')
            A = matrix(A_position, tc='d')    
            c = matrix(self.c_points, tc='d')

            if len(to_add) < 9:
                status, x = self.solve_ilp(c, G, h, A, b)
                if status=='optimal':
                    player_selections = self.tally_player_selections(predictions, player_selections, x)
                    success_trials += 1

        results = self.final_results(player_selections, success_trials)
        if adjust_select:
            results = self.adjust_select_perc(results, open_pos_require)

        team_cnts = self.final_team_cnts(max_team_cnt)

        if len(to_add) == qb_min_iter and open_pos_require['QB'] == 1:
            results = pd.merge(results, self.player_data[['player', 'pos']], on='player')
            results = results[results.pos=='QB'].reset_index(drop=True).drop('pos', axis=1)

        results = results.iloc[:30]

        return results, team_cnts



#%%

# # set the root path and database management object
# from ff.db_operations import DataManage
# from ff import general as ffgeneral

# root_path = ffgeneral.get_main_path('Daily_Fantasy')
# db_path = f'{root_path}/Data/Databases/'
# dm = DataManage(db_path)


# adjust_select = True
# matchup_drop = 2
# full_model_weight = 5
# covar_type = 'no_covar'
# max_team_type = 'vegas_points'
# use_covar = True
# min_players_same_team = 'Auto'
# min_players_opp_team = 2
# top_n_players = 2
# qb_min_iter = 5
# qb_solo_start = False
# qb_set_max_team = False
# static_top_players = True
# use_ownership = 1
# own_neg_frac = 1
# salary_remain_max = 1000
# num_iters = 150

# pred_vers = 'sera1_rsq0_brier1_matt1_lowsample_perc_ffa_fc'
# ens_vers = 'no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3_fullstack'
# std_dev_type = 'boot_reg_quant_frac_random_replace_random'
# ownership_vers = 'standard_ln'

# week = 12
# year = 2022
# salary_cap = 50000
# pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
# set_max_team = None

# sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
#                          ensemble_vers=ens_vers, pred_vers=pred_vers, std_dev_type=std_dev_type,
#                          full_model_rel_weight=full_model_weight, covar_type=covar_type, use_covar=use_covar, 
#                          use_ownership=use_ownership, salary_remain_max=salary_remain_max, matchup_seed=False)


# to_add = []
# to_drop = []

# results, max_team_cnt = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, 
#                                     min_players_opp_team, adjust_select=adjust_select, max_team_type=max_team_type,
#                                     num_matchup_drop=matchup_drop, own_neg_frac=own_neg_frac,
#                                     n_top_players=top_n_players, static_top_players=static_top_players,
#                                      qb_solo_start=qb_solo_start, ownership_vers=ownership_vers,
#                                     qb_set_max_team=qb_set_max_team, qb_min_iter=qb_min_iter)

# print(max_team_cnt)
# results

 # %%

# def calc_winnings(to_add, points, prizes):
#     results = pd.DataFrame(to_add, columns=['player'])
#     results = pd.merge(results, points, on='player')
#     total_pts = results.fantasy_pts.sum()
#     idx_match = np.argmin(abs(prizes.Points - total_pts))
#     prize_money = prizes.loc[idx_match, 'prize']

#     return prize_money

# def rand_drop_selected(total_add, drop_multiplier, lineups_per_param):
#     to_drop = []
#     total_selections = dict(Counter(total_add))
#     for k, v in total_selections.items():
#         prob_drop = (v * drop_multiplier) / lineups_per_param
#         drop_val = np.random.uniform() * prob_drop
#         if  drop_val >= 0.5:
#             to_drop.append(k)
#     return to_drop

# lineups_per_param = 3
# top_n_choices = 1

# winnings = []        
# total_add = []
# to_drop_selected = []
# for t in range(lineups_per_param):

#     to_add = []
#     print('drop_players:', to_drop)
#     sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
#                         pred_vers, ensemble_vers=ens_vers, std_dev_type=std_dev_type,
#                         covar_type=covar_type,  full_model_rel_weight=full_model_weight, 
#                         use_covar=use_covar, use_ownership=use_ownership, 
#                         salary_remain_max=salary_remain_max, matchup_seed=True)
    
#     for i in range(9):
      
#         to_drop = []
#         to_drop.extend(to_drop_selected)
#         results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, 
#                                 min_players_opp_team_input=min_players_opp_team, 
#                                 adjust_select=adjust_select,max_team_type=max_team_type,
#                                 num_matchup_drop=matchup_drop,ownership_vers=ownership_vers,
#                                 own_neg_frac=own_neg_frac, n_top_players=top_n_players,
#                                 static_top_players=static_top_players, qb_min_iter=qb_min_iter,
#                                 qb_set_max_team=qb_set_max_team, qb_solo_start=qb_solo_start)
        
#         prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
        
#         selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
#         to_add.append(selected_player)
    
#         print('add players:', to_add)

#     total_add.extend(to_add)
#     to_drop_selected = rand_drop_selected(total_add, 4, 2)
#     print('player_drop_multiple:', to_drop_selected)
# %%
