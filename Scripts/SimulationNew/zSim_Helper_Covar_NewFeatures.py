#%%

# core packages
import pandas as pd
import numpy as np
import copy
from collections import Counter
import contextlib
import sqlite3

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp

import cvxopt
cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'


#%%
class FootballSimulation:

    def __init__(self, conn, week, set_year, salary_cap, pos_require_start, num_iters, 
                 pred_vers='standard', reg_ens_vers='no_weight', million_ens_vers='random_matt0_brier1_include2_kfold3',
                 std_dev_type='spline', covar_type='team_points', full_model_rel_weight=1, matchup_seed=False,
                 use_covar=True, use_ownership=0, salary_remain_max=None, min_pts_per_dollar=0, max_pts_per_dollar=100,
                 min_pred_pts=0, min_pts_variable=0, max_pts_variable=0):

        self.week = week
        self.set_year = set_year
        self.pos_require_start = pos_require_start
        self.num_iters = num_iters
        self.salary_cap = salary_cap
        self.pred_vers = pred_vers
        self.reg_ens_vers = reg_ens_vers
        self.std_dev_type = std_dev_type
        self.million_ens_vers = million_ens_vers
        self.covar_type = covar_type
        self.full_model_rel_weight = full_model_rel_weight
        self.use_covar = use_covar
        self.use_ownership = use_ownership
        self.salary_remain_max = salary_remain_max 
        self.min_pts_variable = min_pts_variable 
        self.max_pts_variable = max_pts_variable
        self.min_pts_per_dollar = min_pts_per_dollar
        self.min_pred_pts = min_pred_pts
        self.max_pts_per_dollar = max_pts_per_dollar
        self.boot = False
        self.conn = conn

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

    def pull_vegas_points(self):

        # add salaries to the dataframe and set index to player
        df = pd.read_sql_query(f'''SELECT team, implied_points_for vegas_points, std_dev, min_score, max_score
                                   FROM Vegas_Points
                                   WHERE year={self.set_year}
                                            AND week={self.week}
                                            ''', self.conn)
        df = df[df.team.isin(self.player_data.team.unique())].reset_index(drop=True)
        return df

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

    @staticmethod
    def get_min_players(min_players_same_team_input, qb_stack_wt):
        if min_players_same_team_input=='Auto': 
                min_players_same_team= np.random.choice([-1, 2, 3, 4], p=[0.1, 0.55, 0.3, 0.05])
        else:
            min_players_same_team = min_players_same_team_input

        min_players_same_team += (qb_stack_wt - 1)

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
            flex_pos = np.random.choice(['RB', 'WR', 'TE'], p=[0.37, 0.51, 0.12])
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


    @staticmethod
    def drop_players(df, to_drop):
        return df[~df.player.isin(to_drop)].reset_index(drop=True)

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


    @staticmethod
    def sample_c_points(data, max_entries, num_avg_pts):

        labels = data[['player', 'pos', 'team', 'salary']]
        current_points = -1 * data.iloc[:, np.random.choice(range(4, max_entries+4), size=num_avg_pts)].mean(axis=1)

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
    
    @staticmethod
    def remove_max_pick(results, to_add, rb_max_pick, wr_max_pick, te_max_pick, def_max_pick):
        for pos, max_pick in zip(['RB', 'WR', 'TE', 'DEF'], 
                                    [rb_max_pick, wr_max_pick, te_max_pick, def_max_pick]):
            if len(to_add) < max_pick:
                results = results[results.pos!=pos].reset_index(drop=True)
        
        return results

    @staticmethod
    def filter_points_and_sal(df, num_choices, min_pts_per_dollar, max_pts_per_dollar, min_pred_pts, qb_max_sal):

        player_order = df[['player']].copy()
        df['pred_fp_per_game'] = df.loc[:,0:num_choices-1].mean(axis=1)
        df['pts_per_dollar'] = 1000*df.pred_fp_per_game / df.salary
        pts_per_dollar_pct_max = df.groupby('pos').agg({'pts_per_dollar': lambda x: np.percentile(x, max_pts_per_dollar)}).reset_index().rename(columns={'pts_per_dollar': 'pts_per_dollar_pct_max'})
        pts_per_dollar_pct_min = df.groupby('pos').agg({'pts_per_dollar': lambda x: np.percentile(x, min_pts_per_dollar)}).reset_index().rename(columns={'pts_per_dollar': 'pts_per_dollar_pct_min'})
        df = pd.merge(df, pts_per_dollar_pct_max, on='pos')
        df = pd.merge(df, pts_per_dollar_pct_min, on='pos')
        
        df.loc[((df.pos=='QB') & (df.salary > qb_max_sal)) | \
               (df.pts_per_dollar > df.pts_per_dollar_pct_max+0.001) | \
               (df.pred_fp_per_game < min_pred_pts) | \
               (df.pts_per_dollar < df.pts_per_dollar_pct_min-0.001), 
               0:num_choices-1] = 0
        
        df = pd.merge(player_order, df, on='player')
        df = df.drop(['pts_per_dollar', 'pred_fp_per_game', 'pts_per_dollar_pct_max', 'pts_per_dollar_pct_min'], axis=1)

        return df
    



    def run_sim(self, conn, to_add, to_drop, min_players_same_team_input, set_max_team, 
                min_players_opp_team_input=0, adjust_select=False, max_team_type='player_points',
                num_matchup_drop=0, own_neg_frac=1, n_top_players=5, ownership_vers='standard_ln',
                static_top_players=True, qb_min_iter=9, qb_set_max_team=False, qb_solo_start=True,
                num_avg_pts=1, qb_stack_wt=2, rb_max_pick=0, wr_max_pick=0, te_max_pick=0, def_max_pick=0,
                qb_max_sal=10000, rb_min_sal=3000):
                
        min_pred_pts = np.random.choice([0, self.min_pred_pts], p=[1-self.min_pts_variable, self.min_pts_variable])
        min_pts_per_dollar = np.random.choice([0, self.min_pts_per_dollar], p=[1-self.min_pts_variable, self.min_pts_variable])
        max_pts_per_dollar = np.random.choice([100, self.max_pts_per_dollar], p=[1-self.max_pts_variable, self.max_pts_variable])

        # can set as argument, but static set for now
        self.conn = conn
        num_options=3000
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

        for i in range(self.num_iters):

            min_players_same_team = self.get_min_players(min_players_same_team_input, qb_stack_wt)
            min_player_opp_team = self.get_min_players_opp_team(min_players_opp_team_input)
            
            if i ==0:
                # pull out current add players and added teams
                h_player_add, open_pos_require = self.add_players(to_add)
                remaining_pos_cnt = np.sum(list(open_pos_require.values()))

                # remove QB defense opponent from dataset
                if open_pos_require['QB'] == 0:
                    qb_team = self.player_data.loc[(self.player_data.player.isin(to_add)) & \
                                                   (self.player_data.pos=='QB'), 'team'].values[0]
                    qb_def_team = self.matchups[qb_team]
                    if qb_def_team not in to_add: 
                        to_drop.append(qb_def_team)

            # append the flex position to position requirements
            cur_pos_require = self.add_flex(open_pos_require)
            b_position = self.create_b_matrix(cur_pos_require)

            if i % 150 == 0:
                
                # get predictions and remove to drop players
                predictions = self.get_predictions(col='pred_fp_per_game', num_options=num_options)
                predictions = self.filter_points_and_sal(predictions, num_options, min_pts_per_dollar, max_pts_per_dollar, min_pred_pts, qb_max_sal)
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

            # if i == 0:

             
        
            # generate the c matrix with the point values to be optimized
            self.labels, self.c_points = self.sample_c_points(predictions, num_options, num_avg_pts)
            
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
        results = pd.merge(results, self.player_data[['player', 'pos']], on='player')

        if len(to_add) == qb_min_iter and open_pos_require['QB'] == 1:
            results = results[results.pos=='QB'].reset_index(drop=True)

        results = self.remove_max_pick(results, to_add, rb_max_pick, wr_max_pick, te_max_pick, def_max_pick)
        results = results.drop('pos', axis=1).iloc[:30]

        return results, team_cnts



#%%
from joblib import Parallel, delayed

class RunSim:

    def __init__(self, db_path, week, year, salary_cap, pos_require_start,
                 pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups):
        
        if '.sqlite3' not in db_path: self.db_path = f'{db_path}/Simulation.sqlite3'
        else: self.db_path = db_path

        self.week = week
        self.year = year
        self.salary_cap = salary_cap
        self.pos_require_start = pos_require_start
        self.pred_vers = pred_vers
        self.reg_ens_vers = reg_ens_vers
        self.million_ens_vers = million_ens_vers
        self.std_dev_type = std_dev_type
        self.total_lineups = total_lineups
        self.col_ordering = ['adjust_pos_counts', 'player_drop_multiple', 'matchup_seed', 'matchup_drop', 
                             'top_n_choices', 'full_model_weight', 'covar_type', 'max_team_type',
                             'min_player_same_team', 'min_players_opp_team', 'num_top_players', 
                             'ownership_vers_variable', 'ownership_vers', 'qb_min_iter', 'qb_set_max_team',
                             'qb_solo_start', 'qb_stack_wt', 'static_top_players', 'use_ownership', 'own_neg_frac',
                             'max_salary_remain', 'num_iters', 'num_avg_pts', 'use_unique_players',
                             'rb_max_pick', 'wr_max_pick', 'te_max_pick', 'def_max_pick', 'min_pts_per_dollar',
                             'min_pred_pts', 'min_pts_variable', 
                             'max_pts_variable', 'max_pts_per_dollar', 'qb_max_sal', 'rb_min_sal']
        
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
    def rand_drop_selected(total_add, drop_multiplier, lineups_per_param):
        to_drop = []
        total_selections = dict(Counter(total_add))
        for k, v in total_selections.items():
            prob_drop = (v * drop_multiplier) / lineups_per_param
            drop_val = np.random.uniform() * prob_drop
            if  drop_val >= 0.5:
                to_drop.append(k)
        return to_drop
    
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
    
    def setup_sim(self, params, existing_players):

        p = {k: v for k,v in zip(self.col_ordering, params)}
        print(p)
                
        try: p['min_players_opp_team'] = int(p['min_players_opp_team'])
        except: pass

        try: p['min_player_same_team'] = float(p['min_player_same_team'])
        except: pass

        if p['covar_type']=='no_covar': p['use_covar']=False
        else: p['use_covar']=True

        if p['use_unique_players']: drop_length = len(set(existing_players)) + 5
        else: drop_length = self.total_lineups + 5
        to_drop_selected = self.rand_drop_selected(existing_players, p['player_drop_multiple'], drop_length)

        to_add = []
        conn = self.create_conn()
        sim = FootballSimulation(conn, self.week, self.year, self.salary_cap, self.pos_require_start, p['num_iters'], 
                                 self.pred_vers, self.reg_ens_vers, self.million_ens_vers, self.std_dev_type, 
                                 p['covar_type'], p['full_model_weight'], p['matchup_seed'], p['use_covar'], p['use_ownership'], 
                                 p['max_salary_remain'], p['min_pts_per_dollar'], p['max_pts_per_dollar'],
                                 p['min_pred_pts'], p['min_pts_variable'], p['max_pts_variable'])
        conn.close()
        return sim, p, to_add, to_drop_selected
    
    
    def run_single_iter(self, sim, p, to_add, to_drop_selected, set_max_team=None):
        if p['ownership_vers_variable']==1:
            own_opt, own_prob = list(p['ownership_vers'].keys()), list(p['ownership_vers'].values())
            own_vers = np.random.choice(own_opt, p=own_prob)
        else:
            own_vers = p['ownership_vers']

        to_drop = []
        to_drop.extend(to_drop_selected)

        conn = self.create_conn()
        results, _ = sim.run_sim(conn, to_add, to_drop, p['min_player_same_team'], set_max_team, 
                                p['min_players_opp_team'], p['adjust_pos_counts'], p['max_team_type'],
                                p['matchup_drop'], p['own_neg_frac'], p['num_top_players'], own_vers,
                                p['static_top_players'], p['qb_min_iter'], p['qb_set_max_team'], p['qb_solo_start'],
                                p['num_avg_pts'], p['qb_stack_wt'], p['rb_max_pick'], p['wr_max_pick'],
                                p['te_max_pick'], p['def_max_pick'], p['qb_max_sal'], p['rb_min_sal'])
        conn.close()
        results = results[~results.player.isin(to_add)].reset_index(drop=True)
        
        return results
                    
    def run_full_lineup(self, params, existing_players=[], set_max_team=None):

        sim, p, to_add, to_drop_selected = self.setup_sim(params, existing_players)

        i = 0  # Initialize the iteration counter
        while len(to_add) < 9 and i < 18:  # Use a while loop to control iterations and break if necessary
            results = self.run_single_iter(sim, p, to_add, to_drop_selected, set_max_team)
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
                to_add = self.run_full_lineup(cur_params, existing_players)
                existing_players.extend(to_add)
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


#%%

for week in range(2,3):
    year = 2024
    print(f'Running week {week} for year {year}')
    total_lineups = 50
    salary_cap = 50000
    pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}

    model_vers = {
                'million_ens_vers': 'random_full_stack_newp_matt0_brier1_include2_kfold3',
                'pred_vers': 'sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_higherkb',
                'reg_ens_vers': 'random_full_stack_newp_sera0_rsq0_mse1_include2_kfold3',
                'std_dev_type': 'spline_pred_class80_q80_matt0_brier1_kfold3'
                }
    d = {'adjust_pos_counts': {False: 0.3, True: 0.7},
        'covar_type': {'kmeans_pred_trunc': 0.0,
                        'kmeans_pred_trunc_new': 0.0,
                        'no_covar': 0.3,
                        'team_points_trunc': 0.7,
                        'team_points_trunc_avgproj': 0.0},
        'def_max_pick': {0: 1.0, 5: 0.0, 7: 0.0, 8: 0.0},
        'full_model_weight': {0.2: 0.6, 5: 0.4},
        'matchup_drop': {0: 0.8, 1: 0.2, 2: 0.0, 3: 0.0},
        'matchup_seed': {0: 0.8, 1: 0.2},
        'max_pts_per_dollar': {95: 0.1, 98: 0.3, 100: 0.6},
        'max_pts_variable': {0: 0.0, 0.3: 0.0, 0.5: 0.5, 1: 0.5},
        'max_salary_remain': {200: 0.0, 300: 0.0, 500: 0.7, 1000: 0.3, 1500: 0.0},
        'max_team_type': {'player_points': 0.7, 'vegas_points': 0.3},
        'min_player_same_team': {2: 0.2, 3: 0.4, 'Auto': 0.4},
        'min_players_opp_team': {1: 0.1, 2: 0.2, 'Auto': 0.7},
        'min_pred_pts': {0: 0.3, 5: 0.5, 7: 0.2},
        'min_pts_per_dollar': {0: 0, 20: 1},
        'min_pts_variable': {0: 0, 1: 1},
        'num_avg_pts': {1: 0.0, 2: 0.0, 3: 0.0, 5: 0.0, 7: 0.3, 10: 0.7},
        'num_iters': {50: 0.0, 100: 0.0, 150: 0.5, 200: 0.5},
        'num_top_players': {2: 0.0, 3: 0.2, 5: 0.8},
        'own_neg_frac': {0.8: 0.0, 0.9: 0.0, 1: 1.0},
        'ownership_vers': {'mil_div_standard_ln': 0.2,
                            'mil_only': 0.2,
                            'mil_times_standard_ln': 0.3,
                            'standard_ln': 0.3},
        'ownership_vers_variable': {0: 0.0, 1: 1.0},
        'player_drop_multiple': {0: 1.0, 2: 0.0, 4: 0.0, 10: 0.0, 20: 0.0, 30: 0.0},
        'qb_max_sal': {6000: 0.5, 10000: 0.5},
        'qb_min_iter': {0: 0.2, 2: 0.6, 4: 0.2, 9: 0.0},
        'qb_set_max_team': {0: 0.6, 1: 0.4},
        'qb_solo_start': {False: 0.7, True: 0.3},
        'qb_stack_wt': {1: 0.0, 2: 0.0, 3: 0.3, 4: 0.7},
        'rb_max_pick': {0: 0.0, 3: 1.0, 4: 0.0},
        'static_top_players': {False: 0.5, True: 0.5},
        'te_max_pick': {0: 1.0},
        'top_n_choices': {0: 1.0, 1: 0.0, 2: 0.0},
        'use_ownership': {0.7: 0.0, 0.8: 0.5, 0.9: 0.5, 1: 0.0},
        'use_unique_players': {0: 1.0, 1: 0.0},
        'wr_max_pick': {0: 1.0},
        'rb_min_sal': {3000: 1.0}
        }
    
    pred_vers = model_vers['pred_vers']
    reg_ens_vers = model_vers['reg_ens_vers']
    million_ens_vers = model_vers['million_ens_vers']
    std_dev_type = model_vers['std_dev_type']

    path = 'C:/Users/borys/OneDrive/Documents/Github/Daily_Fantasy/Data/Databases/'
    rs = RunSim(path, week, year, salary_cap, pos_require_start, pred_vers, reg_ens_vers, million_ens_vers, std_dev_type, total_lineups)
    params = rs.generate_param_list(d)


#%%

print('New Features')
sim, p, to_add, to_drop_selected = rs.setup_sim(params[0], existing_players=[])

# #%%
# import time
# start = time.time()
# results = rs.run_single_iter(sim, p, to_add, to_drop_selected, set_max_team=None)
# print(time.time()-start)
# results

# #%%

# rs.run_full_lineup(params[0], existing_players=[], set_max_team=None)

# #%%

# total_winnings, player_results, winnings_list = rs.run_multiple_lineups(params, calc_winnings=True, parallelize=True, n_jobs=15, verbose=0)
# print(total_winnings)
# display(player_results.groupby('player').size().sort_values(ascending=False))

# # %%

# %%

df = sim.player_data.copy()
qb_wr_stack = 2
qb_te_stack = 1
num_opp_players = 0
previous_lineups = []
max_teams = 3

sim.conn = rs.create_conn()
unique_matchups = sim.unique_matchup_pairs(sim.get_matchups())
matchups_df = pd.DataFrame(unique_matchups, columns=['team', 'opp'])
df = pd.merge(df, matchups_df, on='team')

df = df[df.team.isin(['NYG', 'WAS', 'ARI', 'LAR'])].reset_index(drop=True)
#%%
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
from cvxopt.glpk import ilp

# Assume df is your DataFrame

# Convert DataFrame to dictionary for easier access
players = df.to_dict('records')

# Create binary variables for each player
n = len(players)
c = [-player['pred_fp_per_game'] for player in players]  # Negative because we're maximizing

# Initialize constraint matrices
G = []  # For inequality constraints (<=)
h = []  # Right-hand side of inequality constraints
A = []  # For equality constraints (=)
b = []  # Right-hand side of equality constraints

# Constraint 1: Salary cap of 50000 (inequality: sum of salaries <= 50000)
salary_constraint = [player['salary'] for player in players]
G.append(salary_constraint)
h.append(50000)

# Constraint 2: Exact number of players per position (equality constraints)
positions = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}

for pos, count in positions.items():
    constraint = [1 if player['pos'] == pos else 0 for player in players]
    A.append(constraint)
    b.append(count)

# Constraint 3: Total number of players (equality constraint)
total_players_constraint = [1 for _ in players]
A.append(total_players_constraint)
b.append(sum(positions.values()))

# Convert constraints to matrix form
G = matrix(G, tc='d')  # This is already 1xn, so no need to transpose
h = matrix(h, tc='d')
A = matrix(A, tc='d')  # This needs to be mxn, where m is the number of constraints
b = matrix(b, tc='d')
c = matrix(c, tc='d')

# Print dimensions and content for debugging
print(f"Number of players: {n}")
print(f"Dimensions - G: {G.size}, h: {h.size}, A: {A.size}, b: {b.size}, c: {c.size}")
print("\nG matrix (Salary constraint):")
print(G)
print("\nh vector:")
print(h)
print("\nA matrix (Position constraints):")
print(A)
print("\nb vector:")
print(b)
print("\nc vector (first 10 elements):")
print(c[:10])

# Print some player data for verification
print("\nSample player data:")
for i, player in enumerate(players[:5]):  # Print first 5 players
    print(f"Player {i+1}: {player['player']} ({player['pos']}) - Salary: {player['salary']}, Predicted FP: {player['pred_fp_per_game']}")

# Set up the ILP problem
try:
    (status, x) = ilp(c, G, h, A, b, B=set(range(n)))

    # Check if a solution was found
    if status != 'optimal':
        print(f"\nOptimization failed. Status: {status}")
        print("This might indicate that no feasible solution exists with the given constraints.")
    else:
        # Extract the selected players
        selected_players = [players[i] for i in range(n) if x[i] > 0.5]

        # Print the optimal lineup
        print("\nOptimal Lineup:")
        for player in selected_players:
            print(f"{player['pos']}: {player['player']} (Predicted FP: {player['pred_fp_per_game']}, Salary: {player['salary']})")

        total_fp = sum(player['pred_fp_per_game'] for player in selected_players)
        total_salary = sum(player['salary'] for player in selected_players)
        print(f"\nTotal Predicted Fantasy Points: {total_fp}")
        print(f"Total Salary: {total_salary}")

        if total_salary <= 50000:
            print("Salary cap constraint satisfied.")
        else:
            print("Warning: Salary cap exceeded!")

except ValueError as e:
    print(f"\nAn error occurred: {e}")
    print("Please check that all required columns are present in your DataFrame and contain valid data.")
# %%
