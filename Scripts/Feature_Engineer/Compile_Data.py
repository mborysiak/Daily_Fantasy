#%%

YEAR = 2020

#%%
import pandas as pd 
import pyarrow.parquet as pq
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

pd.set_option('display.max_columns', 999)

def name_cleanup(df):
    df.player = df.player.apply(dc.name_clean)
    return df

def rolling_stats(df, gcols, rcols, period, agg_type='mean'):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    rolls = df.groupby(gcols)[rcols].rolling(3).agg(agg_type).reset_index(drop=True)
    rolls.columns = [f'r{agg_type}{period}_{c}' for c in rolls.columns]

    return rolls


def rolling_expand(df, gcols, rcols, agg_type='mean'):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    # if agg type is in form of percentile (e.g. p80) then use quantile
    if agg_type[0]=='p':

        # pull out perc amount and convert to decimal float to calculate quantile rolling
        perc_amt = float(agg_type[1:])/100
        rolls =  df.groupby(gcols)[rcols].apply(lambda x: x.expanding().quantile(perc_amt))
    # otherwise, use the string argument of aggregation
    else:
        rolls = df.groupby(gcols)[rcols].apply(lambda x: x.expanding().agg(agg_type))
    
    # clean up the rolled dataset indices and column name prior to returning 
    rolls = rolls.reset_index(drop=True)
    rolls.columns = [f'{agg_type}all_{c}' for c in rolls.columns]

    return rolls


def add_rolling_stats(df, gcols, rcols, ):

    df = df.sort_values(by=[gcols[0], 'season', 'week']).reset_index(drop=True)

    cnt_check = df.groupby([gcols[0], 'season'])['week'].count()
    print(f'Counts of Groupby Category Over 17: {cnt_check[cnt_check>17]}')

    print('Calculating Rolling Stats 3 window')
    rolls_mean = rolling_stats(df, gcols, rcols, 3, agg_type='mean')
    rolls_max = rolling_stats(df, gcols, rcols, 3, agg_type='max')
    rolls_med = rolling_stats(df, gcols, rcols, 3, agg_type='median')

    print('Calculating Expanding Stats')
    hist_mean = rolling_expand(df, gcols, rcols, agg_type='mean')
    hist_std = rolling_expand(df, gcols, rcols, agg_type='std')
    hist_p80 = rolling_expand(df, gcols, rcols, agg_type='p80')
    hist_p20 = rolling_expand(df, gcols, rcols, agg_type='p20')

    df = pd.concat([df, 
                    hist_mean, hist_std, hist_p80, hist_p20, 
                    rolls_mean, rolls_max, rolls_med], axis=1)

    return df


#%%

#--------------
# Rolling Stats
#--------------

team_stats = dm.read(f'''SELECT * FROM Team_Stats WHERE season>={YEAR-3}''', 'FastR')

rcols_team = ['team_rush_touchdown_sum', 'team_tackled_for_loss_sum',
         'team_pass_touchdown_sum', 'team_qb_hit_sum', 'team_sack_sum',
         'team_qb_epa_sum', 'team_cp_sum', 'team_cpoe_sum', 'team_air_epa_sum',
         'team_air_wpa_sum', 'team_air_yards_sum', 'team_comp_air_epa_sum',
         'team_comp_air_wpa_sum', 'team_comp_yac_epa_sum',
         'team_comp_yac_wpa_sum', 'team_complete_pass_sum',
         'team_incomplete_pass_sum', 'team_interception_sum', 'team_ep_sum',
         'team_epa_sum', 'team_touchdown_sum', 'team_fumble_sum',
         'team_fumble_lost_sum', 'team_xyac_epa_sum',
]

team_stats = add_rolling_stats(team_stats, ['team'], rcols_team)


coach_stats = dm.read(f'''SELECT * FROM Coach_Stats WHERE season>={YEAR-3}''', 'FastR')

rcols_coach = ['coach_shotgun_sum',
       'coach_no_huddle_sum', 'coach_rush_attempt_sum', 'coach_first_down_sum',
       'coach_first_down_rush_sum', 'coach_fourth_down_converted_sum',
       'coach_fourth_down_failed_sum', 'coach_third_down_converted_sum',
       'coach_goal_to_go_sum', 'coach_run_middle_sum', 'coach_run_outside_sum',
       'coach_rush_touchdown_sum', 'coach_tackled_for_loss_sum',
       'coach_ep_sum', 'coach_epa_sum', 'coach_touchdown_sum',
       'coach_fumble_sum', 'coach_yardline_100_sum',
       'coach_yards_after_catch_sum', 'coach_yards_gained_sum',
       'coach_ydstogo_sum', 'coach_td_prob_mean', 'coach_wp_mean',
       'coach_wpa_mean', 'coach_ep_mean', 'coach_epa_mean',
       'coach_yardline_100_mean', 'coach_yards_gained_mean',
       'coach_ydstogo_mean']

coach_stats = add_rolling_stats(coach_stats, ['coach'], rcols_coach)


#%%

pos = 'WR'

df = dm.read(f'''SELECT * 
                 FROM {pos}_Stats 
                 WHERE season >= {YEAR-3}
                       AND week != 17''', 'FastR')

rcols_player = [c for c in df.columns if 'rec_' in c]
rcols_player.extend([c for c in df.columns if 'rush_' in c])
rcols = list(set(rcols))

df = add_rolling_stats(df, gcols=['player'], rcols=rcols_player)
df = pd.merge(df, team_stats, on=['week', 'season', 'team'])
df = pd.merge(df, coach_stats, on=['week', 'season', 'team'], how='left')

df = df.dropna()
df = df[df.season >= 2020].reset_index(drop=True)
df = df.rename(columns={'season': 'year'})
df = name_cleanup(df)

# %%

pfr_matchup = dm.read(f'''SELECT player, week, year,
                                 opp_rank, opp_fp_per_game,
                                 opp_dk_pt_per_game, opp_fd_pt_per_game,
                                 proj_fp_rank, proj_dk_rank, proj_fd_rank
                          FROM {pos}_PFR_Matchups
                          WHERE year >= 2020''', 'Pre_PlayerData')
pfr_matchup = name_cleanup(pfr_matchup)
df = pd.merge(df, pfr_matchup, on=['player', 'week', 'year'])

# %%

experts = dm.read(f'''SELECT playerName player, week, a.year,
                             fantasyPoints,  fantasyPointsRank,
                             `Proj Pts` ProjPts,
                             expertConsensus, expertNathanJahnke,
                             expertKevinCole, expertAndrewErickson,
                             expertIanHartitz,
                             dk_salary, fd_salary, yahoo_salary
                      FROM PFF_Proj_Ranks a
                      JOIN (SELECT Name playerName, *
                            FROM PFF_Expert_Ranks )
                            USING (playerName, week, year)
                      JOIN (SELECT player playerName, week, year, 
                                   dk_salary, fd_salary, yahoo_salary
                            FROM Daily_Salaries
                            ) USING (playerName, week, year)
                      WHERE a.position='{pos.lower()}' 
                      ''', 'Pre_PlayerData')

experts = name_cleanup(experts)
df = pd.merge(df, experts, on=['player', 'week','year'])

# fill in null expert rankings
df = df.sort_values(by=['player', 'year', 'week'])
df = df.groupby(['player'], as_index=False).apply(lambda group: group.ffill())
df = df.fillna(df.max())

#%%

matchups = dm.read('''SELECT offPlayer player,
                             offHeighInches, offWeight, offSpeed


''')

# %%

from skmodel import SciKitModel 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

#-----------------
# Run Baseline Model
#-----------------

baseline_m = df.loc[: , ['player', 'week', 'year',
                         'fantasyPoints',  'fantasyPointsRank', 'ProjPts',
                         'expertConsensus', 'expertNathanJahnke',
                         'expertKevinCole', 'expertAndrewErickson','expertIanHartitz',
                         'dk_salary', 'fd_salary', 'yahoo_salary', 
                         'y_act']].copy()

baseline_m = baseline_m.sort_values(by='week')

skm = SciKitModel(baseline_m)
X_base, y_base = skm.Xy_split(y_metric='y_act', to_drop=['player'])
cv_time_base = skm.cv_time_splits('week', X_base, 3)

model_base = skm.model_pipe([skm.piece('std_scale'), 
                             skm.piece('k_best'),
                             skm.piece('ridge')])

params = skm.default_params(model_base)
params['k_best__k'] = range(1, X_base.shape[1])

best_model_base = skm.random_search(model_base, X_base, y_base, params, cv=cv_time_base, n_iter=50)
_, _ = skm.val_scores(best_model_base, X_base, y_base, cv_time_base)

imp_cols = X_base.columns[best_model_base['k_best'].get_support()]
skm.print_coef(best_model_base, imp_cols)

base_predict = skm.cv_predict_time(best_model_base, X_base, y_base, cv_time_base)
base_predict = pd.Series(base_predict, name='base_predict')

pred_labels = skm.return_labels(['player', 'week'], 'time').reset_index(drop=True)
pred_labels = pd.concat([pred_labels, base_predict], axis=1)

pred_labels = pd.merge(pred_labels, baseline, on=['player', 'week'])


#%%

df_m = df.copy()
df_m = df_m.sort_values(by='week')

skm = SciKitModel(df_m)
X, y = skm.Xy_split(y_metric='y_act', to_drop=['player', 'position', 'coach', 'team'])
cv_time = skm.cv_time_splits('week', X, 3)

model = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('select_perc'),
                        skm.feature_union([
                                           skm.piece('agglomeration'), 
                                           skm.piece('k_best'), 
                                           skm.piece('pca')
                                           ]),
                        skm.piece('k_best', label_rename='k_best2'),
                        skm.piece('ridge')])

params = skm.default_params(model)
best_model = skm.random_search(model, X, y, params, cv=cv_time, n_iter=50)
_, _ = skm.val_scores(best_model, X, y, cv_time)

try:
    imp_cols = X.columns[best_model['k_best2'].get_support()]
    skm.print_coef(best_model, imp_cols)
except:
    pass

#%%

df_m = df.copy().sort_values(by='week')

df_m['y_act'] = np.where(#(df_m.y_act > df_m.base_predict * 1.5) & \
                             (df_m.y_act > 26), 1, 0)                 
# df_m['y_act'] = np.where((df_m.y_act > df_m.ProjPts * 1.15) & (df_m.y_act > 15), 1, 0)                 

skm = SciKitModel(df_m, 'class')
X, y = skm.Xy_split(y_metric='y_act', to_drop=['player', 'position', 'coach', 'team'])
cv_time = skm.cv_time_splits('week', X, 3)

model = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('select_perc_c'),
                        skm.feature_union([
                                           skm.piece('agglomeration'), 
                                           skm.piece('k_best_c')
                                           ]),
                        skm.piece('k_best_c', label_rename='k_best2'),
                        skm.piece('lr_c')])

params = skm.default_params(model)
best_model = skm.random_search(model, X, y, params, cv=cv_time, n_iter=50)
_, _ = skm.val_scores(best_model, X, y, cv_time)
# %%
