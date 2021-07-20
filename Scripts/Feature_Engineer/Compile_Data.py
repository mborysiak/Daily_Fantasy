#%%

YEAR = 2020
WEEK = 1

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

    df = df.sort_values(by=[gcols[0], 'year', 'week']).reset_index(drop=True)

    cnt_check = df.groupby([gcols[0], 'year'])['week'].count()
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

team_stats = dm.read(f'''SELECT * FROM Team_Stats WHERE season>={YEAR-4}''', 'FastR')

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

team_stats = team_stats.rename(columns={'season': 'year'})
team_stats = add_rolling_stats(team_stats, ['team'], rcols_team)


coach_stats = dm.read(f'''SELECT * FROM Coach_Stats WHERE season>={YEAR-4}''', 'FastR')

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

coach_stats = coach_stats.rename(columns={'season': 'year'})
coach_stats = add_rolling_stats(coach_stats, ['coach'], rcols_coach)


def add_pfr_matchup(df):

    pfr_matchup = dm.read(f'''SELECT player, week, year,
                                    opp_rank, opp_fp_per_game,
                                    opp_dk_pt_per_game, opp_fd_pt_per_game,
                                    proj_fp_rank, proj_dk_rank, proj_fd_rank
                            FROM {pos}_PFR_Matchups
                            WHERE year >= 2020''', 'Pre_PlayerData')
    pfr_matchup = name_cleanup(pfr_matchup)
    df = pd.merge(df, pfr_matchup, on=['player', 'week', 'year'])

    return df



def add_experts(df, pos):

    experts = dm.read(f'''SELECT playerName player, week, a.year, a.defTeam,
                                fantasyPoints,  fantasyPointsRank,
                                `Proj Pts` ProjPts,
                                rushAtt, rushYds, rushTd, recvTargets,
                                recvReceptions, recvYds, recvTd,
                                fumbles, fumblesLost, twoPt, returnYds, returnTd,
                                expertConsensus, expertNathanJahnke,
                                expertKevinCole, expertAndrewErickson,
                                expertIanHartitz,
                                dk_salary, fd_salary, yahoo_salary
                        FROM PFF_Proj_Ranks a
                        JOIN (SELECT Name playerName, *
                                FROM PFF_Expert_Ranks 
                                WHERE Position='{pos}' )
                                USING (playerName, week, year)
                        LEFT JOIN (SELECT player playerName, week, year, 
                                          dk_salary, fd_salary, yahoo_salary
                                   FROM Daily_Salaries
                                   WHERE Position='{pos}'
                                ) USING (playerName, week, year)
                        WHERE a.position='{pos.lower()}' 
                        ''', 'Pre_PlayerData')

    experts = name_cleanup(experts)
    expert_cols = ['ProjPts', 'rushAtt', 'rushYds', 'rushTd', 'recvTargets',
                'recvReceptions', 'recvYds', 'recvTd',
                'expertConsensus', 'expertNathanJahnke',
                'expertKevinCole', 'expertAndrewErickson',
                'expertIanHartitz', 'dk_salary', 'fd_salary', 'yahoo_salary']
    experts = add_rolling_stats(experts, ['player'], expert_cols)

    df = pd.merge(df, experts, on=['player', 'week','year'])

    # fill in null expert rankings
    df = df.sort_values(by=['player', 'year', 'week'])
    df = df.groupby(['player'], as_index=False).apply(lambda group: group.ffill())
    df = df.fillna(df.max())

    return df


def cb_matchups(df):
    
    matchups = dm.read(f'''SELECT offPlayer player, week, year,
                                offHeightInches, offWeight, offSpeed,
                                offRoutes, offLeft, offSlot, offRight, 
                                offFr, offCPct, offYprr, offGrade, adv,
                                defHeightInches, defWeight, defSpeed,
                                defRoutes, defLeft, defSlot, defRight,
                                defCPct, defYprr, defGrade
                        FROM PFF_WR_CB_Matchups
    ''', 'Pre_PlayerData')
    matchups = name_cleanup(matchups)
    matchups = matchups.sort_values(by=['player', 'year', 'week']).reset_index(drop=True)
    matchups = matchups.groupby('player', as_index=False).fillna(method='ffill')

    mean_speed = matchups.offSpeed.dropna().mean()
    mean_def_speed = matchups.defSpeed.dropna().mean()

    for c, f in zip(['offSpeed', 'defSpeed', 'offGrade', 'defGrade'], 
                    [mean_speed, mean_def_speed, 60, 60]):
        matchups[c] = matchups[c].fillna(f)

    matchups = matchups.fillna(0)

    for c in ['HeightInches', 'Weight', 'Speed']:
        matchups[f'{c}differ'] = matchups[f'off{c}'] - matchups[f'def{c}']

    df = pd.merge(df, matchups, on=['player', 'week', 'year'])

    return df


def te_matchups(df):
    
    matchups = dm.read(f'''SELECT offPlayer player, week, year,
                                offHeightInches, offWeight, 
                                offRoutes,offWide, offSlot, offInline, 
                                offFr, offCPct, offYprr, offGrade, adv,
                                defHeightInches, defWeight, 
                                defRoutes, defCPct, defYprr, defGrade
                        FROM PFF_TE_Matchups
    ''', 'Pre_PlayerData')
    matchups = name_cleanup(matchups)
    matchups = matchups.sort_values(by=['player', 'year', 'week']).reset_index(drop=True)
    matchups = matchups.groupby('player', as_index=False).fillna(method='ffill')

    for c, f in zip(['offGrade', 'defGrade'], [60, 60]):
        matchups[c] = matchups[c].fillna(f)

    matchups = matchups.fillna(0)

    for c in ['HeightInches', 'Weight']:
        matchups[f'{c}differ'] = matchups[f'off{c}'] - matchups[f'def{c}']

    df = pd.merge(df, matchups, on=['player', 'week', 'year'])

    return df


def add_team_matchups():

    team_matchups = dm.read('''SELECT *
                            FROM PFF_Oline_Dline_Matchups''', 'Pre_PlayerData')
    team_matchups = team_matchups.drop(['gameTime', 'offTeam'], axis=1)

    dst = dm.read(f'''SELECT offTeam defTeam, a.defTeam AS offTeam, week, a.year,
                             fantasyPoints fantasyPoints_dst,  
                            fantasyPointsRank fantasyPointsRank_dst,
                            `Proj Pts` ProjPts_dst,
                            dstSacks, dstSafeties, dstInt,
                            dstFumblesForced, dstFumblesRecovered,
                            dstTd, dstReturnYds, dstReturnTd,
                            dstPts0, dstPts16, dstPts713, dstPts1420,
                            dstPts2127, dstPts2834, dstPts35plus,
                            expertConsensus expertConsensus_dst, 
                            expertNathanJahnke expertNathanJahnke_dst,
                            expertKevinCole expertKevinCole_dst, 
                            expertAndrewErickson expertAndrewErickson_dst,
                            expertIanHartitz expertIanHartitz_dst,
                            dk_salary dk_salary_dst, 
                            fd_salary fd_salary_dst, 
                            yahoo_salary yahoo_salary_dst
                    FROM PFF_Proj_Ranks a
                    JOIN (SELECT *
                        FROM PFF_Expert_Ranks )
                        USING (offTeam, week, year)
                    LEFT JOIN (SELECT CASE WHEN team='LV' THEN 'LVR'
                                    ELSE team END offTeam, 
                                week, year, 
                                dk_salary, fd_salary, yahoo_salary
                        FROM Daily_Salaries
                        ) USING (offTeam, week, year)
                ''', 'Pre_TeamData')

    dst = pd.merge(dst, team_matchups, on=['defTeam', 'week', 'year'])
    dst = dst.groupby(['defTeam'], as_index=False).apply(lambda group: group.ffill())

    dst = dst.fillna(dst.mean())

    return dst


def get_player_data(pos, YEAR):

    df = dm.read(f'''SELECT * 
                    FROM {pos}_Stats 
                    WHERE season >= {YEAR-4}
                        AND week != 17''', 'FastR')
    if pos=='QB':
        rcols_player = [c for c in df.columns if 'pass_' in c]
    else:
        rcols_player = [c for c in df.columns if 'rec_' in c]

    rcols_player.extend([c for c in df.columns if 'rush_' in c])
    rcols_player = list(set(rcols_player))

    df = df.rename(columns={'season': 'year'})
    df = add_rolling_stats(df, gcols=['player'], rcols=rcols_player)
    df = pd.merge(df, team_stats, on=['week', 'year', 'team'])
    df = pd.merge(df, coach_stats, on=['week', 'year', 'team'], how='left')

    df = df.dropna()
    df = df[df.year >= 2020].reset_index(drop=True)
    df = name_cleanup(df)

    df['week'] = df['week'] + 1

    return df

def get_max_qb(df):
    qb_cols = ['team', 'week', 'year',
                'fantasyPoints',  'fantasyPointsRank', 'ProjPts',
                'expertConsensus', 'expertNathanJahnke',
                'expertKevinCole', 'expertAndrewErickson','expertIanHartitz',
                'dk_salary', 'fd_salary', 'yahoo_salary', 'pass_qb_epa_sum',
                'pass_air_yards_sum', 'pass_xyac_epa_sum',  'rush_first_down_sum',
                'rush_rush_touchdown_sum', 'rush_epa_mean', 'rush_epa_sum']

    max_qb = df.groupby(['team', 'year', 'week'], as_index=False).agg({'ProjPts': 'max'})
    max_qb = pd.merge(max_qb, df, on=['team', 'year', 'week', 'ProjPts'])
    max_qb = max_qb[qb_cols]
    max_qb.columns = ['qb_'+c if c not in ('team', 'week', 'year') else c for c in max_qb.columns]

    return max_qb

#%%

pos = 'QB'
df = get_player_data(pos, YEAR); print(df.shape[0])
df = add_pfr_matchup(df); print(df.shape[0])
df = add_experts(df, pos); print(df.shape[0])
dst = add_team_matchups().drop('offTeam', axis=1)
df = pd.merge(df, dst, on=['defTeam', 'year', 'week']); print(df.shape[0])

# dm.delete_from_db('Model_Features', 'QB_Data', f"year={YEAR} AND week={WEEK}")
dm.write_to_db(df, 'Model_Features', 'QB_Data', if_exist='replace')

team_qb = get_max_qb(df)


for pos in ['RB', 'WR', 'TE']:
    df = get_player_data(pos, YEAR); print(df.shape[0])
    df = add_pfr_matchup(df); print(df.shape[0])
    df = add_experts(df, pos); print(df.shape[0])
    if pos == 'WR': df = cb_matchups(df); print(df.shape[0])
    if pos == 'TE': df = te_matchups(df); print(df.shape[0])
    dst = add_team_matchups().drop('offTeam', axis=1)

    df = pd.merge(df, dst, on=['defTeam', 'year', 'week']); print(df.shape[0])
    df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])

    df = df.sort_values(by=['team', 'year', 'week'])
    df = df.groupby('player', as_index=False).fillna(method='ffill')
    df = df.fillna(df.mean())

    df = df.sort_values(by=['player', 'year', 'week'])

    # dm.delete_from_db('Model_Features', f'{pos}_Data', f"year={YEAR} AND week={WEEK}")
    dm.write_to_db(df, 'Model_Features', f'{pos}_Data', if_exist='replace')

#%%

defense = dm.read(f'''SELECT * 
                      FROM Defense_Stats 
                      WHERE season>={YEAR-4}
                            AND week != 17''', 'FastR').rename(columns={'defteam': 'defTeam'})
defense['week'] = defense['week'] + 1

cur_abb = ['LV', 'JAX', 'LA']
new_abb = ['LVR', 'JAC', 'LAR']
for c, n in zip(cur_abb, new_abb):
    defense.loc[defense.defTeam==c, 'defTeam'] = n

rcols_def = [c for c in defense.columns if c not in ('defTeam', 'week', 'season', 'y_act')]
defense = defense.rename(columns={'season': 'year'})
defense = add_rolling_stats(defense, gcols=['defTeam'], rcols=rcols_def)
defense = defense.dropna()
defense = defense[defense.year >= 2020].reset_index(drop=True)

dst = add_team_matchups()
defense = pd.merge(defense, dst, on=['defTeam', 'year', 'week'])
team_qb = team_qb.rename(columns={'team': 'offTeam'})

defense = pd.merge(defense, team_qb, on=['offTeam', 'week', 'year'], how='left')
defense = defense.sort_values(by=['offTeam', 'year', 'week'])
defense = defense.groupby('offTeam', as_index=False).fillna(method='ffill')
defense = defense.fillna(defense.mean())

defense = defense.sort_values(by=['defTeam', 'year', 'week'])
defense = defense.copy().rename(columns={'defTeam': 'player'})
defense.columns = [c.replace('_dst', '') for c in defense.columns]

dm.write_to_db(defense, 'Model_Features', f'Defense_Data', if_exist='replace')

df = defense.copy()

# %%

from skmodel import SciKitModel 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

drop_cols = list(df.dtypes[df.dtypes=='object'].index)
# drop_cols.append('y_act')

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
X_base, y_base = skm.Xy_split(y_metric='y_act', to_drop=drop_cols)
cv_time_base = skm.cv_time_splits('week', X_base, 3)

model_base = skm.model_pipe([skm.piece('std_scale'), 
                             skm.piece('k_best'),
                             skm.piece('lr')])

params = skm.default_params(model_base)
params['k_best__k'] = range(1, X_base.shape[1])

best_model_base = skm.random_search(model_base, X_base, y_base, params, cv=cv_time_base, n_iter=50)
_, _ = skm.val_scores(best_model_base, X_base, y_base, cv_time_base)

imp_cols = X_base.columns[best_model_base['k_best'].get_support()]
skm.print_coef(best_model_base, imp_cols)

print('------------------')

baseline_m['y_act'] = np.where((baseline_m.y_act > np.percentile(df.y_act, 90)), 1, 0)                 

skm = SciKitModel(baseline_m, 'class')
X_base_class, y_base_class = skm.Xy_split(y_metric='y_act', 
                                          to_drop=drop_cols)
cv_time = skm.cv_time_splits('week', X_base_class, 3)

model_base_class = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('k_best_c'),
                        skm.piece('lr_c')])

params = skm.default_params(model_base_class)
best_model_base_class = skm.random_search(model_base_class, X_base_class, y_base_class, params, cv=cv_time, n_iter=50)
_, _ = skm.val_scores(best_model_base_class, X_base_class, y_base_class, cv_time)

imp_cols = X_base_class.columns[best_model_base_class['k_best_c'].get_support()]
skm.print_coef(best_model_base_class, imp_cols)


#%%

df_m = df.copy()
df_m = df_m.sort_values(by='week')

skm = SciKitModel(df_m)
X_all_reg, y_all_reg = skm.Xy_split(y_metric='y_act', 
                                    to_drop=list(df.dtypes[df.dtypes=='object'].index))
cv_time = skm.cv_time_splits('week', X_all_reg, 3)

model_all_reg = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('select_perc'),
                        skm.feature_union([
                                           skm.piece('agglomeration'), 
                                           skm.piece('k_best'), 
                                           skm.piece('pca')
                                           ]),
                        skm.piece('k_best', label_rename='k_best2'),
                        skm.piece('ridge')])

params = skm.default_params(model_all_reg)

best_model_all_reg = skm.random_search(model_all_reg, X_all_reg, y_all_reg, params, cv=cv_time, n_iter=50)
_, _ = skm.val_scores(best_model_all_reg, X_all_reg, y_all_reg, cv_time)

try:
    imp_cols = X_all_reg.columns[best_model_all_reg['k_best2'].get_support()]
    skm.print_coef(best_model_all_reg, imp_cols)
except:
    pass

#%%

df_m = df.copy().sort_values(by='week')
df_m['y_act'] = np.where((df_m.y_act > np.percentile(df.y_act, 90)), 1, 0)                 

skm = SciKitModel(df_m, 'class')
X_all_class, y_all_class = skm.Xy_split(y_metric='y_act', 
                                        to_drop=list(df.dtypes[df.dtypes=='object'].index))
cv_time = skm.cv_time_splits('week', X_all_class, 3)

model_all_class = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('select_perc_c'),
                        skm.feature_union([
                                           skm.piece('agglomeration'), 
                                           skm.piece('k_best_c')
                                           ]),
                        skm.piece('k_best_c', label_rename='k_best2'),
                        skm.piece('lr_c')])

params = skm.default_params(model_all_class)

best_model_all_class = skm.random_search(model_all_class, X_all_class, y_all_class, params, cv=cv_time, n_iter=50)
_, _ = skm.val_scores(best_model_all_class, X_all_class, y_all_class, cv_time)

try:
    imp_cols = X_all_class.columns[best_model_all_class['k_best2'].get_support()]
    skm.print_coef(best_model_all_class, imp_cols)
except:
    pass
# %%
