#%%

YEAR = 2021
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


def switch_seasons(df):
    df.loc[df.week==17, 'year'] = df.loc[df.week==17, 'year'] +1
    df.loc[df.week==17, 'week'] = 1 
    return df

def forward_fill(df, cols=None):
    
    if cols is None: cols = df.columns
    df = df.sort_values(by=['player', 'year', 'week'])
    df = df.groupby('player', as_index=False)[cols].fillna(method='ffill')
    df = df.sort_values(by=['player', 'year', 'week'])

    return df

def find_bye_weeks():
    bye_week = dm.read(f'''SELECT DISTINCT team, week, year 
                           FROM FantasyPros 
                           WHERE week != 17
                           ORDER BY team, year, week''', 'Pre_PlayerData')

    bye_week = switch_seasons(bye_week)
    bye_week['next_week'] = bye_week.groupby(['team','year'])['week'].shift(-1)
    bye_week['week_gap'] = bye_week.next_week - bye_week.week
    bye_week = bye_week.loc[bye_week.week_gap==2, ['team', 'week', 'year', 'next_week']]
    bye_week['week'] = bye_week.week+1

    return bye_week.reset_index(drop=True)


def fix_bye_week(df):
    # shift forward the week before the bye week so that it lines up with
    # the week after the bye week
    bye_weeks = find_bye_weeks()
    df = pd.merge(df, bye_weeks, on=['team', 'week', 'year'], how='left')
    df.loc[~df.next_week.isnull(), 'week'] = df.loc[~df.next_week.isnull(), 'next_week']
    df = df.drop('next_week', axis=1)

    return df

#%%


def fantasy_pros(pos):

    fp = dm.read(f'''SELECT * 
                    FROM FantasyPros 
                    WHERE pos='{pos}' ''', 'Pre_PlayerData')
    fp = name_cleanup(fp)
    if pos == 'DST':
        fp = fp.drop('player', axis=1).rename(columns={'team': 'player'})
    else:
        fp = fp.drop('team', axis=1)
    fp_cols = ['fp_rank', 'projected_points']
    fp = add_rolling_stats(fp, ['player'], fp_cols)

    # fill in null expert rankings
    fp = forward_fill(fp)

    return fp



def get_salaries(df, pos):
    sal = dm.read(f'''SELECT player, week, year, 
                             dk_salary, fd_salary, yahoo_salary
                    FROM Daily_Salaries
                    WHERE Position='{pos}'
                    ''', 'Pre_PlayerData')
    df = pd.merge(df, sal, on=['player', 'week', 'year'])

    return df



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



def get_experts(df, pos):

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
                        WHERE a.position='{pos.lower()}' 
                        ''', 'Pre_PlayerData')

    experts = name_cleanup(experts)
    expert_cols = ['ProjPts', 'rushAtt', 'rushYds', 'rushTd', 'recvTargets',
                'recvReceptions', 'recvYds', 'recvTd',
                'expertConsensus', 'expertNathanJahnke',
                'expertKevinCole', 'expertAndrewErickson',
                'expertIanHartitz', 'dk_salary', 'fd_salary', 'yahoo_salary']
    experts = add_rolling_stats(experts, ['player'], expert_cols)

    # fill in null expert rankings
    experts = forward_fill(experts)
    
    for c in ['dk_salary', 'fd_salary', 'yahoo_salary', 
              'ProjPts', 'rushAtt', 'rushYds', 'rushTd', 'recvTargets',
              'recvReceptions', 'recvYds', 'recvTd']:
        experts[c] = experts[c].fillna(experts[c].min())
    
    experts = experts.fillna(experts.max())

    df = pd.merge(df, experts, on=['player', 'week', 'year'])

    return df


def get_player_data(df, pos, YEAR):

    player_data = dm.read(f'''SELECT * 
                FROM {pos}_Stats 
                WHERE season >= {YEAR-4}
                    AND week != 17''', 'FastR')
    if pos=='QB':
        rcols_player = [c for c in player_data.columns if 'pass_' in c]
    else:
        rcols_player = [c for c in player_data.columns if 'rec_' in c]

    rcols_player.extend([c for c in player_data.columns if 'rush_' in c])
    rcols_player = list(set(rcols_player))

    player_data = player_data.rename(columns={'season': 'year'})
    player_data = add_rolling_stats(player_data, gcols=['player'], rcols=rcols_player)

    y_acts = player_data.y_act
    all_cols = [c for c in player_data.columns if c != 'y_act']
    player_data = forward_fill(player_data, all_cols)
    player_data = pd.concat([player_data, y_acts], axis=1)

    player_data = player_data[player_data.year >= 2020].reset_index(drop=True)
    player_data = name_cleanup(player_data)

    player_data['week'] = player_data['week'] + 1
    player_data = switch_seasons(player_data)
    player_data = fix_bye_week(player_data)

    df = pd.merge(df, player_data, on=['player', 'week', 'year'])
    df = df[~(df.y_act.isnull()) | ((df.week==WEEK) & (df.year==YEAR))].reset_index(drop=True)

    return df


def get_team_stats(df, YEAR):
    
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

    team_stats = team_stats[team_stats.year >= 2020].reset_index(drop=True)
    team_stats['week'] = team_stats.week + 1
    team_stats = switch_seasons(team_stats)
    team_stats = fix_bye_week(team_stats)

    df = pd.merge(df, team_stats, on=['team', 'week', 'year'])

    return df


def get_coach_stats(df, YEAR):

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

    coach_stats = coach_stats[coach_stats.year >= 2020].reset_index(drop=True)
    coach_stats['week'] = coach_stats.week + 1
    coach_stats = switch_seasons(coach_stats)
    coach_stats = fix_bye_week(coach_stats)

    df = pd.merge(df, coach_stats, on=['team', 'week', 'year'])

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
    matchups = forward_fill(matchups)

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


def get_max_qb(df):
    qb_cols = ['team', 'week', 'year',
                # 'fantasyPoints',  'fantasyPointsRank', 'ProjPts',
                # 'expertConsensus', 'expertNathanJahnke',
                # 'expertKevinCole', 'expertAndrewErickson','expertIanHartitz',
                'fp_rank', 'projected_points',
                'dk_salary', 'fd_salary', 'yahoo_salary', 'pass_qb_epa_sum',
                'pass_air_yards_sum', 'pass_xyac_epa_sum',  'rush_first_down_sum',
                'rush_rush_touchdown_sum', 'rush_epa_mean', 'rush_epa_sum']

    max_qb = df.groupby(['team', 'year', 'week'], as_index=False).agg({'projected_points': 'max'})
    max_qb = pd.merge(max_qb, df, on=['team', 'year', 'week', 'projected_points'])
    max_qb = max_qb[qb_cols]
    max_qb.columns = ['qb_'+c if c not in ('team', 'week', 'year') else c for c in max_qb.columns]

    return max_qb


def add_rz_stats(df):

    rush_rz = dm.read('''SELECT * FROM PFR_Redzone_Rush''', 'Post_PlayerData')
    rec_rz = dm.read('''SELECT * FROM PFR_Redzone_Rec''', 'Post_PlayerData')

    rush_rz.player = rush_rz.player.apply(dc.name_clean)
    rec_rz.player = rec_rz.player.apply(dc.name_clean)

    rz = pd.merge(rush_rz, rec_rz, how='outer', on=['player', 'week', 'year', 'team'])
    rz = rz.drop_duplicates(subset=['player','week', 'year'])
    bye_weeks = find_bye_weeks().rename(columns={'next_week': 'bye_week'})
    bye_weeks.bye_week = bye_weeks.bye_week - 1
    rz = pd.merge(rz, bye_weeks, on=['team', 'week', 'year'], how='left')
    rz = rz[rz.week!=rz.bye_week].reset_index(drop=True).drop('bye_week', axis=1)

    for c in rz:
        if 'pct' not in c and 'rz' in c:
            rz[c] = rz[c] / rz.week

    rz = rz.fillna(0)
    rz.week = rz.week + 1
    rz = fix_bye_week(rz).drop('team', axis=1)
    rz = switch_seasons(rz)

    rz_cols = [c for c in rz.columns if 'rz' in c]
    rz = add_rolling_stats(rz, gcols=['player'], rcols=rz_cols)
    
    df = pd.merge(df, rz, on=['player', 'week', 'year'], how='left')
    df = forward_fill(df)

    rz_cols = [c for c in df.columns if 'rz' in c]
    df[rz_cols] = df[rz_cols].fillna(0)

    return df

def add_rz_stats_qb(df):

    rz = dm.read('''SELECT * FROM PFR_Redzone_Rush
                    JOIN (SELECT * FROM PFR_Redzone_Pass)
                          USING (player, week, team, year)
                    ''', 'Post_PlayerData')
    rz.player = rz.player.apply(dc.name_clean)
    rz = rz.drop('team', axis=1)
    
    for c in rz:
        if 'pct' not in c and 'rz' in c:
            rz[c] = rz[c] / rz.week

    df = pd.merge(df, rz, on=['player', 'week', 'year'], how='left')
    df = forward_fill(df)

    rz_cols = [c for c in rz.columns if 'rz' in c]
    df = add_rolling_stats(df, gcols=['player'], rcols=rz_cols)
    df = df.fillna(0)

    return df


# def add_gambling_lines(df):

#     lines = 


#%%

pos = 'QB'

df = fantasy_pros(pos); print(df.shape[0])
df = get_salaries(df, pos); print(df.shape[0])

# df = get_experts(df, pos); print(df.shape[0])
df = get_player_data(df, pos, YEAR); print(df.shape[0])
df = get_team_stats(df, YEAR); print(df.shape[0])
df = get_coach_stats(df, YEAR); print(df.shape[0])
df = add_pfr_matchup(df); print(df.shape[0])
# df = add_rz_stats_qb(df); print(df.shape[0])

# dst = add_team_matchups().drop('offTeam', axis=1)
# df = pd.merge(df, dst, on=['defTeam', 'year', 'week']); print(df.shape[0])

df = forward_fill(df)
df = df.dropna().reset_index(drop=True); print(df.shape[0])

# dm.delete_from_db('Model_Features', 'QB_Data', f"year={YEAR} AND week={WEEK}")
dm.write_to_db(df, 'Model_Features', 'QB_Data', if_exist='replace')

team_qb = get_max_qb(df)

for pos in ['RB', 'WR', 'TE']:
    
    # pre-game data
    df = fantasy_pros(pos); print(df.shape[0])
    df = get_salaries(df, pos); print(df.shape[0])
    df = add_pfr_matchup(df); print(df.shape[0])
    # df = get_experts(df, pos); print(df.shape[0])
#     if pos == 'WR': df = cb_matchups(df); print(df.shape[0])
#     if pos == 'TE': df = te_matchups(df); print(df.shape[0])
#     dst = add_team_matchups().drop('offTeam', axis=1)
#     df = pd.merge(df, dst, on=['defTeam', 'year', 'week']); print(df.shape[0])

    # post-game data
    df = get_player_data(df, pos, YEAR); print(df.shape[0])
    df = get_team_stats(df, YEAR); print(df.shape[0])
    df = get_coach_stats(df, YEAR); print(df.shape[0])
    df = add_rz_stats(df); print(df.shape[0])

    df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])

    df = forward_fill(df)
    df = df.dropna().reset_index(drop=True); print(df.shape[0])

    # dm.delete_from_db('Model_Features', f'{pos}_Data', f"year={YEAR} AND week={WEEK}")
    dm.write_to_db(df, 'Model_Features', f'{pos}_Data', if_exist='replace')


#%%

defense = dm.read(f'''SELECT * 
                      FROM Defense_Stats 
                      WHERE season>={YEAR-4}
                            AND week != 17''', 'FastR').rename(columns={'defteam': 'defTeam'})
defense['week'] = defense['week'] + 1

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

# defense = fantasy_pros(defense, 'DST')

dm.write_to_db(defense, 'Model_Features', f'Defense_Data', if_exist='replace')

df = defense.copy()
