#%%

YEAR = 2021
WEEK = 4

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

    rolls_mean = rolling_stats(df, gcols, rcols, 3, agg_type='mean')
    rolls_max = rolling_stats(df, gcols, rcols, 3, agg_type='max')
    # rolls_med = rolling_stats(df, gcols, rcols, 3, agg_type='median')

    hist_mean = rolling_expand(df, gcols, rcols, agg_type='mean')
    hist_std = rolling_expand(df, gcols, rcols, agg_type='std')
    hist_p80 = rolling_expand(df, gcols, rcols, agg_type='p95')
    # hist_p20 = rolling_expand(df, gcols, rcols, agg_type='p20')

    df = pd.concat([df, 
                    hist_mean, hist_std, hist_p80,# hist_p20, 
                    rolls_mean, rolls_max,# rolls_med
                    ], axis=1)

    return df


def switch_seasons(df):
    df.loc[df.week==17, 'year'] = df.loc[df.week==17, 'year'] + 1
    df.loc[df.week==17, 'week'] = 1 
    return df

def forward_fill(df, cols=None):
    
    if cols is None: cols = df.columns
    df = df.sort_values(by=['player', 'year', 'week'])
    df = df.groupby('player', as_index=False)[cols].fillna(method='ffill')
    df = df.sort_values(by=['player', 'year', 'week'])

    return df

def find_bye_weeks():

    return dm.read(f'''SELECT team, week, year, week+1 next_week
                       FROM (
                            SELECT team, year, week, 
                                    rank() OVER(PARTITION BY team, year 
                                                ORDER BY cnts DESC) ranks
                            FROM (
                                    SELECT offTeam team, year, bye week, COUNT(bye) cnts
                                    FROM PFF_Expert_Ranks
                                    GROUP BY team, year, bye
                                    )
                       ) WHERE ranks=1
                       ''', 'Pre_PlayerData')


def fix_bye_week(df):
    # shift forward the week before the bye week so that it lines up with
    # the week after the bye week
    bye_weeks = find_bye_weeks()
    df = pd.merge(df, bye_weeks, on=['team', 'week', 'year'], how='left')
    df.loc[~df.next_week.isnull(), 'week'] = df.loc[~df.next_week.isnull(), 'next_week']
    df = df.drop('next_week', axis=1)

    return df

def drop_extra_bye_week(df):
    # the data is duplicated into the bye week, so
    # this chunk of code removes the duplicated week where games
    # weren't actually played
    df = df.drop_duplicates(subset=['player','week', 'year'])
    bye_weeks = find_bye_weeks().rename(columns={'next_week': 'bye_week'})
    bye_weeks.bye_week = bye_weeks.bye_week - 1
    df = pd.merge(df, bye_weeks, on=['team', 'week', 'year'], how='left')
    df = df[df.week!=df.bye_week].reset_index(drop=True).drop('bye_week', axis=1)

    return df

#%%


#---------------
# Pre Game Data
#---------------

def fantasy_pros(pos, add_rolling=True):

    fp = dm.read(f'''SELECT * 
                    FROM FantasyPros 
                    WHERE pos='{pos}' ''', 'Pre_PlayerData')
    fp = name_cleanup(fp)
    if pos == 'DST': fp = fp.drop('player', axis=1).rename(columns={'team': 'player'})
    
    fp_cols = ['fp_rank', 'projected_points']
    if add_rolling: fp = add_rolling_stats(fp, ['player'], fp_cols)

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



def get_experts(df, pos, add_rolling=True):

    experts = dm.read(f'''SELECT player, week, year, a.defTeam,
                            fantasyPoints,  fantasyPointsRank,
                            `Proj Pts` ProjPts,
                            passComp, passAtt, passYds, passTd, passInt, passSacked,
                            rushAtt, rushYds, rushTd, recvTargets,
                            recvReceptions, recvYds, recvTd,
                            fumbles, fumblesLost, twoPt, returnYds, returnTd,
                            expertConsensus, expertNathanJahnke,
                            expertKevinCole, expertAndrewErickson,
                            expertIanHartitz,expertDwainMcFarland, 
                            expertBenBrown, expertJaradEvans
                    FROM PFF_Proj_Ranks a
                    JOIN (SELECT *
                            FROM PFF_Expert_Ranks 
                            WHERE Position='{pos}' )
                            USING (player, week, year)
                    ''', 'Pre_PlayerData')

    

    experts = name_cleanup(experts)
    expert_cols = ['ProjPts','passYds', 'passTd', 'rushAtt', 
                    'rushYds', 'rushTd', 'recvTargets',
                    'recvReceptions', 'recvYds', 'recvTd',
                    'expertConsensus', 'expertNathanJahnke',
                    'expertKevinCole', 'expertAndrewErickson',
                    'expertIanHartitz', 'expertDwainMcFarland', 
                    'expertBenBrown', 'expertJaradEvans']

    # fill in null expert rankings
    experts = forward_fill(experts)

    if add_rolling: experts = add_rolling_stats(experts, ['player'], expert_cols)

    for c in ['ProjPts', 'rushAtt', 'rushYds', 'rushTd', 'recvTargets',
                'recvReceptions', 'recvYds', 'recvTd']:
        experts[c] = experts[c].fillna(experts[c].min())

    experts = experts.fillna(experts.max())

    df = pd.merge(df, experts, on=['player', 'week', 'year'])

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


def positional_values():

    fp = dm.read('''SELECT player, pos, team, week, year, fp_rank, projected_points
                        FROM FantasyPros
                        WHERE pos!='QB' ''', "Pre_PlayerData")

    dk_sal = dm.read('''SELECT player, team, week, year, dk_salary
                        FROM Daily_Salaries''', "Pre_PlayerData")

    pff = dm.read('''SELECT player, week, year, expertConsensus, fantasyPoints, `Proj Pts` ProjPts
                     FROM PFF_Expert_Ranks
                     JOIN (SELECT player, week, year, fantasyPoints
                           FROM PFF_Proj_Ranks)
                           USING (player, week, year) ''', "Pre_PlayerData")

    data = pd.merge(fp, dk_sal, on=['player', 'team', 'week', 'year'], how='inner')
    data = pd.merge(data, pff, on=['player', 'week', 'year'], how='inner')

    data['log_expertConsensus'] = data.expertConsensus.apply(lambda x: 5 - np.log(x))
    data['log_fp_rank'] = data.fp_rank.apply(lambda x: 5 - np.log(x))

    data = forward_fill(data).dropna()

    data = dc.convert_to_float(data)
    cols = ['log_fp_rank', 'dk_salary', 'log_expertConsensus', 
            'projected_points', 'fantasyPoints', 'ProjPts']

    all_stats = data[['team','week','year']].drop_duplicates()
    for ag in ['sum', 'mean', 'max']:
        to_agg = {c: ag for c in cols}
        team_stats = data.groupby(['team', 'week', 'year']).agg(to_agg)
        team_stats.columns = [f'team_{c}_{ag}' for c in team_stats.columns]
        all_stats = pd.merge(all_stats, team_stats, on=['team','week','year'])
    
    return all_stats


def get_max_qb():

    fp = dm.read('''SELECT player, pos, team, week, year, fp_rank, projected_points
                        FROM FantasyPros
                        WHERE pos='QB' ''', "Pre_PlayerData")

    dk_sal = dm.read('''SELECT player, team, week, year, dk_salary
                        FROM Daily_Salaries''', "Pre_PlayerData")

    pff = dm.read('''SELECT player, week, year, expertConsensus, fantasyPoints, `Proj Pts` ProjPts
                     FROM PFF_Expert_Ranks
                     JOIN (SELECT player, week, year, fantasyPoints
                           FROM PFF_Proj_Ranks)
                           USING (player, week, year) ''', "Pre_PlayerData")

    df = pd.merge(fp, dk_sal, on=['player', 'team', 'week', 'year'], how='inner')
    df = pd.merge(df, pff, on=['player', 'week', 'year'], how='inner')

    qb_cols = ['team', 'week', 'year', 'fp_rank', 'projected_points',
                'fantasyPoints', 'ProjPts',
                'expertConsensus', 'dk_salary']

    max_qb = df.groupby(['team', 'year', 'week'], as_index=False).agg({'projected_points': 'max'})
    max_qb = pd.merge(max_qb, df, on=['team', 'year', 'week', 'projected_points'])
    max_qb = max_qb[qb_cols]
    max_qb.columns = ['qb_'+c if c not in ('team', 'week', 'year') else c for c in max_qb.columns]

    return max_qb


#---------------
# Post Game Data
#---------------

def get_player_data(df, pos, YEAR):

    player_data = dm.read(f'''SELECT * 
                FROM {pos}_Stats 
                WHERE season >= {YEAR-2}
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

    # players who miss a week have their stats lined up with the following week and not the
    # week that they return. Eg. a player plays week 10 and then missed untils 14. His week 10
    # stats are set to week 11, rather than week 14. This code block realigns the stats
    player_data['shift_week'] = player_data.week.shift(-1)
    player_data['week_delta'] = player_data['shift_week'] - player_data['week']
    # pull in the bye weeks and join so that the missed week calculation can remove byes
    byes = find_bye_weeks().rename(columns={'next_week': 'shift_week', 'week': 'bye_week'})
    player_data = pd.merge(player_data, byes, on=['team', 'shift_week', 'year'], how='left')
    
    # filter the data to locate weeks where players missed time in between
    filter_expr = '''(player_data.week + 1 != player_data.bye_week) & \
                     (player_data.week_delta > 1) & \
                     ~((player_data.week==WEEK) & (player_data.year==YEAR))'''
    # calculate the weeks missed prior to adjusting the last week played data to next week played data
    player_data['weeks_missed'] = 0
    player_data.loc[eval(filter_expr), 'weeks_missed'] = player_data.loc[eval(filter_expr), 'week_delta'] - 1
    player_data.loc[eval(filter_expr), 'week'] = player_data.loc[eval(filter_expr), 'shift_week'] - 1
    player_data = player_data.drop(['shift_week', 'week_delta', 'bye_week'], axis=1)

    # merge player data back to the main dataframe
    df = pd.merge(df, player_data, on=['player', 'team', 'week', 'year'])
    df = df[~(df.y_act.isnull()) | ((df.week==WEEK) & (df.year==YEAR))].reset_index(drop=True)

    return df


def calc_market_share(df):
    
    player_cols = ['rush_rush_attempt_sum', 'rec_xyac_mean_yardage_sum', 'rush_yards_gained_sum',
                   'rec_yards_gained_sum', 'rec_xyac_epa_sum', 'rush_ep_sum',
                   'rush_touchdown_sum', 'rec_first_down_sum', 'rec_epa_sum', 'rec_qb_epa_sum',
                   'rec_pass_attempt_sum', 'rec_xyac_success_sum', 'rec_comp_air_epa_sum',
                   'rec_air_yards_sum', 'rec_touchdown_sum']
    team_cols = ['team_' + c for c in player_cols]

    for p, t in zip(player_cols, team_cols):
        df[p+'_share'] = df[p] / (df[t]+0.5)

    share_cols = [c+'_share' for c in player_cols]
    df[share_cols] = df[share_cols].fillna(0)
    df = add_rolling_stats(df, gcols=['player'], rcols=share_cols)

    df = forward_fill(df)
    share_cols = [c for c in df.columns if 'share' in c]
    df[share_cols] = df[share_cols].fillna(0)

    return df


def add_player_comparison(df, cols):
    
    to_agg = {c: [np.mean, np.max, np.min] for c in cols}
    team_stats = df.groupby(['team', 'week', 'year']).agg(to_agg)

    diff_df = df[['player', 'team', 'week', 'year']].drop_duplicates()
    for c in cols:
        tmp_df = team_stats[c].reset_index()
        tmp_df = pd.merge(tmp_df, df[['player', 'team', 'week', 'year', c]], on=['team', 'week', 'year'])

        for a in ['mean', 'amin', 'amax']:
            tmp_df[f'{c}_{a}_diff'] = tmp_df[c] - tmp_df[a]
    
        tmp_df = tmp_df[['player', 'team', 'week', 'year', f'{c}_mean_diff', f'{c}_amax_diff', f'{c}_amin_diff']]
        diff_df = pd.merge(diff_df, tmp_df, on=['player', 'team', 'week', 'year'])
        
    diff_df = diff_df.drop_duplicates()
    team_stats.columns = [f'{c[0]}_{c[1]}' for c in team_stats.columns]
    team_stats = team_stats.reset_index().drop_duplicates()

    df = pd.merge(df, team_stats, on=['team', 'week', 'year'])
    df = pd.merge(df, diff_df, on=['player', 'team', 'week', 'year'])

    return df


def get_team_stats(YEAR, prev_years=2):
    
    team_stats = dm.read(f'''SELECT * FROM Team_Stats WHERE season>={YEAR-prev_years}''', 'FastR')

    rcols_team = [c for c in team_stats.columns if 'rush_' in c or 'rec_' in c]

    team_stats = team_stats.rename(columns={'season': 'year'})
    team_stats = add_rolling_stats(team_stats, ['team'], rcols_team)

    team_stats = team_stats[team_stats.year >= 2020].reset_index(drop=True)
    team_stats['week'] = team_stats.week + 1
    team_stats = switch_seasons(team_stats)
    team_stats = fix_bye_week(team_stats)

    return team_stats


def get_coach_stats(df, YEAR, add_rolling=False):

    coach_stats = dm.read(f'''SELECT * FROM Coach_Stats WHERE season>={YEAR-2}''', 'FastR')

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
    if add_rolling: coach_stats = add_rolling_stats(coach_stats, ['coach'], rcols_coach)

    coach_stats = coach_stats[coach_stats.year >= 2020].reset_index(drop=True)
    coach_stats['week'] = coach_stats.week + 1
    coach_stats = switch_seasons(coach_stats)
    coach_stats = fix_bye_week(coach_stats)

    df = pd.merge(df, coach_stats, on=['team', 'week', 'year'])

    return df


def add_rz_stats(df):

    rush_rz = dm.read('''SELECT * FROM PFR_Redzone_Rush''', 'Post_PlayerData')
    rec_rz = dm.read('''SELECT * FROM PFR_Redzone_Rec''', 'Post_PlayerData')

    rush_rz.player = rush_rz.player.apply(dc.name_clean)
    rec_rz.player = rec_rz.player.apply(dc.name_clean)

    rz = pd.merge(rush_rz, rec_rz, on=['player', 'team', 'week', 'year'])

    # the red zone data is duplicated into the bye week, so
    # this chunk of code removes the duplicated week where games
    # weren't actually played
    rz = drop_extra_bye_week(rz)

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

    rz = dm.read('''SELECT * FROM PFR_Redzone_Pass
                    LEFT JOIN (SELECT * FROM PFR_Redzone_Rush)
                               USING (player, week, team, year)''', 'Post_PlayerData')

    rz = drop_extra_bye_week(rz)

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


def add_gambling_lines(df):

    lines = dm.read("SELECT * FROM Gambling_Lines", 'Pre_TeamData')

    away = lines[['away_team', 'away_line', 'away_moneyline', 'over_under', 'week', 'year']]
    home = lines[['home_team', 'home_line', 'home_moneyline', 'over_under', 'week', 'year']]
    home = home.assign(is_home=1)
    away = away.assign(is_home=1)
    away.columns = ['team', 'line', 'moneyline', 'over_under', 'week', 'year', 'is_home']
    home.columns = ['team', 'line', 'moneyline', 'over_under', 'week', 'year', 'is_home']

    lines = pd.concat([home, away], axis=0)
    lines = dc.convert_to_float(lines)
    lines['implied_points_for'] = (lines.over_under / 2) + (lines.line / 2) 
    lines['implied_points_against'] = (lines.over_under / 2) - (lines.line / 2) 

    df = pd.merge(df, lines, on=['team', 'week', 'year'], how='left')

    return df


def add_weather(df):

    weather = dm.read('''SELECT * FROM Game_Weather''', 'Pre_TeamData')
    weather = weather.drop('gametime_unix', axis=1)

    for p in ['rain', 'snow']:
        weather[p+'_amount'] = 0
        weather.loc[weather.precip_type == p, p + '_amount'] = weather.loc[weather.precip_type == p, 'precip_intensity'] * weather.loc[weather.precip_type ==p, 'precip_prob']

    def heat_index(row):
        T = row[0]
        RH = row[1]
        return -42.379 + 2.04901523*T + 10.14333127*RH - .22475541*T*RH - .00683783*T*T - \
             .05481717*RH*RH + .00122874*T*T*RH + .00085282*T*RH*RH - .00000199*T*T*RH*RH

    weather['heat_index'] = weather[['temp_high', 'humidity']].apply(heat_index, axis=1)
    weather['wind_chill'] = weather.temp_low - (weather.wind_speed * 0.7)

    col_order = ['team', 'week', 'year', 'precip_prob', 'precip_intensity', 'temp_high', 'temp_low',
                'humidity', 'wind_speed', 'wind_gust', 
                'uv_index', 'rain_amount', 'snow_amount', 'heat_index', 'wind_chill', 'is_dome']

    domes = ['LVR', 'NO', 'DAL', 'DET', 'IND', 'ATL', 'HOU', 'LAR', 'LAC', 'ARI', 'MIN']
    weather['is_dome'] = 0
    weather.loc[weather.team.isin(domes), 'is_dome'] = 1
    weather = weather[col_order]

    for c in ['temp_high', 'temp_low', 'heat_index', 'wind_chill']:
        weather.loc[weather.team.isin(domes), c] = 70

    for c in ['rain_amount', 'snow_amount', 'precip_prob', 'precip_intensity', 'wind_speed', 'wind_gust', 'uv_index']:
        weather.loc[weather.team.isin(domes), c] = 0

    matchups = dm.read('''SELECT offTeam team, defTeam, week, year 
                        FROM PFF_Expert_Ranks''', 'Pre_TeamData')

    away_weather = pd.merge(weather, matchups, on=['team', 'week', 'year'])
    away_weather = away_weather.drop('team', axis=1).rename(columns={'defTeam': 'team'})
    away_weather = away_weather[col_order]

    weather = pd.concat([weather, away_weather], axis=0).sort_values(by=['year', 'week'])

    df = pd.merge(df, weather, on=['team', 'week', 'year'])

    return df


def advanced_rb_stats(df):

    adv = dm.read('''SELECT * FROM PFR_Advanced_RB''', 'Post_PlayerData')
    adv = adv[adv.position.isin(['RB', 'rb', 'Rb'])].drop(['rank', 'position'], axis=1)
    adv = dc.convert_to_float(adv)
    adv.player = adv.player.apply(dc.name_clean)
    
    adv['yds_before_contact_pct'] = adv['yds_before_contact'] / adv['rush_yds']
    adv['yds_after_contact_pct'] = adv['yds_after_contact'] / adv['rush_yds']
    adv['yds_before_contact_per_game'] = adv['yds_before_contact'] / adv['games']
    adv['yds_after_contact_per_game'] = adv['yds_after_contact'] / adv['games']
    adv['broke_tackles_per_game'] = adv['broke_tackles'] / adv['games']

    adv = adv[['player', 'team', 'week', 'year', 'yds_before_contact_per_game', 'yds_before_contact_att', 'yds_after_contact_per_game',
                'yac_att', 'broke_tackles_per_game', 'att_broken', 'yds_before_contact_pct', 'yds_after_contact_pct']]

    adv = drop_extra_bye_week(adv)
    adv.week = adv.week + 1
    adv = fix_bye_week(adv).drop('team', axis=1)
    adv = switch_seasons(adv)

    df = pd.merge(df, adv, on=['player', 'week', 'year'], how='left')
    df = forward_fill(df)

    adv_cols = ['yds_before_contact_per_game', 'yds_before_contact_att', 'yds_after_contact_per_game',
                'yac_att', 'broke_tackles_per_game', 'att_broken', 'yds_before_contact_pct', 'yds_after_contact_pct']
    df[adv_cols] = df[adv_cols].fillna(0)

    return df



def advanced_rec_stats(df):

    adv = dm.read('''SELECT * FROM PFR_Advanced_WR''', 'Post_PlayerData')
    adv = dc.convert_to_float(adv)
    adv.player = adv.player.apply(dc.name_clean)
    
    adv['first_downs_per_target'] = adv['first_downs'] / adv['targets']
    adv['yards_before_catch_per_game'] = adv['yds_before_catch'] / adv['games']
    adv['yards_after_catch_per_game'] = adv['yards_after_catch'] / adv['games']
    adv['yards_before_catch_per_target'] = adv['yds_before_catch'] / adv['targets']
    adv['yards_after_catch_per_target'] = adv['yards_after_catch'] / adv['targets']
    adv['broken_tackles_per_game'] = adv['broken_tackles'] / adv['targets']
    
    adv = adv[['player', 'team', 'week', 'year', 'first_downs_per_target', 'yards_before_catch_per_game',
               'yards_after_catch_per_game', 'yards_before_catch_per_target', 'yards_after_catch_per_target',
               'yds_before_catch_per_rec', 'yards_after_catch_per_rec', 'rec_per_broken', 'drop_pct',
               'broken_tackles_per_game']]

    adv = drop_extra_bye_week(adv)
    adv.week = adv.week + 1
    adv = fix_bye_week(adv)
    adv = switch_seasons(adv)

    df = pd.merge(df, adv, on=['player', 'team', 'week', 'year'], how='left')
    df = forward_fill(df)

    adv_cols = ['player', 'team', 'week', 'year', 'first_downs_per_target', 'yards_before_catch_per_game',
               'yards_after_catch_per_game', 'yards_before_catch_per_target', 'yards_after_catch_per_target',
               'yds_before_catch_per_rec', 'yards_after_catch_per_rec', 'rec_per_broken', 'drop_pct',
               'broken_tackles_per_game']
    df[adv_cols] = df[adv_cols].fillna(0)

    return df


def get_defense_stats(prev_years=2):
    d_stats = dm.read(f'''SELECT * 
                        FROM Defense_Stats 
                        WHERE season>={YEAR-prev_years}
                                AND week != 17''', 'FastR')
    d_stats = d_stats.rename(columns={'season': 'year', 'defteam': 'team' })

    d_stats['week'] = d_stats['week'] + 1
    d_stats = switch_seasons(d_stats)
    d_stats = fix_bye_week(d_stats)

    rcols_def = [c for c in d_stats.columns if c not in ('team', 'week', 'season', 'y_act')]
    d_stats = add_rolling_stats(d_stats, gcols=['team'], rcols=rcols_def)

    return d_stats



def add_injuries(df):

    inj = dm.read('''SELECT * FROM PlayerInjuries''', 'Pre_PlayerData')
    inj = pd.concat([inj, pd.get_dummies(inj.game_status)], axis=1)
    inj = inj.dropna(subset=['injuries']).reset_index(drop=True)

    inj['leg_muscle_injury'] = 0
    inj.loc[inj.injuries.str.contains('Hamstring|Groin|Calf|Thigh|Quad'), 'leg_muscle_injury'] = 1

    inj['ankle_foot_injury'] = 0
    inj.loc[inj.injuries.str.contains('Ankle|Foot|Toe|Achilles'), 'ankle_foot_injury'] = 1

    inj['knee_hip_injury'] = 0
    inj.loc[inj.injuries.str.contains('Knee|Hip'), 'knee_hip_injury'] = 1

    inj['upper_body_injury'] = 0
    inj.loc[inj.injuries.str.contains('Wrist|Shoulder|Rib|Chest|Back'), 'knee_hip_injury'] = 1

    inj = inj[['player', 'pos', 'week', 'year', 'Questionable', 'Doubtful', 'Out',
            'leg_muscle_injury', 'ankle_foot_injury', 'knee_hip_injury', 'upper_body_injury']]

    df = pd.merge(df, inj, on=['player', 'pos', 'week', 'year'], how='left')
    cols = ['Questionable', 'Doubtful', 'Out', 'leg_muscle_injury', 'ankle_foot_injury',
            'knee_hip_injury', 'upper_body_injury']
    df[cols] = df[cols].fillna(0)

    players = list(df.loc[(df.week==WEEK) & (df.year==YEAR), 'player'].values)
    df = df[(df.Out != 1) & (df.Doubtful != 1)].reset_index(drop=True)
    player_inj =  list(df.loc[(df.week==WEEK) & (df.year==YEAR), 'player'].values)
    print(f'Players out this week: {[p for p in players if p not in player_inj]}')

    return df


def def_pts_allowed(df, pos):

    pts_allowed = dm.read(f'''SELECT * FROM Def_Allowed_{pos}''', 'Post_PlayerData')

    stat_cols = [c for c in pts_allowed if c not in ('team', 'games', 'week', 'year') and 'per_game' not in c]
    for c in stat_cols:
        pts_allowed[c] = pts_allowed[c] / (pts_allowed['games'].fillna(1))

    if pos == 'TE':
        pts_allowed.columns = [c.replace('wr', 'te') for c in pts_allowed.columns]
        stat_cols = [c for c in pts_allowed if c not in ('team', 'games', 'week', 'year') and 'per_game' not in c]
    
    pts_allowed = pts_allowed.drop([c for c in pts_allowed.columns if 'per_game' in c], axis=1)

    pts_allowed['player'] = pts_allowed.team
    pts_allowed = drop_extra_bye_week(pts_allowed).drop('player', axis=1)

    pts_allowed.week = pts_allowed.week + 1

    pts_allowed = fix_bye_week(pts_allowed)
    pts_allowed = switch_seasons(pts_allowed)

    pts_allowed = pts_allowed.rename(columns={'team': 'defTeam'})
    df = pd.merge(df, pts_allowed.drop('games', axis=1), on=['defTeam', 'week', 'year'], how='left')
    
    df = forward_fill(df)
    df[stat_cols] = df[stat_cols].fillna(df[stat_cols].mean())

    return df



#%%

# defense stats that can be added to the offensive player data
defense = fantasy_pros('DST').rename(columns={'player': 'team'})
d_stats = get_defense_stats(prev_years=2).drop('y_act', axis=1)
defense = pd.merge(defense, d_stats, on=['team', 'week', 'year'], how='inner')
defense = defense.dropna()
defense.columns = [f'def_{c}' if 'def' not in c else c for c in defense.columns]
defense = defense.rename(columns={'def_team': 'defTeam', 'def_week': 'week', 'def_year': 'year'})

#%%

pos = 'QB'

# pre-game data
df = fantasy_pros(pos); print(df.shape[0])
df = get_salaries(df, pos); print(df.shape[0])
df = get_experts(df, pos); print(df.shape[0])
df = add_pfr_matchup(df); print(df.shape[0])
df = add_gambling_lines(df); print(df.shape[0])
df = add_weather(df); print(df.shape[0])
dst = add_team_matchups().drop('offTeam', axis=1)
df = pd.merge(df, dst, on=['defTeam', 'year', 'week']); print(df.shape[0])

# post-game data
df = get_player_data(df, pos, YEAR); print(df.shape[0])

team_stats = get_team_stats(YEAR)
df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print(df.shape[0])

# df = get_coach_stats(df, YEAR); print(df.shape[0])
df = add_rz_stats_qb(df); print(df.shape[0])

# get the positional values for the team
pos_values = positional_values()
df = pd.merge(df, pos_values, on=['team', 'week', 'year']); print(df.shape[0])

# fill in the missing data and drop anything remaining
df = forward_fill(df)
df = df.dropna().reset_index(drop=True); print(df.shape[0])

df = add_injuries(df); print(df.shape[0])
df = pd.merge(df, defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
df = def_pts_allowed(df, pos); print(df.shape[0])

print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
dm.write_to_db(df.iloc[:,:2000], 'Model_Features', 'QB_Data', if_exist='replace')
if df.shape[1] > 2000:
    dm.write_to_db(df.iloc[:,2000:], 'Model_Features', 'QB_Data2', if_exist='replace')

#%%
for pos in ['RB', 'WR', 'TE']:

    #--------------------
    # Pre-Game Data
    #--------------------

    df = fantasy_pros(pos); print(df.shape[0])
    df = add_injuries(df); print(df.shape[0])

    df = get_salaries(df, pos); print(df.shape[0])
    df = add_pfr_matchup(df); print(df.shape[0])
    df = get_experts(df, pos); print(df.shape[0])
    df = add_gambling_lines(df); print(df.shape[0])
    df = add_weather(df); print(df.shape[0])
    if pos == 'WR': df = cb_matchups(df); print(df.shape[0])
    if pos == 'TE': df = te_matchups(df); print(df.shape[0])

    pos_values = positional_values()
    df = pd.merge(df, pos_values, on=['team', 'week', 'year']); print(df.shape[0])
    
    dst = add_team_matchups().drop('offTeam', axis=1)
    df = pd.merge(df, dst, on=['defTeam', 'year', 'week']); print(df.shape[0])

    #-----------------------
    # Post-Game Data
    #----------------------
    df = get_player_data(df, pos, YEAR); print(df.shape[0])
    
    team_stats = get_team_stats(YEAR, prev_years=2)
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print(df.shape[0])
    
    df = calc_market_share(df); print(df.shape[0])
    # df = get_coach_stats(df, YEAR); print(df.shape[0])
    df = add_rz_stats(df); print(df.shape[0])

    team_qb = get_max_qb()
    df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])

    # pre game data reliant on team information
    compare_cols = ['fp_rank', 'dk_salary', 'expertConsensus', 'projected_points', 'fantasyPoints', 'ProjPts']
    df = add_player_comparison(df, compare_cols); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df = df.dropna().reset_index(drop=True); print(df.shape[0])

    df = pd.merge(df, defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = def_pts_allowed(df, pos); print(df.shape[0])

    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    dm.write_to_db(df.iloc[:, :2000], 'Model_Features', f'{pos}_Data', if_exist='replace')
    if df.shape[1] > 2000:
        dm.write_to_db(df.iloc[:, 2000:], 'Model_Features', f'{pos}_Data2', if_exist='replace')

#%%

defense = fantasy_pros('DST').rename(columns={'player': 'team'})
d_stats = get_defense_stats(prev_years=2)
defense = pd.merge(defense, d_stats, on=['team', 'week', 'year'], how='inner')

all_cols = [c for c in defense.columns if c != 'y_act']
defense = defense.dropna(subset=all_cols)

pff_def = add_team_matchups().rename(columns={'defTeam': 'team'})
pff_def = add_rolling_stats(pff_def, gcols=['team'], rcols=[c for c in pff_def.columns if 'expert' in c])
defense = pd.merge(defense, pff_def, on=['team', 'year', 'week'])

team_qb = get_max_qb().rename(columns={'team': 'offTeam'})
defense = pd.merge(defense, team_qb, on=['offTeam', 'week', 'year'], how='left')

defense = add_gambling_lines(defense); print(defense.shape[0])
defense = add_weather(defense); print(defense.shape[0])

pos_values = positional_values().rename(columns={'team': 'offTeam'})
defense = pd.merge(defense, pos_values, on=['offTeam', 'week', 'year']); print(defense.shape[0])

team_stats = get_team_stats(YEAR).rename(columns={'team': 'offTeam'})
defense = pd.merge(defense, team_stats, on=['offTeam', 'week', 'year']); print(defense.shape[0])

defense = defense.copy().rename(columns={'team': 'player'})
defense = forward_fill(defense)
defense = defense.fillna(defense.mean())

defense.columns = [c.replace('_dst', '') for c in defense.columns]

dm.write_to_db(defense, 'Model_Features', f'Defense_Data', if_exist='replace')

# %%


# defense stats that can be added to the offensive player data
defense = fantasy_pros('DST').rename(columns={'player': 'team'})
d_stats = get_defense_stats(prev_years=2).drop('y_act', axis=1)
defense = pd.merge(defense, d_stats, on=['team', 'week', 'year'], how='inner')
defense = defense.dropna()
defense.columns = [f'def_{c}' if 'def' not in c else c for c in defense.columns]
defense = defense.rename(columns={'def_team': 'defTeam', 'def_week': 'week', 'def_year': 'year'})


# pre-game data
output = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:
    
    df = fantasy_pros(pos, add_rolling=False); print(df.shape[0])
    df = add_injuries(df); print(df.shape[0])
    df = get_salaries(df, pos); print(df.shape[0])
    df = get_experts(df, pos, add_rolling=False); print(df.shape[0])
    dst = add_team_matchups().drop('offTeam', axis=1)
    df = pd.merge(df, dst, on=['defTeam', 'year', 'week']); print(df.shape[0])

    df = add_weather(df); print(df.shape[0])
    df = add_gambling_lines(df); print(df.shape[0])
    team_stats = get_team_stats(YEAR, prev_years=2)
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print(df.shape[0])
    # df = get_coach_stats(df, YEAR); print(df.shape[0])
    
    pos_values = positional_values()
    df = pd.merge(df, pos_values, on=['team', 'week', 'year']); print(df.shape[0])

    team_qb = get_max_qb()
    df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])

    y_acts = dm.read(f'''SELECT player, team, week, season year, y_act 
                         FROM {pos}_Stats
                         WHERE year >= 2020
                               AND y_act IS NOT NULL''', 'FastR')

    y_acts.week = y_acts.week + 1
    y_acts = switch_seasons(y_acts)
    y_acts = fix_bye_week(y_acts)

    df = pd.merge(df, y_acts, on=['player', 'team', 'week', 'year'], how='left')
    df = df[~(df.y_act.isnull()) | ((df.week==WEEK) & (df.year==YEAR))].reset_index(drop=True); print(df.shape[0])

    df.loc[df.fd_salary < 100, 'fd_salary'] = np.nan
    df = forward_fill(df)
    df.fd_salary = df.fd_salary.fillna(df.fd_salary.mean())

    compare_cols = ['fp_rank', 'projected_points',
                    'dk_salary', 'fd_salary', 'yahoo_salary', 'fantasyPoints',
                    'fantasyPointsRank', 'ProjPts', 'expertConsensus']
    df = add_player_comparison(df, compare_cols)

    df = pd.merge(df, defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = def_pts_allowed(df, pos); print(df.shape[0])

    df = df.dropna(subset=[c for c in df.columns if c !='y_act']).reset_index(drop=True); print(df.shape[0])
    df['pos'] = pos
        
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    output = pd.concat([output, df], axis=0)

dm.write_to_db(output, 'Model_Features', 'Backfill', 'replace')


# %%