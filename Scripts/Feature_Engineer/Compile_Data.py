#%%

YEAR = 2022
WEEK = 15

#%%
import pandas as pd 
import pyarrow.parquet as pq
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from scipy.stats.morestats import shapiro

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
    
    rolls = df.groupby(gcols)[rcols].rolling(period, min_periods=1).agg(agg_type).reset_index(drop=True)
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

    rolls3_mean = rolling_stats(df, gcols, rcols, 3, agg_type='mean')
    rolls3_max = rolling_stats(df, gcols, rcols, 3, agg_type='max')

    rolls8_mean = rolling_stats(df, gcols, rcols, 8, agg_type='mean')
    rolls8_max = rolling_stats(df, gcols, rcols, 8, agg_type='max')
    rolls8_std = rolling_stats(df, gcols, rcols, 8, agg_type='std')

    hist_mean = rolling_expand(df, gcols, rcols, agg_type='mean')
    hist_std = rolling_expand(df, gcols, rcols, agg_type='std')
    hist_p80 = rolling_expand(df, gcols, rcols, agg_type='p95')

    df = pd.concat([df, 
                    rolls8_mean, rolls8_max, rolls8_std,
                    rolls3_mean, rolls3_max,
                    #hist_mean, hist_std, hist_p80
                    ], axis=1)

    return df


def switch_seasons(df):

    # any seasons 2020 or earlier when it was a 17 week season, convert week 17 to the first week of the next year
    # note that the year is +1 and week is set to one in a single step
    df.loc[(df.week==17) & (df.year <= 2020), ['year', 'week']] = [df.loc[(df.week==17) & (df.year <= 2020), 'year'] + 1, 1]
    
    # for any seasons after 2020, set week 18 to the following year first week.
    # however, don't do the conversion for the current year since you need to keep current week for this year's predictions
    df.loc[(df.week==18) & (df.year >= 2021) & (df.year != YEAR), ['year', 'week']] = \
        [df.loc[(df.week==18) & (df.year >= 2021) & (df.year != YEAR), 'year'] + 1, 1]

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


def fantasy_pros(pos):

    fp = dm.read(f'''SELECT * 
                    FROM FantasyPros 
                    WHERE pos='{pos}' 
                          AND team is NOT NULL''', 'Pre_PlayerData')
    fp = name_cleanup(fp)
    if pos == 'DST': fp = fp.drop('player', axis=1).rename(columns={'team': 'player'})
    
    for c in ['fp_rank', 'rankadj_fp_rank', 'playeradj_fp_rank']:
        fp['log_' + c] = np.log(fp[c]+1)

    return fp


def fantasy_pros_rolling(df):

    fp_cols = ['log_fp_rank', 'fp_rank', 'projected_points', 'rankadj_fp_rank', 'playeradj_fp_rank', 
               'log_rankadj_fp_rank', 'log_playeradj_fp_rank']
    df = add_rolling_stats(df, ['player'], fp_cols)
    df = forward_fill(df)

    return df


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
                            expertConsensus, expertNathanJahnke, expertIanHartitz,
                            rankadj_expertConsensus, rankadj_expertNathanJahnke,
                            playeradj_expertNathanJahnke,playeradj_expertConsensus 
                       
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
                    'expertConsensus', 'expertNathanJahnke','expertIanHartitz',
                    'rankadj_expertConsensus', 'rankadj_expertNathanJahnke',
                    'playeradj_expertNathanJahnke','playeradj_expertConsensus', 
                     ]

    # convert all the expert rankings to log values
    for c in expert_cols:
        if 'expert' in c: 
            experts[f'log_{c}'] = np.log(experts[c]+1)

    # append the log columns to list for rolling stats
    for c in experts.columns:
        if 'log' in c: expert_cols.append(c) 

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

    pff = dm.read('''SELECT player, offTeam team, week, year, 
                            expertConsensus, fantasyPoints, `Proj Pts` ProjPts
                     FROM PFF_Expert_Ranks
                     JOIN (SELECT player, offTeam, week, year, fantasyPoints
                           FROM PFF_Proj_Ranks)
                           USING (player, offTeam, week, year) ''', "Pre_PlayerData")

    df = pd.merge(fp, dk_sal, on=['player', 'team', 'week', 'year'], how='inner')
    df = pd.merge(df, pff, on=['player', 'team', 'week', 'year'], how='inner')

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

def get_player_data(df, pos, YEAR, prev_years=1):

    player_data = dm.read(f'''SELECT * 
                              FROM {pos}_Stats 
                              WHERE (season = 2020 AND week != 17)
                                     OR (season >=2021 AND week != 18)
                                ''', 'FastR')

    if pos=='QB':
        rcols_player = [c for c in player_data.columns if 'pass_' in c]
        rcols_player = [c for c in rcols_player if c not in ('y_act_rush', 'y_act_pass')]
    else:
        rcols_player = [c for c in player_data.columns if 'rec_' in c]

    rcols_player.extend([c for c in player_data.columns if 'rush_' in c])
    rcols_player = list(set(rcols_player))

    player_data = player_data.rename(columns={'season': 'year'})
    player_data = add_rolling_stats(player_data, gcols=['player'], rcols=rcols_player)

    player_data = forward_fill(player_data)
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


def get_team_stats(YEAR, prev_years=1):
    
    team_stats = dm.read(f'''SELECT * 
                             FROM Team_Stats 
                             WHERE (season = 2020 AND week != 17)
                                    OR (season >=2021 AND week != 18)''', 'FastR')

    rcols_team = [c for c in team_stats.columns if 'rush_' in c or 'rec_' in c]

    team_stats = team_stats.rename(columns={'season': 'year'})
    team_stats = add_rolling_stats(team_stats, ['team'], rcols_team)

    team_stats = team_stats[team_stats.year >= 2020].reset_index(drop=True)
    team_stats['week'] = team_stats.week + 1
    team_stats = switch_seasons(team_stats)
    team_stats = fix_bye_week(team_stats)

    return team_stats


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

    rz = forward_fill(rz).fillna(0)
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
    away = away.assign(is_home=0)
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


def add_next_gen(df, pos, stat_type):

    next_gen = dm.read(f"SELECT * FROM NextGen_{stat_type}", 'Post_PlayerData')
    try: next_gen = next_gen.drop('pos', axis=1)
    except: pass
    teams = dm.read(f'''SELECT player, week, year, team 
                        FROM Player_Teams
                         ''', 'Simulation')
    next_gen = pd.merge(next_gen, teams, on=['player', 'week', 'year'])

    next_gen = drop_extra_bye_week(next_gen)
    next_gen.week = next_gen.week + 1
    next_gen = fix_bye_week(next_gen)
    next_gen = switch_seasons(next_gen)

    # issue with bye week creating duplicates after trading
    next_gen.loc[(next_gen.player=='Chase Claypool') & \
                 (next_gen.week==10) & (next_gen.year==2022) & \
                  (next_gen.team=='PIT'), 'week'] = 9

    df = pd.merge(df, next_gen.drop('team', axis=1), on=['player', 'week', 'year'], how='left')

    fill_cols = [c for c in next_gen.columns if c not in (['player', 'pos', 'week', 'year', 'team'])]
    
    df = forward_fill(df)
    df[fill_cols] = df[fill_cols].fillna(df[fill_cols].mean())
    df = add_rolling_stats(df, gcols=['player'], rcols=fill_cols)

    return df


def get_defense_stats(prev_years=1):
    d_stats = dm.read(f'''SELECT * 
                        FROM Defense_Stats 
                        WHERE (season = 2020 AND week != 17)
                               OR (season >=2021 AND week != 18)
                              ''', 'FastR')
    d_stats = d_stats.rename(columns={'season': 'year', 'defteam': 'team' })

    d_stats['week'] = d_stats['week'] + 1
    d_stats = switch_seasons(d_stats)
    d_stats = fix_bye_week(d_stats)

    rcols_def = [c for c in d_stats.columns if c not in ('team', 'week', 'season')]
    d_stats = add_rolling_stats(d_stats, gcols=['team'], rcols=rcols_def)

    return d_stats



def add_injuries(df, pos=None, def_join=False, oline_join=False):

    inj = dm.read('''SELECT * FROM PlayerInjuries''', 'Pre_PlayerData')
    inj = pd.concat([inj, pd.get_dummies(inj.game_status),  pd.get_dummies(inj.practice_status)], axis=1)
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
            'leg_muscle_injury', 'ankle_foot_injury', 'knee_hip_injury', 'upper_body_injury',
             'Did Not Participate In Practice', 'Full Participation in Practice',
             'Limited Participation in Practice'
            ]]

    # rcols = ['Did Not Participate In Practice', 'Limited Participation in Practice', 'Questionable', 'Out']
    # inj = inj.sort_values(by=['player','year', 'week']).reset_index(drop=True)
    # inj = add_rolling_stats(inj, ['player'], rcols).fillna(0)

    if def_join: df = pd.merge(df.drop('pos', axis=1), inj, on=['player', 'week', 'year'], how='left')
    elif oline_join: df = pd.merge(df, inj.drop('pos', axis=1), on=['player', 'week', 'year'], how='left')
    else:  df = pd.merge(df, inj, on=['player', 'pos', 'week', 'year'], how='left')

    cols = ['Questionable', 'Doubtful', 'Out', 'leg_muscle_injury', 'ankle_foot_injury',
            'knee_hip_injury', 'upper_body_injury', 'Did Not Participate In Practice', 
            'Limited Participation in Practice']
    df['Full Participation in Practice'] = df['Full Participation in Practice'].fillna(1)
    df[cols] = df[cols].fillna(0)

    players = list(df.loc[(df.week==WEEK) & (df.year==YEAR), 'player'].values)

    # keep doubtful and out players for defense to calculate missing value
    if not def_join and not oline_join:
        df = df[(df.Out != 1) & (df.Doubtful != 1)].reset_index(drop=True)
        
        # also remove any players that don't have an actual score value
        df = attach_y_act(df, pos)
        df = drop_y_act_except_current(df, WEEK, YEAR)
        df = df.drop('y_act', axis=1)

    player_inj =  list(df.loc[(df.week==WEEK) & (df.year==YEAR), 'player'].values)
    print(f'Players out this week: {[p for p in players if p not in player_inj]}')

    return df, player_inj


def pts_allowed_data(pos):

    pts_allowed = dm.read(f'''SELECT * FROM Def_Allowed_{pos}''', 'Post_PlayerData')
    ignore_cols = ('team', 'games', 'week', 'year')
    pts_allowed.columns = [f'{pos.lower()}_{c}' if pos.lower() not in c and c not in ignore_cols else c for c in pts_allowed.columns]

    return pts_allowed

def def_pts_allowed(df, join_col='defTeam'):

    qb = pts_allowed_data('QB')
    rb = pts_allowed_data('RB').drop('games', axis=1)
    wr = pts_allowed_data('WR').drop('games', axis=1)
    te = pts_allowed_data('TE').drop('games', axis=1)

    pts_allowed = pd.merge(qb, rb, on=['team', 'week', 'year'])
    pts_allowed = pd.merge(pts_allowed, wr, on=['team', 'week', 'year'])
    pts_allowed = pd.merge(pts_allowed, te, on=['team', 'week', 'year'])

    stat_cols = [c for c in pts_allowed if c not in ('team', 'games', 'week', 'year') and 'per_game' not in c]
    for c in stat_cols:
        pts_allowed[c] = pts_allowed[c] / (pts_allowed['games'].fillna(1))

    pts_allowed = pts_allowed.drop([c for c in pts_allowed.columns if 'per_game' in c], axis=1)

    # fix the bye week and conversion between seasons
    pts_allowed['player'] = pts_allowed.team
    pts_allowed = drop_extra_bye_week(pts_allowed).drop('player', axis=1)
    pts_allowed.week = pts_allowed.week + 1
    pts_allowed = fix_bye_week(pts_allowed)
    pts_allowed = switch_seasons(pts_allowed)

    # add in new columns
    pts_allowed['pass_over_rush_yd_allowed'] = pts_allowed['qb_pass_yds_allowed'] / pts_allowed['rb_rush_yds_allowed']
    pts_allowed['pass_minus_rush_yd_allowed'] = pts_allowed['qb_pass_yds_allowed'] - pts_allowed['rb_rush_yds_allowed']
    pts_allowed['wr_over_rb_yd_allowed'] = pts_allowed['wr_rec_yds_allowed'] / pts_allowed['rb_rec_yds_allowed']
    pts_allowed['wr_over_te_yd_allowed'] = pts_allowed['wr_rec_yds_allowed'] / pts_allowed['te_rec_yds_allowed']

    pts_allowed['pass_over_rush_dk_allowed'] = pts_allowed['qb_dk_fp_allowed'] / pts_allowed['rb_dk_pts_allowed']
    pts_allowed['pass_minus_rush_dk_allowed'] = pts_allowed['qb_dk_fp_allowed'] - pts_allowed['rb_dk_pts_allowed']
    pts_allowed['wr_over_rb_dk_allowed'] = pts_allowed['wr_dk_pts_allowed'] / pts_allowed['rb_dk_pts_allowed']
    pts_allowed['wr_over_te_dk_allowed'] = pts_allowed['wr_dk_pts_allowed'] / pts_allowed['te_wr_dk_pts_allowed']

    # average out the stats over rolling 3 games
    stat_cols = [c for c in pts_allowed if c not in ('team', 'games', 'week', 'year') and 'per_game' not in c]
    pts_allowed = pts_allowed.sort_values(by=['team', 'year', 'week']).reset_index(drop=True)
    labels = pts_allowed[['team', 'week', 'year', 'games']]
    roll_stats = pts_allowed.groupby('team')[stat_cols].rolling(3).agg('mean').reset_index(drop=True)
    pts_allowed = pd.concat([labels, roll_stats], axis=1).dropna().reset_index(drop=True)

    # join the pts allowed to the dataframe
    pts_allowed = pts_allowed.rename(columns={'team': join_col})
    df = pd.merge(df, pts_allowed.drop('games', axis=1), on=[join_col, 'week', 'year'], how='left')
    
    df = forward_fill(df)
    df[stat_cols] = df[stat_cols].fillna(df[stat_cols].mean())

    return df



def pff_defense_rollup():

    def_players = dm.read('''SELECT * FROM Defense_Players''', 'Post_PlayerData')
    def_players.player = def_players.player.apply(dc.name_clean)
    def_players = def_players.rename(columns={'position': 'pos'})

    all_cols = ['player', 'team', 'week', 'year', 'pos', 'player_game_count', 
            'snap_counts_defense', 'snap_counts_coverage', 'snap_counts_pass_rush',
            'grades_coverage_defense', 'grades_defense',  
            'grades_pass_rush_defense', 'grades_run_defense', 'grades_tackle',
            'hits', 'hurries', 'qb_rating_against', 'interceptions']
    def_players = def_players.loc[~def_players.pos.isnull(), all_cols].reset_index(drop=True)
    def_players = def_players.fillna(def_players.mean())

    def_players = drop_extra_bye_week(def_players)
    def_players.week = def_players.week + 1
    def_players = fix_bye_week(def_players)
    def_players = switch_seasons(def_players)
    def_players, _ = add_injuries(def_players, def_join=True)

    stat_cols = ['grades_coverage_defense', 'grades_defense',  
                    'grades_pass_rush_defense', 'grades_run_defense', 'grades_tackle',
                    'hits', 'hurries',  'interceptions']
    def_players.loc[((def_players.Out == 1) | (def_players.Doubtful == 1)),stat_cols] = \
                        def_players.loc[((def_players.Out == 1) | (def_players.Doubtful == 1)), stat_cols] / 3

    def_players.loc[((def_players.Out == 1) | (def_players.Doubtful == 1)), 'qb_rating_against'] = 110

    wm_def = lambda x: np.average(x, weights=def_players.loc[x.index, "snap_counts_defense"])
    wm_cov = lambda x: np.average(x, weights=def_players.loc[x.index, "snap_counts_coverage"])
    wm_pass_rush = lambda x: np.average(x, weights=def_players.loc[x.index, "snap_counts_pass_rush"])
    wm_run_def = lambda x: np.average(x, weights=def_players.loc[x.index, "grades_pass_rush_defense"])

    gcols = ["team", "week", "year"]
    def_team = def_players.groupby(gcols).agg(team_games=('player_game_count', 'max'),
                                            grades_defense=("grades_defense", wm_def),
                                            grades_coverage_defense=("grades_coverage_defense", wm_cov),  
                                            grades_pass_rush_defense=("grades_pass_rush_defense", wm_pass_rush),
                                            grades_run_defense=("grades_run_defense", wm_run_def),
                                            hits=('hits', 'sum'),
                                            hurries=('hurries', 'sum'),
                                            qb_rating_against=('qb_rating_against', wm_cov),
                                            grades_tackle=('grades_tackle', wm_run_def)).reset_index()

    per_game_cols = ['hits', 'hurries']
    for c in per_game_cols:
        def_team[f'{c}_per_game'] = def_team[c] / def_team.team_games
    def_team = def_team.drop(per_game_cols, axis=1)

    team_stat_cols = ['grades_defense', 'grades_coverage_defense', 'grades_pass_rush_defense',
                    'grades_run_defense', 'qb_rating_against', 'grades_tackle',
                    'hits_per_game', 'hurries_per_game']

    def_team = def_team.sort_values(by=['team', 'year', 'week']).reset_index(drop=True)
    labels = def_team[['team', 'week', 'year', 'team_games']]
    roll_stats = def_team.groupby('team')[team_stat_cols].rolling(3).agg('mean').reset_index(drop=True)
    def_team = pd.concat([labels, roll_stats], axis=1).dropna().reset_index(drop=True)

    return def_team


def pff_oline_rollup():
    oline = dm.read('''SELECT * FROM Offensive_Line_Players''', 'Post_PlayerData')
    oline.player = oline.player.apply(dc.name_clean)
    oline = oline.rename(columns={'position': 'pos'})

    all_cols = ['player', 'team', 'week', 'year', 'pos', 'player_game_count', 
                'snap_counts_offense', 'grades_offense', 'grades_pass_block', 
                'grades_run_block', 'hits_allowed', 'hurries_allowed','pressures_allowed', 'pbe']
    oline = oline.loc[oline.pos.isin(['C', 'G', 'T']), all_cols].reset_index(drop=True)
    oline = oline.fillna(oline.mean())

    oline = drop_extra_bye_week(oline)
    oline.week = oline.week + 1
    oline = fix_bye_week(oline)
    oline = switch_seasons(oline)
    oline, _ = add_injuries(oline, oline_join=True)

    stat_cols1 = ['snap_counts_offense', 'grades_offense', 'grades_pass_block', 
                'grades_run_block', 'pbe']
    oline.loc[((oline.Out == 1) | (oline.Doubtful == 1)), stat_cols1] = \
                oline.loc[((oline.Out == 1) | (oline.Doubtful == 1)), stat_cols1] / 3

    stat_cols2 = ['hits_allowed', 'hurries_allowed','pressures_allowed',]
    oline.loc[((oline.Out == 1) | (oline.Doubtful == 1)), stat_cols1] = \
                        2 * (oline.loc[((oline.Out == 1) | (oline.Doubtful == 1)), stat_cols2] + 1)
    oline = oline.fillna(oline.mean())

    wm = lambda x: np.average(x, weights=oline.loc[x.index, "snap_counts_offense"])
    gcols = ["team", "week", "year"]
    oline_team = oline.groupby(gcols).agg(team_games=('player_game_count', 'max'),
                                        grades_offense=("grades_offense", wm),
                                            grades_pass_block=("grades_pass_block", wm),  
                                            grades_run_block=("grades_run_block", wm),
                                            hits_allowed=('hits_allowed', 'sum'),
                                            hurries_allowed=('hurries_allowed', 'sum'),
                                            pressures_allowed=('pressures_allowed', 'sum'),
                                            pbe=('pbe', wm)).reset_index()

    per_game_cols = ['hits_allowed', 'hurries_allowed', 'pressures_allowed']
    for c in per_game_cols:
        oline_team[f'{c}_per_game'] = oline_team[c] / oline_team.team_games
    oline_team = oline_team.drop(per_game_cols, axis=1)

    team_stat_cols = ['grades_offense', 'grades_pass_block', 'grades_run_block', 
                    'hits_allowed_per_game', 'hurries_allowed_per_game','pressures_allowed_per_game', 'pbe']
    
    oline_team = oline_team.sort_values(by=['team', 'year', 'week']).reset_index(drop=True)
    labels = oline_team[['team', 'week', 'year', 'team_games']]
    roll_stats = oline_team.groupby('team')[team_stat_cols].rolling(3).agg('mean').reset_index(drop=True)
    oline_team = pd.concat([labels, roll_stats], axis=1).dropna().reset_index(drop=True)

    return oline_team



def add_qbr(df):

    qbr = dm.read('''SELECT * FROM ESPN_QBR''', 'Post_PlayerData')

    def remove_team(name):

        uppers = [x.isupper() for x in name]
        split_start = [i for i, x in enumerate(uppers) if x][2]
        
        return name[:split_start]

    qbr.columns = ['rank', 'player', 'qbr', 'paa', 'plays', 'epa', 'pass_rating', 'run_rating', 
                   'sack_rating', 'penalty_rating', 'raw_rating', 'week', 'year']
    qbr.player = qbr.player.apply(remove_team).apply(dc.name_clean)

    for c in ['paa', 'epa', 'pass_rating', 'run_rating', 'sack_rating', 'penalty_rating']:
        qbr[c] = qbr[c] / qbr['plays']
    
    teams = dm.read('''SELECT player, week, year, team 
                       FROM FantasyPros
                       WHERE pos='QB' ''', 'Pre_PlayerData')
    qbr = pd.merge(qbr, teams, on=['player', 'week', 'year'])

    qbr = drop_extra_bye_week(qbr)
    qbr.week = qbr.week + 1
    qbr = fix_bye_week(qbr)
    qbr = switch_seasons(qbr)
    
    df = pd.merge(df, qbr.drop('team', axis=1), on=['player', 'week', 'year'], how='left')

    df = forward_fill(df)
    fill_cols = ['rank', 'player', 'qbr', 'paa', 'plays', 'epa', 'pass_rating', 'run_rating', 
                 'sack_rating', 'penalty_rating', 'raw_rating']
    df[fill_cols] = df[fill_cols].fillna(df[fill_cols].mean())

    return df



def add_qb_adv(df):
    qb_adv = dm.read('''SELECT player, team, week, year,
                            G games, GS games_started,
                            Cmp pass_completions, Att pass_attempts
                        FROM PFR_Advanced_QB
                        WHERE pos IN ('QB', 'qb') ''', 'Post_PlayerData')
    qb_adv.player = qb_adv.player.apply(dc.name_clean)

    adv = dm.read('''SELECT player, team, week, year,
                            Cmp_pct comp_pct, TD_Pct td_pct,
                            Int_pct int_pct, `1d` first_downs,
                            Y_A yards_per_attempt, AY_A adj_yard_per_attempt,
                            Y_C yards_per_completion, Rate qb_rating, 
                            QBR qbr_rating, `Yds.1` sack_yards_lost,
                            NY_A net_yards_per_att, ANY_A adj_net_yards_per_att,
                            Sk_pct sack_pct, `4QC` fourth_qtr_comback, 
                            GWD game_winning_drives, wins / losses win_pct
                    FROM PFR_Advanced_QB
                    ''', 'Post_PlayerData')

    adv_acc = dm.read('''SELECT player, team, week, year,
                                PassingBadTh bad_throws, PassingBad_Pct bad_pass_pct,
                                PassingOnTgt pass_on_tgt, PassingOnTgt_pct pass_on_tgt_pct
                        FROM PFR_Advanced_QB_Accuracy''', 'Post_PlayerData')

    adv_ay = dm.read('''SELECT player, team, week, year,
                            PassingIAY intended_ay, 
                            PassingIAY_PA intended_ay_per_att, 
                            PassingCAY completed_ay,
                            PassingCAY_Cmp completed_ay_per_complete,
                            PassingCAY_PA completed_ay_per_att, 
                            PassingYAC pass_yac, 
                            PassingYAC_Cmp pass_yac_per_complete
                        FROM PFR_Advanced_QB_AirYards''', 'Post_PlayerData' )

    adv_ptype = dm.read('''SELECT player, team, week, year,
                                RPOPlays rpo_plays, RPOYds rpo_yards, 
                                RPOPassAtt rpo_pass_att, RPOPassYds rpo_pass_yds,
                                RPORushAtt rpo_rush_att, RPORushYds rpo_rush_yds, 
                                PlayActionPassAtt play_action_att, PlayActionPassYds play_action_yds
                        FROM PFR_Advanced_QB_PlayType''', 'Post_PlayerData' )

    adv_pressure = dm.read('''SELECT player, team, week, year,
                                    PassingSk pass_sacks, PassingPktTime pocket_time, 
                                    PassingBltz pass_against_blitz,
                                    PassingHrry pass_hurries, PassingHits pass_hits, 
                                    PassingPrss_pct complete_pct_pressure, PassingScrm scramble_passes,
                                    PassingYds_Scr pass_yds_per_scramble
                        FROM PFR_Advanced_QB_Pressure''', 'Post_PlayerData' )

    for _df in [adv, adv_acc, adv_ay, adv_ptype, adv_pressure]:
        _df.player = _df.player.apply(dc.name_clean)
        qb_adv = pd.merge(qb_adv, _df, on=['player', 'team', 'week', 'year'])

    qb_adv = dc.convert_to_float(qb_adv)

    per_game = ['pass_completions', 'pass_attempts', 'first_downs', 'sack_yards_lost',
                'fourth_qtr_comback', 'game_winning_drives', 'bad_throws', 'pass_on_tgt', 'intended_ay',
                'completed_ay','pass_yac', 'rpo_plays', 'rpo_yards', 'rpo_pass_att', 'rpo_pass_yds',
                'rpo_rush_att', 'rpo_rush_yds', 'play_action_att', 'play_action_yds', 'pass_sacks', 
                'pass_against_blitz', 'pass_hurries', 'pass_hits',  'scramble_passes', 'pass_yds_per_scramble']

    for c in per_game:
        qb_adv[f'{c}_per_game'] = qb_adv[c] / qb_adv.games

    per_att = ['first_downs',  'sack_yards_lost', 'rpo_yards', 'rpo_yards', 'play_action_yds', 'pass_on_tgt']
    for c in per_att:
        qb_adv[f'{c}_per_att'] = qb_adv[c] / qb_adv.pass_attempts

    qb_adv['complete_over_intended_ay'] = qb_adv.completed_ay - qb_adv.intended_ay


    qb_adv = drop_extra_bye_week(qb_adv)
    qb_adv.week = qb_adv.week + 1
    qb_adv = fix_bye_week(qb_adv).drop(['team', 'games', 'games_started', 
                                        'pass_completions', 'pass_attempts'], axis=1)
    qb_adv = switch_seasons(qb_adv)

    df = pd.merge(df, qb_adv, on=['player', 'week', 'year'], how='left')
    df = forward_fill(df)
    fill_cols = [c for c in qb_adv if c not in ('player', 'week', 'year')]
    df[fill_cols] = df[fill_cols].fillna(df[fill_cols].mean())

    return df




def results_vs_predicted(df, col):

    df = df.sort_values(by=['player','year', 'week']).reset_index(drop=True)
    df[f'{col}_miss'] = df.y_act - df[col]
    df[f'{col}_miss'] = (df.groupby('player')[f'{col}_miss'].shift(1)).fillna(0)
    df = add_rolling_stats(df, ['player'], [f'{col}_miss'])
    df[f'{col}_miss_recent_vs8'] = df[f'rmean3_{col}_miss'] - df[f'rmean8_{col}_miss']

    good_cols = [c for c in df.columns if 'miss' in c or c in ('player', 'team', 'week', 'year')]
    df = df[good_cols]

    return df

def projected_pts_vs_predicted(df, pos):

    proj_pts_actual = dm.read(f'''SELECT player, team, week, year, 
                                projected_points, fantasyPoints
                        FROM FantasyPros
                        JOIN (SELECT player, week, year, fantasyPoints
                                FROM PFF_Proj_Ranks
                                WHERE position='{pos.lower()}')
                                USING (player, week, year)
                        WHERE pos='{pos}'
                        ''', 'Pre_PlayerData')

    proj_pts_actual = attach_y_act(proj_pts_actual, pos)
    proj_pts_actual = drop_y_act_except_current(proj_pts_actual, WEEK, YEAR)

    proj_pts = results_vs_predicted(proj_pts_actual, 'projected_points')
    fant_pts = results_vs_predicted(proj_pts_actual, 'fantasyPoints')

    proj_pts_actual = proj_pts_actual.drop(['fantasyPoints', 'y_act', 'projected_points'], axis=1)
    proj_pts_actual = pd.merge(proj_pts_actual, proj_pts, on=['player', 'team', 'week', 'year'])
    proj_pts_actual = pd.merge(proj_pts_actual, fant_pts, on=['player', 'team', 'week', 'year'])

    df = pd.merge(df, proj_pts_actual, on=['player', 'team', 'week', 'year'], how='left')
    df = forward_fill(df)

    miss_cols = [c for c in df.columns if 'miss' in c]
    df[miss_cols] = df[miss_cols].fillna(0)

    return df


def get_snap_data():

    snaps = dm.read('''SELECT player, team, snap_counts, week, year
                    FROM Snap_Counts_V2
                    WHERE snap_counts!='bye' 
                            AND week != 18
                    ''', 'Post_PlayerData')

    snaps.snap_counts = snaps.snap_counts.apply(lambda x: int(float(x)))
    snaps.week = snaps.week.astype('int')

    team_snaps = snaps.groupby(['team', 'week','year']).agg(team_snap_count=('snap_counts', 'max'))
    snaps = pd.merge(snaps, team_snaps, on=['team', 'week', 'year']).drop('team', axis=1)
    snaps['snap_pct'] = snaps.snap_counts / snaps.team_snap_count

    snaps = snaps[snaps.snap_counts > 0].sort_values(by=['player', 'year', 'week']).reset_index(drop=True)
    snaps['avg_snap_pct'] = snaps.snap_pct.rolling(4).mean()
    snaps = snaps.drop(['snap_counts', 'team_snap_count'], axis=1)

    return snaps


def attach_y_act(df, pos, defense=False, rush_or_pass=''):

    if defense:
        y_act = dm.read(f'''SELECT defTeam player, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020''', 'FastR')
    
        df = pd.merge(df, y_act, on=['player',  'week', 'year'], how='left')
    
    elif pos=='QB':
        y_act = dm.read(f'''SELECT player, team, week, season year,
                                   fantasy_pts{rush_or_pass} y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020
                                  AND pass_pass_attempt_sum > 15''', 'FastR')
    
        df = pd.merge(df, y_act, on=['player', 'team', 'week', 'year'], how='left')
    
    else:
        y_act = dm.read(f'''SELECT player, team, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020''', 'FastR')
                            
        snaps = get_snap_data()
        proj = dm.read('''SELECT player, week, year, fantasyPoints
                        FROM PFF_Proj_Ranks''', 'Pre_PlayerData')

        y_act = pd.merge(y_act, snaps, on=['player', 'week', 'year'], how='left')
        y_act = pd.merge(y_act, proj, on=['player', 'week', 'year'], how='left')

        y_act = y_act[~((y_act.fantasyPoints > 12) & \
                        (y_act.snap_pct < y_act.avg_snap_pct*0.5) & \
                        (y_act.snap_pct <= 0.4) & \
                        (y_act.snap_pct > 0))].drop(['snap_pct', 'avg_snap_pct', 'fantasyPoints'], axis=1)
        
        df = pd.merge(df, y_act, on=['player', 'team', 'week', 'year'], how='left')

    return df



def drop_y_act_except_current(df, week, year):
    
    
    df = df[~(df.y_act.isnull()) | ((df.week==week) & (df.year==year))].reset_index(drop=True)
    df.loc[((df.week==week) & (df.year==year)), 'y_act'] = 0

    return df

def remove_non_uniques(df):
    cols = df.nunique()[df.nunique()==1].index
    cols = [c for c in cols if c != 'pos']
    df = df.drop(cols, axis=1)
    return df

def one_qb_per_week(df):
    max_qb = df.groupby(['team', 'year', 'week']).agg({'projected_points': 'max',
                                                       'fp_rank': 'min'}).reset_index()
    cols = df.columns
    df = pd.merge(df, max_qb.drop('fp_rank', axis=1), on=['team', 'year', 'week', 'projected_points'])
    df = pd.merge(df, max_qb.drop('projected_points', axis=1), on=['team', 'year', 'week', 'fp_rank'])
    df = df[cols]

    return df


def add_fp_rolling(df, pos):

    if pos == 'Defense':
        y_act = dm.read(f'''SELECT defTeam player, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020 ''', 'FastR')
        df = pd.merge(df, y_act, on=['player', 'week', 'year'], how='outer')

    else:
        y_act = dm.read(f'''SELECT player, team, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020''', 'FastR')

        df = pd.merge(df, y_act, on=['player', 'team', 'week', 'year'], how='outer')

    df = df.sort_values(by=['player', 'year', 'week'])
    df['fantasy_pts_score'] = df.groupby('player')['y_act'].shift(1)
    df = df.drop('y_act', axis=1)
    df = add_rolling_stats(df, ['player'], ['fantasy_pts_score'])

    df = forward_fill(df)
    df = df[~((df.week==1)  & (df.year==2020))]

    return df


#%%

# defense stats that can be added to the offensive player data
defense = fantasy_pros('DST').rename(columns={'player': 'team'})
d_stats = get_defense_stats(prev_years=1)
defense = pd.merge(defense, d_stats, on=['team', 'week', 'year'], how='inner')
defense = defense.dropna()
defense.columns = [f'def_{c}' if 'def' not in c else c for c in defense.columns]
defense = defense.rename(columns={'def_team': 'defTeam', 'def_week': 'week', 'def_year': 'year'})

#%%

pos = 'QB'

def qb_pull(rush_or_pass):

    # pre-game data
    df = fantasy_pros(pos); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])
    df = add_fp_rolling(df, pos); print(df.shape[0])
    df = fantasy_pros_rolling(df); print(df.shape[0])
    df = get_salaries(df, pos); print(df.shape[0])
    df = get_experts(df, pos); print(df.shape[0])
    df = add_pfr_matchup(df); print(df.shape[0])
    df = add_gambling_lines(df); print(df.shape[0])
    df = add_weather(df); print(df.shape[0])
    dst = add_team_matchups().drop('offTeam', axis=1)
    df = pd.merge(df, dst, on=['defTeam', 'year', 'week']); print(df.shape[0])


    # post-game data
    df = get_player_data(df, pos, YEAR, prev_years=1); print(df.shape[0])

    team_stats = get_team_stats(YEAR)
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print( df.shape[0])

    df = add_rz_stats_qb(df); print(df.shape[0])
    df = add_qbr(df); print(df.shape[0])
    df = add_qb_adv(df); print(df.shape[0])
    df = add_next_gen(df, pos, 'Passing'); print(df.shape[0])

    # get the positional values for the team
    pos_values = positional_values()
    df = pd.merge(df, pos_values, on=['team', 'week', 'year']); print(df.shape[0])

    df = pd.merge(df, defense, on=['defTeam', 'week', 'year']); print('dstats', df.shape[0])
    df = def_pts_allowed(df); print(df.shape[0])

    pff_def = pff_defense_rollup().rename(columns={'team': 'defTeam'})
    df = pd.merge(df, pff_def, on=['defTeam', 'week', 'year']); print(df.shape[0])

    pff_oline = pff_oline_rollup()
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])

    df = attach_y_act(df, pos, rush_or_pass=rush_or_pass)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df, pos); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df = df.dropna().reset_index(drop=True); print(df.shape[0])

    df = one_qb_per_week(df); print(df.shape[0])

    df = remove_non_uniques(df)
    df = df[(df.ProjPts > 10) & (df.projected_points > 10)].reset_index(drop=True)

    print('Total Rows:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())

    dm.write_to_db(df.iloc[:,:2000], 'Model_Features', f"QB_Data{rush_or_pass.replace('_', '')}", if_exist='replace')
    if df.shape[1] > 2000:
        dm.write_to_db(df.iloc[:,2000:], 'Model_Features', f"QB_Data{rush_or_pass.replace('_', '')}2", if_exist='replace')

    return df

qb_both = qb_pull('')
# # qb_rush = qb_pull('_rush')
# # qb_pass = qb_pull('_pass')

#%%
for pos in [
           'RB', 
           'WR', 
           'TE'
            ]:

    #--------------------
    # Pre-Game Data
    #--------------------

    df = fantasy_pros(pos); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])
    df = add_fp_rolling(df, pos); print(df.shape[0])
    df = fantasy_pros_rolling(df); print(df.shape[0])
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

    df = get_player_data(df, pos, YEAR, prev_years=1); print(df.shape[0])
    
    team_stats = get_team_stats(YEAR, prev_years=1)
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print('team_stats', df.shape[0])
    
    df = calc_market_share(df); print(df.shape[0])
    df = add_rz_stats(df); print(df.shape[0])
    df = advanced_rec_stats(df)
    if pos in ('WR', 'TE'):
        df = add_next_gen(df, pos, 'Receiving'); print('next_gen', df.shape[0])
    if pos == 'RB': 
        df = advanced_rb_stats(df)
        df = add_next_gen(df, pos, 'Rushing'); print(df.shape[0])

    team_qb = get_max_qb()
    df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])

    # pre game data reliant on team information
    compare_cols = ['fp_rank', 'dk_salary', 'expertConsensus', 'projected_points', 'fantasyPoints', 'ProjPts']
    df = add_player_comparison(df, compare_cols); print(df.shape[0])

    df = pd.merge(df, defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = def_pts_allowed(df); print(df.shape[0])

    pff_def = pff_defense_rollup().rename(columns={'team': 'defTeam'})
    df = pd.merge(df, pff_def, on=['defTeam', 'week', 'year']); print(df.shape[0])

    pff_oline = pff_oline_rollup()
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])
    
    df = attach_y_act(df, pos)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df, pos); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df = df.dropna().reset_index(drop=True); print(df.shape[0])
    df = remove_non_uniques(df)

    print('Total Rows:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())
    
    dm.write_to_db(df.iloc[:, :2000], 'Model_Features', f'{pos}_Data', if_exist='replace')
    if df.shape[1] > 2000:
        dm.write_to_db(df.iloc[:, 2000:], 'Model_Features', f'{pos}_Data2', if_exist='replace')

#%%

defense = fantasy_pros('DST')
defense = add_fp_rolling(defense, 'Defense'); print(defense.shape[0])
defense = defense.rename(columns={'player': 'team'}); print(defense.shape[0])

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

pff_def = pff_defense_rollup()
defense = pd.merge(defense, pff_def, on=['team', 'week', 'year']); print(defense.shape[0])

pff_oline = pff_oline_rollup().rename(columns={'team': 'offTeam'})
defense = pd.merge(defense, pff_oline, on=['offTeam', 'week', 'year']); print(defense.shape[0])

defense = defense.copy().rename(columns={'team': 'player'})
defense = forward_fill(defense)

defense = attach_y_act(defense, pos='Defense', defense=True)
defense = drop_y_act_except_current(defense, WEEK, YEAR); print(defense.shape[0])
defense = defense.dropna(); print(defense.shape[0])

print('Unique player-week-years:', defense[['player', 'week', 'year']].drop_duplicates().shape[0])
print('Team Counts by Week:', defense[['year', 'week', 'player']].drop_duplicates().groupby(['year', 'week'])['player'].count())

defense.columns = [c.replace('_dst', '') for c in defense.columns]
defense = remove_non_uniques(defense)

dm.write_to_db(defense, 'Model_Features', f'Defense_Data', if_exist='replace')

# %%

# defense stats that can be added to the offensive player data
defense = fantasy_pros('DST').rename(columns={'player': 'team'})
d_stats = get_defense_stats(prev_years=1)
defense = pd.merge(defense, d_stats, on=['team', 'week', 'year'], how='inner')
defense = defense.dropna()
defense.columns = [f'def_{c}' if 'def' not in c else c for c in defense.columns]
defense = defense.rename(columns={'def_team': 'defTeam', 'def_week': 'week', 'def_year': 'year'})


# pre-game data
output = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:
    
    df = fantasy_pros(pos); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])
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

    df.loc[df.fd_salary < 100, 'fd_salary'] = np.nan

    compare_cols = ['fp_rank', 'projected_points',
                    'dk_salary', 'fd_salary', 'yahoo_salary', 'fantasyPoints',
                    'fantasyPointsRank', 'ProjPts', 'expertConsensus']
    df = add_player_comparison(df, compare_cols)

    df = pd.merge(df, defense, on=['defTeam', 'week', 'year']); print(df.shape[0])

    pff_def = pff_defense_rollup().rename(columns={'team': 'defTeam'})
    df = pd.merge(df, pff_def, on=['defTeam', 'week', 'year']); print(df.shape[0])

    pff_oline = pff_oline_rollup()
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])


    df = attach_y_act(df, pos)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df, pos); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df = df.dropna().reset_index(drop=True); print(df.shape[0])

    df['pos'] = pos
    if pos=='QB': df = one_qb_per_week(df); print(df.shape[0])

    print('Data Size:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())
    
    output = pd.concat([output, df], axis=0)

output = def_pts_allowed(output); print(output.shape[0])
output = remove_non_uniques(output)

dm.write_to_db(output, 'Model_Features', 'Backfill', 'replace')

#%%

output.loc[(output.week==13) & (output.year==2022), ['player', 'y_act']]
#%%

backfill_chk = dm.read(f"SELECT player FROM Backfill WHERE week={WEEK} AND year={YEAR}", 'Model_Features').player.values
sal = dm.read(f"SELECT player, salary FROM Salaries WHERE league={WEEK} AND year={YEAR}", 'Simulation')
sal[~sal.player.isin(backfill_chk)].sort_values(by='salary', ascending=False).iloc[:50]

#%%
count_chk = dm.read(f"SELECT player, week, year, count(*) cnts FROM Backfill GROUP BY player, week, year", 'Model_Features')
count_chk[count_chk.cnts > 1]
#%%
bf = dm.read("SELECT * FROM Backfill", 'Model_Features')
bf = bf[~((bf.player=='Josh Johnson') & (bf.week==16) & (bf.year==2021))].reset_index(drop=True)
dm.write_to_db(bf, 'Model_Features', 'Backfill', 'Model_Features')
# %%
# TO DO LIST
# - add in PFF scores
# - add in snaps and snap share
# - Market share in terms of projected FP, dk salary, etc

#%%
output['avg_pts'] = output[['ProjPts', 'fantasyPoints', 'projected_points']].mean(axis=1)
output = output.sort_values(by=['year', 'week', 'team', 'avg_pts'],
                            ascending=[True, True, True, False]).reset_index(drop=True)

team_pts = output.groupby(['year', 'week', 'team']).agg({'avg_pts': 'sum', 'y_act': 'sum'}).reset_index()

team_off = dm.read("SELECT * FROM Defense_Data", 'Model_Features').drop('y_act', axis=1)
team_off = team_off.rename(columns={'player': 'defTeam', 'offTeam': 'team'})
team_off = pd.merge(team_pts, team_off, on=['team', 'week', 'year'])
team_off = team_off.rename(columns={'team': 'player'})
team_off['team'] = team_off.player

print('Unique team-week-years:', team_off[['player', 'week', 'year']].drop_duplicates().shape[0])
print('Team Counts by Week:', team_off[['year', 'week', 'player']].drop_duplicates().groupby(['year', 'week'])['player'].count())

dm.write_to_db(team_off, 'Model_Features', f'Team_Offense_Data', if_exist='replace')

#%%
cur_pos = 'RB'

dk_sal = dm.read('''SELECT player, team, week, year, dk_salary
                    FROM Daily_Salaries
                    WHERE dk_salary > 5500 
                          AND position='QB'
                    UNION
                    SELECT player, team, week, year, dk_salary
                    FROM Daily_Salaries
                    WHERE dk_salary > 4500 
                          AND position!='QB' ''', "Pre_PlayerData")

pff = dm.read('''SELECT player, offTeam team, week, year, expertConsensus, fantasyPoints, `Proj Pts` ProjPts
                    FROM PFF_Expert_Ranks
                    JOIN (SELECT player, week, year, fantasyPoints
                        FROM PFF_Proj_Ranks)
                        USING (player, week, year) ''', "Pre_PlayerData")

inj = dm.read('''SELECT player, week, year, 1 as is_out
                 FROM PlayerInjuries
                 WHERE game_status IN ('Out', 'Doubtful') 
                       AND pos in ('QB', 'RB', 'WR', 'TE')''', 'Pre_PlayerData')

data = pd.merge(dk_sal, pff, on=['player', 'team', 'week', 'year'], how='left')
data = pd.merge(data, inj, on=['player',  'week', 'year'], how='left')
data.is_out = data.is_out.fillna(0)
data
# missing_game = data.loc[(data.is_out==1) | (data.expertConsensus.isnull()),
#                         ['player', 'team', 'week', 'year', 'dk_salary']]

# pos = dm.read('''SELECT DISTINCT player, team, year, pos
#                  FROM FantasyPros
#                  ''', "Pre_PlayerData")

# missing_game = pd.merge(missing_game, pos, on=['player', 'team', 'year'])
# missing_game = missing_game.groupby(['team', 'pos', 'week', 'year']).agg({'dk_salary': 'sum'}).reset_index()
# missing_game = missing_game.rename(columns={'dk_salary': 'missing_salary'})
# missing_game_pos = missing_game[missing_game.pos==cur_pos].drop('pos', axis=1)

# xx = pd.merge(df, missing_game_pos, on=['team', 'week', 'year'], how='left').fillna({'missing_salary': 0})

# missing_game[missing_game.team=='SEA'].iloc[:50]
# %%

# to_agg = {c: [np.mean, np.max, np.min] for c in cols}
# team_stats = df.groupby(['team', 'week', 'year']).agg(to_agg)

# diff_df = df[['player', 'team', 'week', 'year']].drop_duplicates()
# for c in cols:
#     tmp_df = team_stats[c].reset_index()
#     tmp_df = pd.merge(tmp_df, df[['player', 'team', 'week', 'year', c]], on=['team', 'week', 'year'])

#     for a in ['mean', 'amin', 'amax']:
#         tmp_df[f'{c}_{a}_diff'] = tmp_df[c] - tmp_df[a]

#     tmp_df = tmp_df[['player', 'team', 'week', 'year', f'{c}_mean_diff', f'{c}_amax_diff', f'{c}_amin_diff']]
#     diff_df = pd.merge(diff_df, tmp_df, on=['player', 'team', 'week', 'year'])
    
# diff_df = diff_df.drop_duplicates()
# team_stats.columns = [f'{c[0]}_{c[1]}' for c in team_stats.columns]
# team_stats = team_stats.reset_index().drop_duplicates()

# df = pd.merge(df, team_stats, on=['team', 'week', 'year'])
# df = pd.merge(df, diff_df, on=['player', 'team', 'week', 'year'])

# #%%

# df = dm.read('''SELECT * FROM WR_Stats WHERE season>=2020 AND rec_yards_gained_sum > 10 ''', 'FastR')
# df.fantasy_pts.plot.hist()

#%%

def fantasy_pros_new(pos):

    fp = dm.read(f'''SELECT * 
                    FROM FantasyPros 
                    WHERE pos='{pos}' 
                          AND team is NOT NULL''', 'Pre_PlayerData')
    fp = name_cleanup(fp)
    if pos == 'DST': fp = fp.drop('player', axis=1).rename(columns={'team': 'player'})

    return fp


def ffa_compile(df, table_name, pos):
    
    if table_name == 'FFA_Projections':
        cols = ['player','week', 'year', 'ffa_points', 'ffa_sd_pts',
                'ffa_dropoff', 'ffa_floor', 'ffa_ceiling', 'ffa_points_vor', 'ffa_floor_vor', 'ffa_ceiling_vor',
                'ffa_rank', 'ffa_floor_rank', 'ffa_ceiling_rank', 'ffa_position_rank', 'ffa_tier', 'ffa_uncertainty']
        
    elif table_name == 'FFA_RawStats':
        if pos == 'QB':
            cols = ['player', 'week', 'year',
                    'ffa_pass_yds', 'ffa_pass_yds_sd', 'ffa_pass_tds', 'ffa_pass_tds_sd', 'ffa_pass_int',
                    'ffa_pass_int_sd', 'ffa_rush_yds', 'ffa_rush_yds_sd', 'ffa_rush_tds', 'ffa_rush_tds_sd',
                    ]
        elif pos in ('RB', 'WR', 'TE'):
            cols =  ['player', 'week', 'year', 'ffa_rush_yds', 'ffa_rush_yds_sd', 'ffa_rush_tds', 'ffa_rush_tds_sd',
                     'ffa_rec', 'ffa_rec_sd', 'ffa_rec_yds', 'ffa_rec_tds']

    ffa = dm.read(f"SELECT * FROM {table_name} WHERE position='{pos}'", 'Pre_PlayerData')
    ffa = ffa[~((ffa.week==6) & (ffa.year==2020))].reset_index(drop=True)
    ffa = ffa[cols]

    df = pd.merge(df, ffa, on=['player', 'week', 'year'], how='left')

    return df


def pff_experts_new(df, pos):

    experts = dm.read(f'''SELECT player, week, year, a.defTeam,
                            fantasyPoints,  fantasyPointsRank,
                            `Proj Pts` ProjPts,
                            passComp, passAtt, passYds, passTd, passInt, passSacked,
                            rushAtt, rushYds, rushTd, recvTargets,
                            recvReceptions, recvYds, recvTd,
                            fumbles, fumblesLost, twoPt, returnYds, returnTd,
                            expertConsensus, expertNathanJahnke, expertIanHartitz,
                            rankadj_expertConsensus, rankadj_expertNathanJahnke,
                            playeradj_expertNathanJahnke,playeradj_expertConsensus 
                       
                    FROM PFF_Proj_Ranks a
                    JOIN (SELECT *
                            FROM PFF_Expert_Ranks 
                            WHERE Position='{pos}' )
                            USING (player, week, year)
                    ''', 'Pre_PlayerData')
    
    df = pd.merge(df, experts, on=['player', 'week', 'year'], how='left')

    return df

def fantasy_cruncher(df, pos):
    fc = dm.read(f"SELECT * FROM FantasyCruncher WHERE pos='{pos}'", 'Pre_PlayerData')

    if pos!='DST':
        cols = ['player', 'week', 'year',
                'fc_proj_passing_stats_att', 'fc_proj_passing_stats_yrds', 'fc_proj_passing_stats_tds',
                'fc_proj_passing_stats_int', 'fc_proj_rushing_stats_pct', 'fc_proj_rushing_stats_att', 
                'fc_proj_rushing_stats_yrds', 'fc_proj_rushing_stats_tds', 'fc_proj_rushing_stats_att_tar',
                'fc_proj_receiving_stats_pct', 'fc_proj_receiving_stats_tar', 'fc_proj_receiving_stats_rec',
                'fc_proj_receiving_stats_yrds', 'fc_proj_receiving_stats_tds', 'fc_proj_fantasy_pts_fc', 
                'fc_projected_values_floor', 'fc_projected_values_ceiling']
    else:
        cols = ['player', 'week', 'year',
                'fc_proj_defensive_stats_int', 'fc_proj_defensive_stats_fum',
                'fc_proj_defensive_stats_sfty', 'fc_proj_defensive_stats_tds',
                'fc_proj_defensive_stats_pts',  'fc_proj_defensive_stats_sacks',
                'fc_proj_fantasy_pts_fc', 'fc_projected_values_floor', 'fc_projected_values_ceiling']

    fc = fc[cols]
    fc = fc.sort_values(by=['year', 'week', 'fc_proj_fantasy_pts_fc'], ascending=[True, True, False]).reset_index(drop=True)
    fc['fc_rank'] = fc.groupby(['year', 'week']).cumcount().values
    df = pd.merge(df, fc, on=['player', 'week', 'year'], how='left')
    df = dc.convert_to_float(df)
    df[['week', 'year']] = df[['week', 'year']].astype('int')

    return df

def consensus_fill(df):
    to_fill = {

        # stat fills
        'avg_proj_pass_yds': ['passYds', 'ffa_pass_yds', 'fc_proj_passing_stats_yrds'],
        'avg_proj_pass_td': ['passTd', 'ffa_pass_tds', 'fc_proj_passing_stats_tds'],
        'avg_proj_pass_int': ['passInt', 'ffa_pass_int', 'fc_proj_passing_stats_int'],
        'avg_proj_pass_att': ['passAtt', 'fc_proj_passing_stats_att'],
        'avg_proj_rush_yds': ['rushYds', 'ffa_rush_yds', 'fc_proj_rushing_stats_yrds'],
        'avg_proj_rush_att': ['rushAtt', 'fc_proj_rushing_stats_att'],
        'avg_proj_rush_td': ['rushTd', 'ffa_rush_tds', 'fc_proj_rushing_stats_tds'],
        'avg_proj_rec': ['recvReceptions', 'ffa_rec', 'fc_proj_receiving_stats_rec'],
        'avg_proj_rec_yds': ['recvYds', 'ffa_rec_yds', 'fc_proj_receiving_stats_yrds'],
        'avg_proj_rec_td': ['recvTd', 'ffa_rec_tds', 'fc_proj_receiving_stats_tds'],
        'avg_proj_rec_tgts': ['recvTargets', 'fc_proj_receiving_stats_tar'],

        # point and rank fills
        'avg_proj_points': ['projected_points', 'fantasyPoints', 'ProjPts', 'ffa_points', 'fc_proj_fantasy_pts_fc'],
        'avg_proj_rank': ['fp_rank', 'rankadj_fp_rank', 'playeradj_fp_rank', 'expertConsensus', 'expertNathanJahnke', 
                          'expertIanHartitz', 'rankadj_expertConsensus', 'rankadj_expertNathanJahnke', 'fantasyPointsRank',
                          'playeradj_expertNathanJahnke', 'playeradj_expertConsensus', 'ffa_position_rank', 'fc_rank']
    }

    for k, tf in to_fill.items():

        # find columns that exist in dataset
        tf = [c for c in tf if c in df.columns]
        
        # fill in nulls based on available data
        for c in tf:
            df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), tf].mean(axis=1)
        
        # fill in the average for all cols
        df[k] = df[tf].mean(axis=1)
    
    return df


def fill_ratio_nulls(df):
    ratio_fill_cols = ['ffa_sd_pts', 'ffa_dropoff', 'ffa_floor', 'ffa_ceiling', 'ffa_points_vor', 'ffa_floor_vor',
                        'ffa_ceiling_vor', 'ffa_rank', 'ffa_floor_rank', 'ffa_ceiling_rank', 'ffa_rec_sd',
                        'ffa_tier', 'ffa_uncertainty','ffa_pass_yds_sd', 'ffa_pass_tds_sd', 'ffa_pass_int_sd',
                        'ffa_rush_yds_sd',  'ffa_rush_tds_sd', 'fc_proj_rushing_stats_pct', 'fc_proj_rushing_stats_att_tar', 
                        'fc_proj_receiving_stats_pct', 'fc_projected_values_floor', 'fc_projected_values_ceiling']
    for c in ratio_fill_cols:
        if c in df.columns:
            fill_ratio = (df[c] / (df['ffa_points']+1)).mean()
            df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), 'ffa_points'] * fill_ratio + fill_ratio
    return df

def log_rank_cols(df):
    rank_cols = [c for c in df.columns if 'rank' in c or 'expert' in c]
    for c in rank_cols:
        df['log_' + c] = np.log(df[c]+1)
    return df

def rolling_proj_stats(df):
    df = forward_fill(df)
    proj_cols = [c for c in df.columns if 'ffa' in c or 'rank' in c or 'fc' in c or 'proj' in c \
                or 'expert' in c or 'Pts' in c or 'Points' in c or 'points' in c]
    df = add_rolling_stats(df, ['player'], proj_cols)
    return df

def add_ffa_defense(df):

    ffa = dm.read('''SELECT * 
                    FROM FFA_Projections
                    WHERE position=='DST' 
                    ''', 'Pre_PlayerData').drop(['ffa_adp','ffa_aav'], axis=1)
    ffa = ffa[~((ffa.week==6) & (ffa.year==2020))].reset_index(drop=True)
    ffa = ffa.rename(columns={'player': 'defTeam'})


    ffa_stats = dm.read('''SELECT * 
                    FROM FFA_RawStats
                    WHERE position=='DST' 
                    ''', 'Pre_PlayerData')
    ffa_stats = ffa_stats[['player', 'week', 'year', 'ffa_dst_int', 'ffa_dst_int_sd',
                           'ffa_dst_sacks', 'ffa_dst_sacks_sd', 'ffa_dst_safety',
                           'ffa_dst_safety_sd', 'ffa_dst_td', 'ffa_dst_td_sd']]
    ffa_stats = ffa_stats[~((ffa_stats.week==6) & (ffa_stats.year==2020))].reset_index(drop=True)
    ffa_stats = ffa_stats.rename(columns={'player': 'defTeam'})

    df = pd.merge(df, ffa, on=['defTeam', 'week', 'year'], how='left')
    df = pd.merge(df, ffa_stats, on=['defTeam', 'week', 'year'], how='left')

    return df

def show_corrs(df):
    corrs = df.corr()['y_act'].dropna().sort_values()
    display(corrs.iloc[:25])
    display(corrs.iloc[-25:])

#%%

# defense stats that can be added to the offensive player data
defense = fantasy_pros('DST').rename(columns={'player': 'team'})
d_stats = get_defense_stats(prev_years=1)
defense = pd.merge(defense, d_stats, on=['team', 'week', 'year'], how='inner')
defense = defense.dropna()
defense.columns = [f'def_{c}' if 'def' not in c else c for c in defense.columns]
defense = defense.rename(columns={'def_team': 'defTeam', 'def_week': 'week', 'def_year': 'year'})
defense = add_ffa_defense(defense).drop('team', axis=1)
pff_def = add_team_matchups()
defense = pd.merge(defense, pff_def, on=['defTeam', 'year', 'week'])

defense = forward_fill(defense.rename(columns={'defTeam': 'player'}))
defense = defense.rename(columns={'player': 'defTeam'})
defense.isnull().sum()[defense.isnull().sum()>0]

#%%
for pos in ['RB', 'WR', 'TE']:
    df = fantasy_pros_new(pos); print(df.shape[0])
    df = pff_experts_new(df, pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fantasy_cruncher(df, pos); print(df.shape[0])

    df = consensus_fill(df); print(df.shape[0])
    df = fill_ratio_nulls(df); print(df.shape[0])
    df = log_rank_cols(df); print(df.shape[0])
    df = rolling_proj_stats(df); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])

    df = add_fp_rolling(df, pos); print(df.shape[0])
    df = get_salaries(df, pos); print(df.shape[0])
    df = add_pfr_matchup(df); print(df.shape[0])
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

    df = get_player_data(df, pos, YEAR, prev_years=1); print(df.shape[0])

    team_stats = get_team_stats(YEAR, prev_years=1)
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print('team_stats', df.shape[0])

    df = calc_market_share(df); print(df.shape[0])
    df = add_rz_stats(df); print(df.shape[0])
    df = advanced_rec_stats(df)
    if pos in ('WR', 'TE'):
        df = add_next_gen(df, pos, 'Receiving'); print('next_gen', df.shape[0])
    if pos == 'RB': 
        df = advanced_rb_stats(df)
        df = add_next_gen(df, pos, 'Rushing'); print(df.shape[0])

    team_qb = get_max_qb()
    df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])

    # pre game data reliant on team information
    compare_cols = ['fp_rank', 'dk_salary', 'expertConsensus', 'projected_points', 'fantasyPoints', 'ProjPts',
                    'ffa_points', 'fc_proj_fantasy_pts_fc', 'ffa_position_rank', 'fc_rank', 'avg_proj_rank', 'avg_proj_points']
    df = add_player_comparison(df, compare_cols); print(df.shape[0])

    df = pd.merge(df, defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = def_pts_allowed(df); print(df.shape[0])

    pff_def = pff_defense_rollup().rename(columns={'team': 'defTeam'})
    df = pd.merge(df, pff_def, on=['defTeam', 'week', 'year']); print(df.shape[0])

    pff_oline = pff_oline_rollup()
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])

    df = attach_y_act(df, pos)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df, pos); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df = df.dropna().reset_index(drop=True); print(df.shape[0])
    df = remove_non_uniques(df)

    print('Total Rows:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())

    dm.write_to_db(df.iloc[:, :2000], 'Model_Features', f'{pos}_Data', if_exist='replace')
    if df.shape[1] > 2000:
        dm.write_to_db(df.iloc[:, 2000:], 'Model_Features', f'{pos}_Data2', if_exist='replace')

#%%

pos = 'QB'
rush_or_pass = ''

# pre-game data
df = fantasy_pros_new(pos); print(df.shape[0])
df = pff_experts_new(df, pos); print(df.shape[0])
df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
df = fantasy_cruncher(df, pos); print(df.shape[0])

df = consensus_fill(df); print(df.shape[0])
df = fill_ratio_nulls(df); print(df.shape[0])
df = log_rank_cols(df); print(df.shape[0])
df = rolling_proj_stats(df); print(df.shape[0])
df, _ = add_injuries(df, pos); print(df.shape[0])

df = add_fp_rolling(df, pos); print(df.shape[0])
df = get_salaries(df, pos); print(df.shape[0])
df = add_pfr_matchup(df); print(df.shape[0])
df = add_gambling_lines(df); print(df.shape[0])
df = add_weather(df); print(df.shape[0])
dst = add_team_matchups().drop('offTeam', axis=1)
df = pd.merge(df, dst, on=['defTeam', 'year', 'week']); print(df.shape[0])


# post-game data
df = get_player_data(df, pos, YEAR, prev_years=1); print(df.shape[0])

team_stats = get_team_stats(YEAR)
df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print( df.shape[0])

df = add_rz_stats_qb(df); print(df.shape[0])
df = add_qbr(df); print(df.shape[0])
df = add_qb_adv(df); print(df.shape[0])
df = add_next_gen(df, pos, 'Passing'); print(df.shape[0])

# get the positional values for the team
pos_values = positional_values()
df = pd.merge(df, pos_values, on=['team', 'week', 'year']); print(df.shape[0])

df = pd.merge(df, defense, on=['defTeam', 'week', 'year']); print('dstats', df.shape[0])
df = def_pts_allowed(df); print(df.shape[0])

pff_def = pff_defense_rollup().rename(columns={'team': 'defTeam'})
df = pd.merge(df, pff_def, on=['defTeam', 'week', 'year']); print(df.shape[0])

pff_oline = pff_oline_rollup()
df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])

df = attach_y_act(df, pos, rush_or_pass=rush_or_pass)
df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
df = projected_pts_vs_predicted(df, pos); print(df.shape[0])

# fill in missing data and drop any remaining rows
df = forward_fill(df)
df = df.dropna().reset_index(drop=True); print(df.shape[0])

df = one_qb_per_week(df); print(df.shape[0])

df = remove_non_uniques(df)
df = df[(df.ProjPts > 10) & (df.projected_points > 10)].reset_index(drop=True)

print('Total Rows:', df.shape[0])
print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())

dm.write_to_db(df.iloc[:,:2000], 'Model_Features', f"QB_Data{rush_or_pass.replace('_', '')}", if_exist='replace')
if df.shape[1] > 2000:
    dm.write_to_db(df.iloc[:,2000:], 'Model_Features', f"QB_Data{rush_or_pass.replace('_', '')}2", if_exist='replace')

#%%

# pre-game data
output = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:

    df = fantasy_pros_new(pos); print(df.shape[0])
    df = pff_experts_new(df, pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fantasy_cruncher(df, pos); print(df.shape[0])

    df = consensus_fill(df); print(df.shape[0])
    df = fill_ratio_nulls(df); print(df.shape[0])
    df = log_rank_cols(df); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])

    df = get_salaries(df, pos); print(df.shape[0])
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

    df.loc[df.fd_salary < 100, 'fd_salary'] = np.nan

    compare_cols = ['fp_rank', 'projected_points',
                    'dk_salary', 'fd_salary', 'yahoo_salary', 'fantasyPoints',
                    'fantasyPointsRank', 'ProjPts', 'expertConsensus']
    df = add_player_comparison(df, compare_cols)

    df = pd.merge(df, defense, on=['defTeam', 'week', 'year']); print(df.shape[0])

    pff_def = pff_defense_rollup().rename(columns={'team': 'defTeam'})
    df = pd.merge(df, pff_def, on=['defTeam', 'week', 'year']); print(df.shape[0])

    pff_oline = pff_oline_rollup()
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])


    df = attach_y_act(df, pos)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df, pos); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df = df.dropna().reset_index(drop=True); print(df.shape[0])

    df['pos'] = pos
    if pos=='QB': df = one_qb_per_week(df); print(df.shape[0])

    print('Data Size:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())
    
    output = pd.concat([output, df], axis=0)

output = def_pts_allowed(output); print(output.shape[0])
output = remove_non_uniques(output)

dm.write_to_db(output, 'Model_Features', 'Backfill', 'replace')

#%%

backfill_chk = dm.read(f"SELECT player FROM Backfill WHERE week={WEEK} AND year={YEAR}", 'Model_Features').player.values
sal = dm.read(f"SELECT player, salary FROM Salaries WHERE league={WEEK} AND year={YEAR}", 'Simulation')
sal[~sal.player.isin(backfill_chk)].sort_values(by='salary', ascending=False).iloc[:50]

#%%
count_chk = dm.read(f"SELECT player, week, year, count(*) cnts FROM Backfill GROUP BY player, week, year", 'Model_Features')
count_chk[count_chk.cnts > 1]

#%%
count_chk = dm.read(f"SELECT player, week, year, count(*) cnts FROM RB_Data GROUP BY player, week, year", 'Model_Features')
count_chk[count_chk.cnts > 1]

#%%
bf = dm.read("SELECT * FROM Backfill", 'Model_Features')
bf = bf[~((bf.player=='Josh Johnson') & (bf.week==16) & (bf.year==2021))].reset_index(drop=True)
dm.write_to_db(bf, 'Model_Features', 'Backfill', 'replace')


#%%

to_agg = {c: [np.mean, np.max, np.min, 'count'] for c in compare_cols}
team_stats = df.groupby(['team', 'week', 'year']).agg(to_agg)

diff_df = df[['player', 'team', 'week', 'year']].drop_duplicates()
for c in compare_cols:
    tmp_df = team_stats[c].reset_index()
    tmp_df = pd.merge(tmp_df, df[['player', 'team', 'week', 'year', c]], on=['team', 'week', 'year'])

    for a in ['mean', 'amin', 'amax']:
        tmp_df[f'{c}_{a}_diff'] = tmp_df[c] - tmp_df[a]

    tmp_df = tmp_df[['player', 'team', 'week', 'year', f'{c}_mean_diff', f'{c}_amax_diff', f'{c}_amin_diff']]
    diff_df = pd.merge(diff_df, tmp_df, on=['player', 'team', 'week', 'year'])
#%%

to_agg = {c: [np.mean, np.max, np.min, 'count'] for c in compare_cols}
team_stats = df.groupby(['team', 'week', 'year']).agg(to_agg)

diff_df = df[['player', 'team', 'week', 'year']].drop_duplicates()
for c in compare_cols:
    tmp_df = team_stats[c].reset_index()
    tmp_df = pd.merge(tmp_df, df[['player', 'team', 'week', 'year', c]], on=['team', 'week', 'year'])

# %%


# defense stats that can be added to the offensive player data
defense = fantasy_pros('DST').rename(columns={'player': 'team'})
d_stats = get_defense_stats(prev_years=1)
defense = pd.merge(defense, d_stats, on=['team', 'week', 'year'], how='inner')
defense = defense.dropna()
defense.columns = [f'def_{c}' if 'def' not in c else c for c in defense.columns]
defense = defense.rename(columns={'def_team': 'defTeam', 'def_week': 'week', 'def_year': 'year'})
defense = add_ffa_defense(defense)
pff_def = add_team_matchups()
defense = pd.merge(defense, pff_def, on=['defTeam', 'year', 'week'])

#%%

defense = fantasy_cruncher(defense, 'DST')
# %%
[c for c in defense if 'sack' in c or 'Sack' in c]
# %%
{
'proj_int': ['def_interception', 'ffa_dst_int', 'fc_proj_defensive_stats_int'],
'proj_fumble': ['rmean3_def_fumble_sum', 'fc_proj_defensive_stats_fum'],
'proj_sack': ['def_rmean3_sack', 'ffa_dst_sacks', 'fc_proj_defensive_stats_sacks'],


}