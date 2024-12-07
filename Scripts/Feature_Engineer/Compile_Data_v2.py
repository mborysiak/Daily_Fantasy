#%%

YEAR = 2024
WEEK = 14

import pandas as pd 
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
import warnings
from scipy.stats import poisson, truncnorm
from typing import Dict

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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


def add_rolling_stats(df, gcols, rcols, perform_check=True):

    df = df.sort_values(by=[gcols[0], 'year', 'week']).reset_index(drop=True)

    if perform_check:
        cnt_check = df.groupby([gcols[0], 'year'])['week'].count()
        print(f'Counts of Groupby Category Over 17: {cnt_check[cnt_check>17]}')

    rolls3_mean = rolling_stats(df, gcols, rcols, 3, agg_type='mean')
    rolls3_max = rolling_stats(df, gcols, rcols, 3, agg_type='max')

    rolls8_mean = rolling_stats(df, gcols, rcols, 8, agg_type='mean')
    rolls8_max = rolling_stats(df, gcols, rcols, 8, agg_type='max')
    rolls8_std = rolling_stats(df, gcols, rcols, 8, agg_type='std')

    # hist_mean = rolling_expand(df, gcols, rcols, agg_type='mean')
    # hist_std = rolling_expand(df, gcols, rcols, agg_type='std')
    # hist_p80 = rolling_expand(df, gcols, rcols, agg_type='p95')

    df = pd.concat([df, 
                    rolls8_mean, rolls8_max, rolls8_std,
                    rolls3_mean, rolls3_max,
                    #hist_mean, hist_std, hist_p80
                    ], axis=1)

    return df


def switch_seasons(df):

    # use week 16 (adjusted to 17 prior to input) as the next season week.
    # must switch if running week 17 of current year
    idx = df.loc[(df.week==17) & (df.year != YEAR)].index
    df.loc[idx, 'year'] = df.loc[idx, 'year'] + 1
    df.loc[idx, 'week'] = 1

    # # any seasons 2020 or earlier when it was a 17 week season, convert week 17 to the first week of the next year
    # # note that the year is +1 and week is set to one in a single step
    # idx = df.loc[(df.week==17) & (df.year <= 2020)].index
    # df.loc[idx, 'year'] = df.loc[idx, 'year'] + 1
    # df.loc[idx, 'week'] = 1
    
    # # for any seasons after 2020, set week 18 to the following year first week.
    # # however, don't do the conversion for the current year since you need to keep current week for this year's predictions
    # idx = df.loc[(df.week==18) & (df.year >= 2021) & (df.year != YEAR)].index
    # df.loc[idx, 'year'] = df.loc[idx, 'year'] + 1
    # df.loc[idx, 'week'] = 1

    return df



def forward_fill(df, cols=None):
    
    if cols is None: cols = df.columns
    df = df.sort_values(by=['player', 'year', 'week'])
    df = df.groupby('player', as_index=False)[cols].fillna(method='ffill')
    df = df.sort_values(by=['player', 'year', 'week'])

    return df

def find_bye_weeks():

    df = dm.read(f'''SELECT team, week, year, week+1 next_week
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
    
    # # fix the cancelled game due to injury
    # cin_buf_fix = pd.DataFrame({'team': ['CIN','BUF'],
    #                         'week': [17, 17],
    #                         'year': [2022, 2022],
    #                         'next_week': [18, 18]})
    # df = pd.concat([df, cin_buf_fix], axis=0).reset_index(drop=True)
    return df


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


#---------------
# Pre Game Data
#---------------



def fantasy_pros_new(pos):

    fp = dm.read(f'''SELECT * 
                    FROM FantasyPros 
                    WHERE pos='{pos}' 
                          AND team is NOT NULL
                         ''', 'Pre_PlayerData')
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


def fftoday_proj(df, pos):
    fft = dm.read(f"SELECT * FROM FFToday_Projections WHERE pos='{pos}'", 'Pre_PlayerData').drop(['team','pos'], axis=1)
    fft['fft_rank'] = fft.groupby(['year', 'week'])['fft_proj_pts'].rank(ascending=False, method='min')
    df = pd.merge(df, fft, on=['player', 'week','year'], how='left')
    return df

def numberfire_proj(df):
    nf = dm.read(f"SELECT * FROM NumberFire", 'Pre_PlayerData')
    df = pd.merge(df, nf, on=['player', 'week','year'], how='left')

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


def fantasy_data_proj(df, pos):
    
    fd = dm.read(f'''SELECT * 
                     FROM FantasyData
                     WHERE position='{pos}' 
                    ''', 'Pre_PlayerData')
    
    fd = fd.drop(['team', 'opp', 'position', 'fdta_rank', 'fdta_sack', 'fdta_int', 'fdta_fum_forced', 
                  'fdta_fum_rec', 'fdta_dk_points_per_game'], axis=1)

    fd['fdta_rank'] = fd.groupby(['year', 'week'])['fdta_proj_points'].rank(ascending=False, method='min')
    
    df = pd.merge(df, fd, on=['player', 'week', 'year'], how='left')

    return df

def fantasy_data_proj_def(df):
    
    fd = dm.read(f'''SELECT * 
                     FROM FantasyData_Defense
                    ''', 'Pre_PlayerData')
    
    fd = fd.drop(['opp', 'fdta_rank'], axis=1)

    fd['fdta_rank'] = fd.groupby(['year', 'week'])['fdta_proj_points'].rank(ascending=False, method='min')
    
    df = pd.merge(df, fd, on=['player', 'week', 'year'], how='left')

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

def etr_projection(df, pos):

    etr = dm.read(f'''SELECT player, week, year,
                            etr_proj_points,
                            etr_proj_floor,
                            etr_proj_ceiling 
                    FROM ETR_Projections 
                    LEFT JOIN (
                        SELECT player, week, year, etr_proj_floor
                        FROM ETR_Projections_DK
                    ) USING (player, week, year)
                    WHERE pos='{pos}'
                ''', 'Pre_PlayerData')
    
    etr_stats = dm.read(f'''SELECT * 
                            FROM ETR_Projections_Detail
                            WHERE pos='{pos}'
                        ''', 'Pre_PlayerData')
    etr_stats = etr_stats.drop(['team', 'opp', 'pos'], axis=1)

    df = pd.merge(df, etr, on=['player', 'week', 'year'], how='left')
    df = pd.merge(df, etr_stats, on=['player', 'week', 'year'], how='left')

    return df

def etr_projection_def(df):

    etr = dm.read(f'''SELECT team player, week, year,
                            etr_proj_points,
                            etr_proj_floor,
                            etr_proj_ceiling 
                    FROM ETR_Projections 
                    LEFT JOIN (
                        SELECT team player, week, year, etr_proj_floor
                        FROM ETR_Projections_DK
                    ) USING (player, week, year)
                    WHERE pos='DST'
                ''', 'Pre_PlayerData')

    df = pd.merge(df, etr, on=['player', 'week', 'year'], how='left')

    return df

def fantasy_points_projection(df, pos):

    fpts = dm.read(f'''SELECT * 
                       FROM FantasyPoints
                       WHERE pos='{pos}'
                       ''', 'Pre_PlayerData')

    fpts = fpts.drop(['fpts_overall_rank', 'team', 'opp', 'pos',  'fpts_up', 'fpts_down', 'fpts_move', 'fpts_ww', 'fpts_inj',
                      'fpts_pass_yds_per_att', 'fpts_pass_yds_per_cmp', 'fpts_rush_yds_per_att', 'fpts_rec_yds_per_rec'], axis=1)
    fpts['fpts_rank'] = fpts.groupby(['year', 'week'])['fpts_proj_points'].rank(ascending=False, method='min')
    
    df = pd.merge(df, fpts, on=['player', 'week', 'year'], how='left')

    return df

def fantasy_points_projection_def(df):

    fpts = dm.read(f'''SELECT player, week, year, fpts_proj_points, fpts_int, 
                              fpts_fum_rec, fpts_sack, fpts_def_td + fpts_special_td fpts_td
                       FROM FantasyPoints_Defense
                       ''', 'Pre_PlayerData')

    fpts['fpts_rank'] = fpts.groupby(['year', 'week'])['fpts_proj_points'].rank(ascending=False, method='min')
    
    df = pd.merge(df, fpts, on=['player', 'week', 'year'], how='left')

    return df

team_d_map = {
 'Arizona Cardinals': 'ARI','Arizona': 'ARI','Ari': 'ARI',
 'Atlanta Falcons': 'ATL','Atlanta': 'ATL','Atl': 'ATL', 'Atl Falcons': 'ATL',
 'Baltimore Ravens': 'BAL','Baltimore': 'BAL','Bal': 'BAL',
 'Buffalo Bills': 'BUF','Buffalo': 'BUF','Buf': 'BUF',
 'Carolina Panthers': 'CAR','Carolina': 'CAR','Car': 'CAR',
 'Chicago Bears': 'CHI','Chicago': 'CHI','Chi': 'CHI',
 'Cincinnati Bengals': 'CIN','Cincinnati': 'CIN', 'Cin': 'CIN', 'Cin Bengals': 'CIN',
 'Cleveland Browns': 'CLE','Cleveland': 'CLE','Cle': 'CLE','Cle Browns': 'CLE',
 'Dallas Cowboys': 'DAL','Dallas': 'DAL', 'Dal': 'DAL',
 'Denver Broncos': 'DEN','Denver': 'DEN','Den': 'DEN',
 'Detroit Lions': 'DET','Detroit': 'DET', 'Det': 'DET',
 'Green Bay Packers': 'GB','Green Bay': 'GB','Gb': 'GB',
 'Houston Texans': 'HOU','Houston': 'HOU','Hou': 'HOU',
 'Indianapolis Colts': 'IND','Indianapolis': 'IND','Ind': 'IND',
 'Jacksonville Jaguars': 'JAC','Jacksonville': 'JAC','Jac': 'JAC',
 'Kansas City Chiefs': 'KC','Kansas City': 'KC','Kc': 'KC','Kansas City Cheifs': 'KC',
 'Los Angeles Chargers': 'LAC','La Chargers': 'LAC','Lac': 'LAC','Los Angeles Chargers Chargers': 'LAC',
 'Los Angeles Rams Rams': 'LAR','Los Angeles Rams': 'LAR','La Rams': 'LAR','Lar': 'LAR',
 'Miami Dolphins': 'MIA','Miami': 'MIA','Mia': 'MIA',
 'Minnesota Vikings': 'MIN','Minnesota': 'MIN','Min': 'MIN',
 'New England Patriots': 'NE','New England': 'NE','Ne': 'NE',
 'New Orleans Saints': 'NO','New Orleans': 'NO','No': 'NO', 'Saints': 'NO',
 'New York Giants': 'NYG','Ny Giants': 'NYG','Nyg': 'NYG',
 'New York Jets': 'NYJ','Ny Jets': 'NYJ','Nyj': 'NYJ','New York Jets Jets': 'NYJ','New York Giants Giants': 'NYG',
 'Oakland Raiders': 'LVR','Las Vegas Raiders': 'LVR','Las Vegas': 'LVR','Lvr': 'LVR',
 'Philadelphia Eagles': 'PHI','Philadelphia': 'PHI','Phi': 'PHI',
 'Pittsburgh Steelers': 'PIT','Pittsburgh': 'PIT','Pit': 'PIT',
 'San Francisco 49ers': 'SF','San Francisco': 'SF','Sf': 'SF','San Francisco 49Ers': 'SF',
 'Seattle Seahawks': 'SEA','Seattle': 'SEA', 'Sea': 'SEA',
 'Tampa Bay Buccaneers': 'TB','Tampa Bay': 'TB','Tb': 'TB',
 'Tennessee Titans': 'TEN','Tennessee': 'TEN','Ten': 'TEN',
 'Washington Redskins': 'WAS','Washington': 'WAS','Was': 'WAS',
 'Washington Football Team': 'WAS','Washington Commanders': 'WAS'}

def consensus_fill(df, is_dst=False):

    if is_dst:
        to_fill = {
                'proj_dst_int': ['def_rmean3_interception', 'dstInt', 'ffa_dst_int', 'fc_proj_defensive_stats_int', 'fdta_def_int', 'fpts_int'],
                'proj_dst_fumble': ['rmean3_def_fumble_sum', 'dstFumblesRecovered', 'fc_proj_defensive_stats_fum', 'fdta_fum_rec', 'fpts_fum_rec'],
                'proj_dst_sack': ['def_rmean3_sack', 'dstSacks', 'ffa_dst_sacks', 'fc_proj_defensive_stats_sacks', 'fdta_sack', 'fpts_sack'],
                'proj_dst_safety': ['def_rmean3_safety', 'dstSafeties', 'ffa_dst_safety', 'fc_proj_defensive_stats_sfty'],
                'proj_dst_td': ['rmean3_def_td', 'dstTd', 'ffa_dst_td', 'fc_proj_defensive_stats_tds', 'fdta_def_td', 'fpts_td'],
                'proj_dst_points': ['projected_points', 'ProjPts_dst', 'ffa_points', 'fc_proj_fantasy_pts_fc', 'fdta_proj_points', 'fpts_proj_points',
                                    'etr_proj_points'],
                'proj_dst_rank': ['fp_rank', 'rankadj_fp_rank', 'playeradj_fp_rank', 'ffa_position_rank',
                                  'expertConsensus_dst', 'fc_rank', 'fdta_rank', 'fpts_rank'],
                }
    else:
        to_fill = {

            # stat fills
            'proj_pass_yds': ['passYds', 'ffa_pass_yds', 'fft_pass_yds', 'fc_proj_passing_stats_yrds', 'fdta_pass_yds', 
                              'nf_passing_yds', 'etr_pass_yds', 'fpts_pass_yds'],
            'proj_pass_td': ['passTd', 'ffa_pass_tds', 'fft_pass_td', 'fc_proj_passing_stats_tds', 'fdta_pass_td', 
                             'nf_passing_tds', 'etr_pass_tds', 'fpts_pass_td'],
            'proj_pass_int': ['passInt', 'ffa_pass_int', 'fft_pass_int', 'fc_proj_passing_stats_int', 
                              'nf_passing_ints', 'etr_pass_int', 'fpts_pass_int'],
            'proj_pass_comp': ['etr_pass_cmp', 'fft_pass_comp', 'fpts_pass_cmp', 'passComp', 'fpts_pass_cmp'],
            'proj_pass_att': ['passAtt', 'fft_pass_att', 'fc_proj_passing_stats_att', 'etr_pass_att', 'fpts_pass_att'],
            'proj_rush_yds': ['rushYds', 'ffa_rush_yds', 'fft_rush_yds', 'fc_proj_rushing_stats_yrds', 'fdta_rush_yds', 
                              'nf_rushing_yds', 'etr_rush_yds', 'fpts_rush_yds'],
            'proj_rush_att': ['rushAtt', 'fft_rush_att', 'fc_proj_rushing_stats_att', 'nf_rushing_att', 'etr_rush_att', 'fpts_rush_att'],
            'proj_rush_td': ['rushTd', 'ffa_rush_tds', 'fft_rush_td', 'fc_proj_rushing_stats_tds', 'fdta_rush_td', 
                             'nf_rushing_tds', 'etr_rush_tds', 'fpts_rush_td'],
            'proj_rec': ['recvReceptions', 'ffa_rec', 'fft_rec', 'fc_proj_receiving_stats_rec', 'fdta_rec', 
                         'nf_receiving_rec', 'etr_rec', 'fpts_rec'],
            'proj_rec_yds': ['recvYds', 'ffa_rec_yds', 'fft_rec_yds', 'fc_proj_receiving_stats_yrds', 'fdta_rec_yds', 
                             'nf_receiving_yds', 'etr_rec_yds', 'fpts_rec_yds'],
            'proj_rec_td': ['recvTd', 'ffa_rec_tds', 'fft_rec_td', 'fc_proj_receiving_stats_tds', 'fdta_rec_td', 
                            'nf_receiving_tds', 'etr_rec_td', 'fpts_rec_td'],
            'proj_rec_tgts': ['recvTargets', 'fc_proj_receiving_stats_tar'],

            # point and rank fills
            'proj_points': ['projected_points', 'fantasyPoints', 'ProjPts', 
                            'ffa_points', 'fft_proj_pts', 'fc_proj_fantasy_pts_fc', 
                            'fdta_proj_points', 'nf_proj_points', 'etr_proj_points', 'fpts_proj_points'],
            'proj_floor': ['ffa_floor', 'fc_projected_values_floor', 'etr_proj_floor'],
            'proj_ceiling': ['ffa_ceiling', 'fc_projected_values_ceiling', 'etr_proj_ceiling'],

            'proj_rank': ['fp_rank',   'expertConsensus', 'expertNathanJahnke', 'expertIanHartitz',  'nf_ranks_pos',
                          'ffa_position_rank', 'fc_rank', 'fdta_rank', 'etr_rank', 'fft_rank', 'fpts_rank'],
            'player_adj_proj_rank': ['playeradj_expertNathanJahnke', 'playeradj_expertConsensus','playeradj_fp_rank'],
            'rank_adj_proj_rank': ['rankadj_expertConsensus', 'rankadj_expertNathanJahnke','rankadj_fp_rank'],
            'floor_rank': ['ffa_floor_rank', 'etr_floor_rank'],
            'ceiling_rank': ['ffa_ceiling_rank', 'etr_ceiling_rank'],
            }

    for k, tf in to_fill.items():

        # find columns that exist in dataset
        tf = [c for c in tf if c in df.columns]
        
        # fill in nulls based on available data
        for c in tf:
            df.loc[df[c].isnull(), c] = df.loc[df[c].isnull(), tf].mean(axis=1)
        
        # fill in the average for all cols
        df['avg_' + k] = df[tf].mean(axis=1)
        df['std_' + k] = df[tf].std(axis=1)

        if 'rank' in k:
            df['min' + k] = df[tf].min(axis=1)
        else:
            df['max_' + k] = df[tf].max(axis=1)

        if k == 'proj_points':
            etr_ratio_ceiling = (df['etr_proj_ceiling'] / (df['etr_proj_points']+0.1)).mean()
            df.loc[df.etr_proj_ceiling.isnull(), 'etr_proj_ceiling'] = df.loc[df.etr_proj_ceiling.isnull(), 'etr_proj_points'] * etr_ratio_ceiling
            
            etr_ratio_floor = (df['etr_proj_floor'] / (df['etr_proj_points']+0.1)).mean()
            df.loc[df.etr_proj_floor.isnull(), 'etr_proj_floor'] = df.loc[df.etr_proj_floor.isnull(), 'etr_proj_points'] * etr_ratio_floor

            fc_ratio_ceiling = (df['fc_projected_values_ceiling'] / (df['fc_proj_fantasy_pts_fc']+0.1)).mean()
            df.loc[df.fc_projected_values_ceiling.isnull(), 'fc_projected_values_ceiling'] = df.loc[df.fc_projected_values_ceiling.isnull(), 'fc_proj_fantasy_pts_fc'] * fc_ratio_ceiling

            fc_ratio_floor = (df['fc_projected_values_floor'] / (df['fc_proj_fantasy_pts_fc']+0.1)).mean()
            df.loc[df.fc_projected_values_floor.isnull(), 'fc_projected_values_floor'] = df.loc[df.fc_projected_values_floor.isnull(), 'fc_proj_fantasy_pts_fc'] * fc_ratio_floor
        
        if k == 'proj_ceiling':
            df['etr_floor_rank'] = df.groupby(['year', 'week'])['etr_proj_floor'].rank(ascending=False, method='min')
            df['etr_rank'] = df.groupby(['year', 'week'])['etr_proj_points'].rank(ascending=False, method='min')
            df['etr_ceiling_rank'] = df.groupby(['year', 'week'])['etr_proj_ceiling'].rank(ascending=False, method='min')

    if not is_dst:
        df['avg_proj_pass_cmp_pct'] = df.avg_proj_pass_comp / (df.avg_proj_pass_att + 1)

        df['avg_proj_rush_points'] = df.avg_proj_rush_yds * 0.1 + df.avg_proj_rush_td * 6
        df['avg_proj_rec_points'] = df.avg_proj_rec_yds * 0.1 + df.avg_proj_rec_td * 6 + df.avg_proj_rec
        df['avg_proj_pass_points'] = df.avg_proj_pass_yds * 0.04 + df.avg_proj_pass_td * 4 - df.avg_proj_pass_int * 1

        df['avg_proj_pass_ratio'] = df.avg_proj_pass_points / (df.avg_proj_pass_points + df.avg_proj_rush_points + 1)
        df['avg_proj_rush_ratio'] = df.avg_proj_rush_points / (df.avg_proj_rec_points + df.avg_proj_rush_points + 1)

        df['avg_proj_rush_yds_per_att'] = df.avg_proj_rush_yds / (df.avg_proj_rush_att + 1)
        df['avg_proj_rec_yds_per_tgt'] = df.avg_proj_rec_yds / (df.avg_proj_rec_tgts + 1)
        df['avg_proj_rec_yds_per_rec'] = df.avg_proj_rec_yds / (df.avg_proj_rec + 1)
        df['avg_proj_pass_yds_per_att'] = df.avg_proj_pass_yds / (df.avg_proj_pass_att + 1)

        df['avg_proj_rush_td_per_att'] = df.avg_proj_rush_td / (df.avg_proj_rush_att + 1)
        df['avg_proj_rec_td_per_tgt'] = df.avg_proj_rec_td / (df.avg_proj_rec_tgts + 1)
        df['avg_proj_pass_td_per_att'] = df.avg_proj_pass_td / (df.avg_proj_pass_att + 1)
    
    return df


def fill_ratio_nulls(df):
    ratio_fill_cols = ['ffa_sd_pts', 'ffa_dropoff', 'ffa_floor', 'ffa_ceiling', 'ffa_points_vor', 'ffa_floor_vor',
                        'ffa_ceiling_vor', 'ffa_rank', 'ffa_floor_rank', 'ffa_ceiling_rank', 'ffa_rec_sd',
                        'ffa_tier', 'ffa_uncertainty','ffa_pass_yds_sd', 'ffa_pass_tds_sd', 'ffa_pass_int_sd',
                        'ffa_rush_yds_sd',  'ffa_rush_tds_sd', 'fc_proj_rushing_stats_pct', 'fc_proj_rushing_stats_att_tar', 
                        'fc_proj_receiving_stats_pct', 'fc_projected_values_floor', 'fc_projected_values_ceiling',
                        'ffa_sd_pts', 'ffa_uncertainty', 'ffa_dst_int_sd', 'ffa_dst_sacks_sd',
                        'ffa_dst_safety_sd', 'ffa_dst_td_sd', 'fc_proj_defensive_stats_pts']
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
                 or 'fft' in c or 'expert' in c or 'points' in c or 'fdta' in c
                 or 'vegas' in c or 'etr' in c or 'fpts' in c]
    df = add_rolling_stats(df, ['player'], proj_cols)

    for c in proj_cols:
        df[f'trend_diff_{c}'] = df[f'rmean3_{c}'] - df[f'rmean8_{c}']
        df[f'trend_chg_{c}'] = df[f'trend_diff_{c}'] / (df[f'rmean8_{c}']+5)

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


def get_salaries(df, pos):
    sal = dm.read(f'''SELECT player, week, year, 
                             dk_salary, fd_salary, yahoo_salary
                    FROM Daily_Salaries
                    WHERE Position='{pos}'
                    ''', 'Pre_PlayerData')
    df = pd.merge(df, sal, on=['player', 'week', 'year'])
    df.loc[(df.fd_salary < 100) | (df.fd_salary > 10000), 'fd_salary'] = np.nan


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

    df = pd.merge(df, matchups, on=['player', 'week', 'year'], how='left')

    off_cols= ['offHeightInches', 'offWeight', 'offSpeed', 'offRoutes', 'offLeft',
           'offSlot', 'offRight', 'offFr', 'offCPct', 'offYprr', 'offGrade']

    def_cols = ['defHeightInches', 'defWeight', 'defSpeed', 'defRoutes', 'defLeft',
                'defSlot', 'defRight', 'defCPct', 'defYprr', 'defGrade',]

    df = df.sort_values(by=['player', 'year', 'week'])
    df[off_cols] = df.groupby('player', as_index=False)[off_cols].fillna(method='ffill').values

    df = df.sort_values(by=['team', 'year', 'week', 'defGrade'], ascending=[True, True, True, False])
    df[def_cols] = df.groupby('team', as_index=False)[def_cols].fillna(method='ffill').values
    df = df.sort_values(by=['player', 'year', 'week'])

    df.loc[df.adv.isnull(), 'adv'] = df.loc[df.adv.isnull(), 'offGrade'] -  df.loc[df.adv.isnull(), 'defGrade']
    df.loc[df.HeightInchesdiffer.isnull(), 'HeightInchesdiffer'] = df.loc[df.HeightInchesdiffer.isnull(), 'offHeightInches'] - \
                                                    df.loc[df.HeightInchesdiffer.isnull(), 'defHeightInches']
    df.loc[df.Weightdiffer.isnull(), 'Weightdiffer'] = df.loc[df.Weightdiffer.isnull(), 'offWeight'] - \
                                                    df.loc[df.Weightdiffer.isnull(), 'defWeight']
    df.loc[df.Speeddiffer.isnull(), 'Speeddiffer'] = df.loc[df.Speeddiffer.isnull(), 'offSpeed'] - \
                                                    df.loc[df.Speeddiffer.isnull(), 'defSpeed']

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
    matchups = forward_fill(matchups)

    for c, f in zip(['offGrade', 'defGrade'], [60, 60]):
        matchups[c] = matchups[c].fillna(f)

    matchups = matchups.fillna(0)

    for c in ['HeightInches', 'Weight']:
        matchups[f'{c}differ'] = matchups[f'off{c}'] - matchups[f'def{c}']

    df = pd.merge(df, matchups, on=['player', 'week', 'year'], how='left')

    off_cols= ['offHeightInches', 'offWeight', 'offRoutes', 'offWide',
    'offSlot', 'offInline', 'offFr', 'offCPct', 'offYprr', 'offGrade']

    def_cols = ['defHeightInches', 'defWeight', 'defRoutes', 'defCPct',
    'defYprr', 'defGrade']

    df = df.sort_values(by=['player', 'year', 'week'])
    df[off_cols] = df.groupby('player', as_index=False)[off_cols].fillna(method='ffill').values

    df = df.sort_values(by=['team', 'year', 'week', 'defGrade'], ascending=[True, True, True, False])
    df[def_cols] = df.groupby('team', as_index=False)[def_cols].fillna(method='ffill').values
    df = df.sort_values(by=['player', 'year', 'week'])

    df.loc[df.adv.isnull(), 'adv'] = df.loc[df.adv.isnull(), 'offGrade'] -  df.loc[df.adv.isnull(), 'defGrade']
    df.loc[df.HeightInchesdiffer.isnull(), 'HeightInchesdiffer'] = df.loc[df.HeightInchesdiffer.isnull(), 'offHeightInches'] - \
                                                    df.loc[df.HeightInchesdiffer.isnull(), 'defHeightInches']
    df.loc[df.Weightdiffer.isnull(), 'Weightdiffer'] = df.loc[df.Weightdiffer.isnull(), 'offWeight'] - \
                                                    df.loc[df.Weightdiffer.isnull(), 'defWeight']

    return df


#---------------
# Post Game Data
#---------------

def get_player_data(df, pos):

    player_data = dm.read(f'''SELECT * 
                              FROM {pos}_Stats 
                              WHERE week < 17
                                    AND season >= 2020
                              --WHERE (season = 2020 AND week != 17)
                              --       OR (season >=2021 AND week != 18)
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
                   'rush_ep_sum', 'rush_ydstogo_sum', 'rush_touchdown_sum',
                   'rush_yardline_100_sum', 'rush_first_down_sum', 'rec_yards_gained_sum', 
                   'rec_first_down_sum', 'rec_epa_sum', 'rec_qb_epa_sum', 'rec_cp_sum','rec_xyac_epa_sum', 
                   'rec_pass_attempt_sum', 'rec_xyac_success_sum', 'rec_comp_air_epa_sum',
                   'rec_air_yards_sum', 'rec_touchdown_sum', 'rec_xyac_median_yardage_sum',
                   'rec_complete_pass_sum', 'rec_qb_dropback_sum',
                   
                   'rush_red_zone_rush_attempt_sum', 'rush_red_zone_yards_gained_sum',
                   'rush_goalline_rush_attempt_sum', 'rush_goalline_ep_sum',
                   'rush_third_fourth_rush_attempt_sum', 'rush_third_fourth_yards_gained_sum',
                   'rec_red_zone_pass_attempt_sum', 'rec_red_zone_ep_sum', 'rec_red_zone_complete_pass_sum',
                   'rec_third_fourth_pass_attempt_sum', 'rec_third_fourth_complete_pass_sum', 'rec_third_fourth_yards_gained_sum',
                   'rec_goalline_pass_attempt_sum'
                   ]

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


def pos_rank_stats(df, team_pos_rank, pos):
    
    pos_stats = dm.read(f'''SELECT * 
                            FROM {pos}_Stats 
                            WHERE week < 17
                                  AND season >= 2020
                            --WHERE (season = 2020 AND week != 17)
                            --        OR (season >=2021 AND week != 18)
                            ''', 'FastR')
    pos_stats = pos_stats.rename(columns={'season': 'year'})

    pos_rank_cols = ['rush_rush_attempt_sum', 'rec_xyac_mean_yardage_sum', 'rush_yards_gained_sum',
                    'rush_ep_sum', 'rush_ydstogo_sum', 'rush_touchdown_sum',
                    'rush_yardline_100_sum', 'rush_first_down_sum', 'rec_yards_gained_sum', 
                    'rec_first_down_sum', 'rec_epa_sum', 'rec_qb_epa_sum', 'rec_cp_sum','rec_xyac_epa_sum', 
                    'rec_pass_attempt_sum', 'rec_xyac_success_sum', 'rec_comp_air_epa_sum',
                    'rec_air_yards_sum', 'rec_touchdown_sum', 'rec_xyac_median_yardage_sum',
                    'rec_complete_pass_sum', 'rec_qb_dropback_sum'
    ]
    agg_cols = {c: 'sum' for c in pos_rank_cols}

    pos_stats = pd.merge(team_pos_rank, pos_stats, on=['player', 'team', 'week', 'year'], how='left')

    pos_stats = pos_stats.groupby(['pos_rank', 'team', 'week','year']).agg(agg_cols)
    pos_stats.columns = ['pos_rank_' + c for c in pos_stats.columns]
    pos_stats = pos_stats.reset_index()

    gcols = ['team', 'pos_rank']
    rcols=['pos_rank_' + c for c in pos_rank_cols]
    pos_stats = pos_stats.sort_values(by=['team', 'pos_rank', 'year', 'week']).reset_index(drop=True)

    rolls3_mean = rolling_stats(pos_stats, gcols, rcols, 3, agg_type='mean')
    rolls3_max = rolling_stats(pos_stats, gcols, rcols, 3, agg_type='max')

    rolls8_mean = rolling_stats(pos_stats, gcols, rcols, 8, agg_type='mean')
    rolls8_max = rolling_stats(pos_stats, gcols, rcols, 8, agg_type='max')

    pos_stats = pd.concat([pos_stats, rolls8_mean, rolls8_max, rolls3_mean, rolls3_max], axis=1)

    pos_stats = pd.merge(team_pos_rank, pos_stats, on=['pos_rank', 'team', 'week', 'year'])
    pos_stats = pos_stats.drop(['pos_rank_' + c for c in pos_rank_cols], axis=1)

    # remove cincy-buf due to cancellation
    pos_stats = pos_stats[~((pos_stats.team.isin(['BUF', 'CIN'])) & (pos_stats.week==17) & (pos_stats.year==2022))].reset_index(drop=True)

    pos_stats['week'] = pos_stats['week'] + 1
    pos_stats = switch_seasons(pos_stats)
    pos_stats = fix_bye_week(pos_stats)

    df = pd.merge(df, pos_stats, on=['player', 'team', 'pos', 'week', 'year'], how='left')

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


def get_team_stats():
    
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

def format_lines(lines, is_home):
    if is_home==1: label = 'home'
    else: label = 'away'

    lines = lines[[f'{label}_team', f'{label}_line', 'over_under', 'week', 'year']]
    lines = lines.assign(is_home=is_home)
    lines.columns = ['team', 'line', 'over_under', 'week', 'year', 'is_home']
    
    lines = dc.convert_to_float(lines)
    lines['implied_points_for'] = (lines.over_under / 2) - (lines.line / 2) 
    lines['implied_points_against'] = (lines.over_under / 2) + (lines.line / 2) 
    
    return lines

def format_scores(scores, is_home):
    if is_home==1: label = 'home'
    else: label = 'away'

    scores = scores[[f'{label}_team', f'{label}_score', 'week', 'year']]
    scores = scores.assign(is_home=is_home)
    scores.columns = ['team', 'final_score', 'week', 'year', 'is_home']
    
    return scores

def join_lines_scores(lines, final_scores):

    home_lines = format_lines(lines, is_home=1)
    away_lines = format_lines(lines, is_home=0)
    all_lines = pd.concat([home_lines, away_lines], axis=0)

    home_scores = format_scores(final_scores, is_home=1)
    away_scores = format_scores(final_scores, is_home=0)
    scores = pd.concat([home_scores, away_scores], axis=0)
    
    scores = pd.merge(all_lines, scores, on=['team', 'week', 'year', 'is_home'])

    return scores

def create_scores_lines_table(WEEK, YEAR):

    lines = dm.read("SELECT * FROM Gambling_Lines WHERE year>=2020", 'Pre_TeamData')
    final_scores = dm.read("SELECT * FROM Final_Scores WHERE year>=2020", 'FastR')
    scores_lines = join_lines_scores(lines, final_scores)
    try:
        cur_lines = dm.read(f"SELECT * FROM Gambling_Lines WHERE year={YEAR} AND week={WEEK}", 'Pre_TeamData')
        cur_home = format_lines(cur_lines, is_home=1)
        cur_away = format_lines(cur_lines, is_home=0)
        cur_lines = pd.concat([cur_home, cur_away], axis=0)
        scores_lines = pd.concat([scores_lines, cur_lines], axis=0)
    except:
        print('Current week not available')

    dm.write_to_db(scores_lines, f'Model_Features_{YEAR}', 'Scores_Lines', 'replace')


def add_gambling_lines(df):

    lines = dm.read("SELECT * FROM Gambling_Lines", 'Pre_TeamData')
    home_lines = format_lines(lines, is_home=1)
    away_lines = format_lines(lines, is_home=0)
    lines = pd.concat([home_lines, away_lines], axis=0)

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

    df = pd.merge(df, weather, on=['team', 'week', 'year'], how='left')

    return df


def advanced_rb_stats(df):

    adv = dm.read('''SELECT * FROM PFR_Advanced_RB''', 'Post_PlayerData')
    adv = adv[adv.position.isin(['RB', 'rb', 'Rb'])].drop(['rank', 'position'], axis=1)
    adv = dc.convert_to_float(adv)
    adv.player = adv.player.apply(dc.name_clean)
    
    adv['yds_before_contact_pct'] = adv['yds_before_contact'] / (adv['rush_yds']+0.1)
    adv['yds_after_contact_pct'] = adv['yds_after_contact'] / (adv['rush_yds']+0.1)
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


def add_next_gen(df, stat_type):

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


def get_defense_stats():
    d_stats = dm.read(f'''SELECT * 
                        FROM Defense_Stats 
                        WHERE week < 17
                        --WHERE (season = 2020 AND week != 17)
                        --       OR (season >=2021 AND week != 18)
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



#-----------------------
# Attach Points
#-----------------------

def results_vs_predicted(df, col):

    df = df.sort_values(by=['player','year', 'week']).reset_index(drop=True)
    df[f'{col}_miss'] = df.y_act - df[col]
    df[f'{col}_miss'] = (df.groupby('player')[f'{col}_miss'].shift(1)).fillna(0)
    df = add_rolling_stats(df, ['player'], [f'{col}_miss'], perform_check=False)
    df[f'{col}_miss_recent_vs8'] = df[f'rmean3_{col}_miss'] - df[f'rmean8_{col}_miss']

    good_cols = [c for c in df.columns if 'miss' in c or c in ('player', 'team', 'week', 'year')]
    df = df[good_cols]

    return df

def projected_pts_vs_predicted(df):

    proj_pts_miss = df[['player', 'team', 'week', 'year']].copy()
    for c in ['ffa_points', 'projected_points', 'fantasyPoints', 'ProjPts', 
              'fc_proj_fantasy_pts_fc', 'fft_proj_pts', 'avg_proj_points', 'fdta_proj_points']:
        if c in df.columns:
            cur_miss = results_vs_predicted(df[['player', 'team', 'week', 'year', 'y_act', c]].copy(), c)
            proj_pts_miss = pd.merge(proj_pts_miss, cur_miss, on=['player', 'team', 'week', 'year']) 

    df = pd.merge(df, proj_pts_miss, on=['player', 'team', 'week', 'year'], how='left')
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
    snaps['avg_snap_pct'] = snaps.groupby('player').snap_pct.rolling(4).mean().values
    snaps = snaps.drop(['snap_counts', 'team_snap_count'], axis=1)

    return snaps


def attach_y_act(df, pos, defense=False, rush_or_pass=''):

    if defense:
        y_act = dm.read(f'''SELECT defTeam player, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020''', 'FastR')
    
        df = pd.merge(df, y_act, on=['player',  'week', 'year'], how='left')
    
    # elif pos=='QB':
    #     y_act = dm.read(f'''SELECT player, team, week, season year,
    #                                fantasy_pts{rush_or_pass} y_act
    #                         FROM {pos}_Stats
    #                         WHERE season >= 2020
    #                               AND pass_pass_attempt_sum > 15''', 'FastR')
        
    
    #     df = pd.merge(df, y_act, on=['player', 'team', 'week', 'year'], how='left')
    
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

#-------------------
# Final Cleanup
#-------------------

def drop_y_act_except_current(df, week, year):
    
    
    df = df[~(df.y_act.isnull()) | ((df.week==week) & (df.year==year))].reset_index(drop=True)
    df.loc[((df.week==week) & (df.year==year)), 'y_act'] = 0

    return df

def remove_non_uniques(df):
    cols = df.nunique()[df.nunique()==1].index
    cols = [c for c in cols if c != 'pos']
    df = df.drop(cols, axis=1)
    return df

def drop_duplicate_players(df):
    df = df.sort_values(by=['player', 'year', 'week', 'avg_proj_points', 'projected_points'],
                    ascending=[True, True, True, False, False])
    df = df.drop_duplicates(subset=['player', 'year', 'week'], keep='first').reset_index(drop=True)
    return df

def one_qb_per_week(df):
    max_qb = df.groupby(['team', 'year', 'week']).agg({'projected_points': 'max',
                                                       'fp_rank': 'min'}).reset_index()
    cols = df.columns
    df = pd.merge(df, max_qb.drop('fp_rank', axis=1), on=['team', 'year', 'week', 'projected_points'])
    df = pd.merge(df, max_qb.drop('projected_points', axis=1), on=['team', 'year', 'week', 'fp_rank'])
    df = df[cols]

    return df


def replace_inf_values(df):
    for c in df.columns:
        try:
            if len(df[df[c].isin([np.inf, -np.inf])])>0:
                print('Infinite Value:', c)
                df.loc[df[c].isin([np.inf, -np.inf]), c] = df.loc[~df[c].isin([np.inf, -np.inf]), c].mean()
                print('Remaining Errors:', len(df[df[c].isin([np.inf, -np.inf])])>0)
        except:
            pass
    return df

def remove_low_corrs(df, corr_cut=0.015):
    orig_columns = df.shape[1]
    obj_cols = df.dtypes[df.dtypes=='object'].index
    corrs = df[~((df.week==WEEK) & (df.year==YEAR))].copy().reset_index(drop=True)
    corrs = pd.DataFrame(np.corrcoef(corrs.drop(obj_cols, axis=1).values, rowvar=False), 
                         columns=[c for c in corrs.columns if c not in obj_cols],
                         index=[c for c in corrs.columns if c not in obj_cols])
    corrs = corrs['y_act']
    low_corrs = list(corrs[abs(corrs) < corr_cut].index)
    low_corrs = [c for c in low_corrs if c not in ('week', 'year', 'fd_salary')]
    df = df.drop(low_corrs, axis=1)
    print(f'Removed {len(low_corrs)}/{orig_columns} columns')
    
    corrs = corrs.dropna().sort_values()
    display(corrs.iloc[:20])
    display(corrs.iloc[-20:])
    display(corrs[~corrs.index.str.contains('ffa|proj|expert|rank|Proj|fc|salary|Points|Rank|fdta|fft|proj') | \
                   corrs.index.str.contains('team') | \
                   corrs.index.str.contains('qb')].iloc[:20])
    display(corrs[~corrs.index.str.contains('ffa|proj|expert|rank|Proj|fc|salary|Points|Rank|fdta|fft|proj') | \
                   corrs.index.str.contains('team') | \
                   corrs.index.str.contains('qb')].iloc[-20:])
    return df


#--------------------
# Data to apply to all datasets
#--------------------

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

def defense_for_pos():

    # defense stats that can be added to the offensive player data
    defense = fantasy_pros_new('DST').rename(columns={'player': 'defTeam'})
    defense = add_ffa_defense(defense)

    defense = defense.drop(['pos', 'position', 'team'], axis=1)

    pff_def = add_team_matchups()
    defense = pd.merge(defense, pff_def, on=['defTeam', 'year', 'week'])

    defense = fantasy_cruncher(defense.rename(columns={'defTeam': 'player'}), 'DST')
    defense = defense.rename(columns={'player': 'defTeam'})

    d_stats = get_defense_stats().rename(columns={'team': 'defTeam'})
    defense = pd.merge(defense, d_stats, on=['defTeam', 'week', 'year'], how='inner')
    
    defense = fantasy_data_proj_def(defense.rename(columns={'defTeam': 'player'}))
    defense = fantasy_points_projection_def(defense)
    defense = etr_projection_def(defense)
    defense = defense.rename(columns={'player': 'defTeam'})

    defense = consensus_fill(defense, is_dst=True)
    defense = fill_ratio_nulls(defense)
    defense = log_rank_cols(defense)
    # defense = rolling_proj_stats(defense)

    defense = defense.dropna(axis=1, thresh=defense.shape[0]-100)

    defense = defense.dropna()
    pff_def = pff_defense_rollup().rename(columns={'team': 'defTeam'})
    defense = pd.merge(defense, pff_def, on=['defTeam', 'week', 'year'])

    defense.columns = [f'def_{c}' if 'def' not in c else c for c in defense.columns]
    defense = defense.rename(columns={'def_week': 'week', 'def_year': 'year'})

    return defense


def get_max_qb():

    df = fantasy_pros_new('QB')
    df = pff_experts_new(df, 'QB')
    df = ffa_compile(df, 'FFA_Projections', 'QB')
    df = ffa_compile(df, 'FFA_RawStats', 'QB')
    df = fftoday_proj(df, 'QB')
    df = fantasy_cruncher(df, 'QB')
    df = fantasy_data_proj(df, 'QB')
    df = numberfire_proj(df)
    df = etr_projection(df, 'QB')
    df = fantasy_points_projection(df, 'QB')
    df = get_salaries(df, 'QB')

    df = consensus_fill(df)
    df = fill_ratio_nulls(df)
    player_vegas_stats = get_all_vegas_stats('QB')
    df = fill_vegas_stats(df, player_vegas_stats, 'QB')
    df = log_rank_cols(df)

    qb_cols = [
               'team', 'week', 'year', 
               'ffa_pass_yds', 'ffa_pass_tds','ffa_pass_int','ffa_rush_yds', 'ffa_rush_tds',
               'passComp', 'passAtt', 'passYds', 'passTd', 'passInt', 'passSacked', 'rushAtt', 'rushYds', 'rushTd',
               'fc_proj_passing_stats_att', 'fc_proj_passing_stats_yrds','fc_proj_passing_stats_tds', 'fc_proj_passing_stats_int',
               'fc_proj_rushing_stats_att',  'fc_proj_rushing_stats_yrds', 'fc_proj_rushing_stats_tds',
               'fft_proj_pts', 'fft_pass_att', 'fft_pass_int', 'fft_rush_yds', 'fft_rush_att', 'fft_pass_yds', 'fft_pass_td',
               'fft_rush_td', 'fft_pass_comp',  'fdta_pass_yds', 'fdta_pass_td', 'fdta_rush_yds','fdta_rush_td',
               'etr_proj_points', 'etr_proj_ceiling', 'etr_proj_floor', 'etr_pass_yds', 'etr_pass_tds',
               'etr_rush_yds', 'etr_rush_tds', 'etr_pass_att', 'etr_pass_cmp',
               'nf_passing_yds', 'nf_passing_tds', 'nf_rushing_att', 'nf_rushing_tds', 'nf_rushing_yds',
               'avg_proj_pass_yds', 'avg_proj_pass_td',  'avg_proj_pass_int', 'avg_proj_pass_att', 'avg_proj_pass_comp',
               'avg_proj_rush_yds',  'avg_proj_rush_att', 'avg_proj_rush_td', 
               'ffa_points', 'projected_points', 'avg_proj_points', 'ProjPts', 
               'log_avg_proj_rank', 'log_playeradj_fp_rank', 'log_expertConsensus', 'avg_proj_rank',
               'fp_rank', 'log_ffa_position_rank', 'log_rankadj_fp_rank', 'dk_salary',
               'avg_proj_pass_points', 'avg_proj_rush_points', 'avg_proj_pass_ratio',
               'player_tds_over_ev_trunc_norm', 'player_pass_yds_ev_trunc_norm', 'player_pass_tds_ev_trunc_norm',
               'player_rush_yds_ev_trunc_norm', 'player_rush_attempts_ev_trunc_norm', 
               'player_pass_attempts_ev_trunc_norm', 'player_pass_completions_ev_trunc_norm',
               'good_avg_proj_points', 'avg_vegas_proj_points', 'vegas_proj_points'
               
               ]
    
    df = drop_duplicate_players(df)
    df = one_qb_per_week(df)
    df = df[qb_cols]
    df.columns = ['qb_'+c if c not in ('team', 'week', 'year') else c for c in df.columns]

    return df


def create_pos_rank(df, extra_pos=False):
    df = df.sort_values(by=['team', 'pos', 'year', 'week', 'avg_proj_points'],
                        ascending=[True, True, True, True, False]).reset_index(drop=True)

    df['pos_rank'] = df.pos + df.groupby(['team', 'pos', 'year', 'week']).cumcount().apply(lambda x: str(x))
    if extra_pos:
        df = df[df['pos_rank'].isin(['RB0', 'RB1', 'RB2', 'WR0', 'WR1', 'WR2', 'WR3', 'WR4', 'TE0', 'TE1'])].reset_index(drop=True)
    else:
        df = df[df['pos_rank'].isin(['QB0', 'RB0', 'RB1', 'WR0', 'WR1', 'WR2', 'TE0'])].reset_index(drop=True)
    return df


def get_team_projections():

    team_proj = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE']:

        tp = fantasy_pros_new(pos)
        tp = pff_experts_new(tp, pos)
        tp = ffa_compile(tp, 'FFA_Projections', pos)
        tp = ffa_compile(tp, 'FFA_RawStats', pos)
        tp = fftoday_proj(tp, pos)
        tp = fantasy_cruncher(tp, pos)
        tp = get_salaries(tp, pos)
        tp = numberfire_proj(tp)
        tp = fantasy_data_proj(tp, pos)
        tp = etr_projection(tp, pos)
        tp = fantasy_points_projection(tp, pos)

        tp = consensus_fill(tp)
        tp = fill_ratio_nulls(tp)

        player_vegas_stats = get_all_vegas_stats(pos)
        tp = fill_vegas_stats(tp, player_vegas_stats, pos)

        tp = log_rank_cols(tp)

        tp_this_week = tp.copy()
        tp_this_week = tp_this_week[(tp_this_week.week==WEEK) & (tp_this_week.year==YEAR)]
        tp_other_weeks = tp.copy()
        tp_other_weeks = tp_other_weeks[~((tp_other_weeks.week==WEEK) & (tp_other_weeks.year==YEAR))]

        tp_this_week, _ = add_injuries(tp_this_week, pos); print(tp.shape[0])
        tp = pd.concat([tp_this_week, tp_other_weeks], axis=0).reset_index(drop=True)
        tp = drop_duplicate_players(tp)

        team_proj = pd.concat([team_proj, tp], axis=0)

    team_proj = create_pos_rank(team_proj)
    team_proj = forward_fill(team_proj)
    team_proj = team_proj.fillna(0)

    cnts = team_proj.groupby(['team', 'week', 'year']).agg({'avg_proj_points': 'count'})
    print('Team counts that do not equal 7:', cnts[cnts.avg_proj_points!=7])

    cols = [
            'projected_points', 'ProjPts', 'ffa_points', 'fc_proj_fantasy_pts_fc', 'avg_proj_points', 
            'fantasyPoints', 'dk_salary', 'fd_salary',
            'log_ffa_rank', 'log_avg_proj_rank', 'log_expertConsensus', 'log_rankadj_fp_rank', 'log_playeradj_fp_rank', 'log_fp_rank',
            'rushAtt', 'rushYds', 'rushTd', 'recvTargets', 'recvReceptions', 'recvYds', 'recvTd',
            'ffa_rush_yds', 'ffa_rush_tds','ffa_rec', 'ffa_rec_yds','ffa_rec_tds',
            'fft_rush_att', 'fft_rush_yds', 'fft_rush_td', 'fft_rec', 'fft_rec_yds', 'fft_rec_td', 'fft_proj_pts',
            'fc_proj_rushing_stats_att', 'fc_proj_rushing_stats_yrds', 'fc_proj_rushing_stats_tds',
            'fc_proj_receiving_stats_tar', 'fc_proj_receiving_stats_rec',
            'fc_proj_receiving_stats_yrds', 'fc_proj_receiving_stats_tds', 
            'fdta_rush_yds', 'fdta_rush_td', 'fdta_rec', 'fdta_rec_yds', 'fdta_rec_td', 'fdta_proj_points',
            'etr_proj_points', 'etr_proj_ceiling', 'etr_proj_floor', 'etr_rush_yds', 'etr_rush_tds', 'etr_rec', 'etr_rec_yds', 'etr_rec_td',
            'nf_rushing_yds', 'nf_rushing_tds', 'nf_receiving_rec', 'nf_receiving_yds', 'nf_receiving_tds',
            'fpts_rush_yds', 'fpts_rec', 'fpts_rush_td', 'fpts_rec_td', 'fpts_rush_yds', 'fpts_rec_yds',
            'avg_proj_rush_att', 'avg_proj_rush_td', 'avg_proj_rec', 'avg_proj_rec_tgts','avg_proj_rec_yds','avg_proj_rec_td', 
            'avg_vegas_proj_points', 'vegas_proj_points',
            'player_rush_yds_ev_trunc_norm', 'player_rush_attempts_ev_trunc_norm', 'player_receptions_ev_trunc_norm',
            'player_reception_yds_ev_trunc_norm', 
            ]

    to_agg = {c: 'sum' for c in cols}

    # get the projections broken out by RB and WR/TE
    team_proj_pos = team_proj[team_proj.pos.isin(['RB', 'WR', 'TE'])].copy()
    team_proj_pos.loc[team_proj_pos.pos=='TE', 'pos'] = 'WR'
    team_proj_pos = team_proj_pos.groupby(['pos', 'team', 'week', 'year']).agg(to_agg)
    team_proj_pos.columns = [f'pos_proj_{c}' for c in team_proj_pos.columns]
    team_proj_pos = team_proj_pos.reset_index()
    team_proj_pos_te = team_proj_pos[team_proj_pos.pos=='WR'].copy()
    team_proj_pos_te['pos'] = 'TE'
    team_proj_pos = pd.concat([team_proj_pos, team_proj_pos_te], axis=0).reset_index(drop=True)
    
    # get the projections broken out by team
    team_proj = team_proj.groupby(['team', 'week', 'year']).agg(to_agg)
    team_proj.columns = [f'team_proj_{c}' for c in team_proj.columns]
    team_proj = team_proj.reset_index()

    return team_proj, team_proj_pos

def proj_market_share(df, proj_col_name):

    proj_cols = [c for c in df.columns if proj_col_name in c]

    for proj_col in proj_cols:
        orig_col = proj_col.replace(proj_col_name, '')
        if orig_col in df.columns:
            df[f'{proj_col_name}share_{orig_col}'] = df[orig_col] / (df[proj_col]+3)
            df[f'{proj_col_name}share_diff_{orig_col}'] = df[orig_col] - df[proj_col]
            df[[f'{proj_col_name}share_{orig_col}', f'{proj_col_name}share_diff_{orig_col}']] = \
                df[[f'{proj_col_name}share_{orig_col}', f'{proj_col_name}share_diff_{orig_col}']].fillna(0)
    return df

def non_qb_team_pos_rank():
    team_pos_rank = pd.DataFrame()
    for pos in ['RB', 'WR', 'TE']:

        tp = fantasy_pros_new(pos)
        tp = pff_experts_new(tp, pos)
        tp = ffa_compile(tp, 'FFA_Projections', pos)
        tp = ffa_compile(tp, 'FFA_RawStats', pos)
        tp = fftoday_proj(tp, pos)
        tp = fantasy_cruncher(tp, pos)
        tp = fantasy_data_proj(tp, pos)
        tp = etr_projection(tp, pos)
        tp = numberfire_proj(tp)
        tp = fantasy_points_projection(tp, pos)
        tp = consensus_fill(tp)
        tp = fill_ratio_nulls(tp)
        team_pos_rank = pd.concat([team_pos_rank, tp], axis=0)
        tp, _ = add_injuries(tp, pos); print(tp.shape[0])
        tp = drop_duplicate_players(tp)


    team_pos_rank = create_pos_rank(team_pos_rank, extra_pos=True)
    return  team_pos_rank[['player', 'pos', 'pos_rank', 'team', 'week', 'year']]


def get_stat_constraints(past_data, pos, stat_cols):

    if pos in ('RB', 'WR', 'TE'): filter_str = '(rec_pass_attempt_sum + rush_rush_attempt_sum) > 5'
    elif pos == 'QB': filter_str = '(pass_pass_attempt_sum + rush_rush_attempt_sum) > 15'
    else: filter_str = 'week > 0'

    player_data = dm.read(f'''SELECT * 
                              FROM {pos}_Stats 
                              WHERE week < 17
                                        AND season >= 2020
                                        AND {filter_str}
                                    ''', 'FastR').rename(columns={'season': 'year'})
    
    if pos == 'QB': player_data['rec_pass_touchdown_sum'] = 0
    if pos == 'Defense': player_data = player_data.rename(columns={'defteam': 'player'})
    
    player_data = pd.merge(past_data, player_data, on=['player', 'week', 'year'])
    stat_mean = player_data[stat_cols].sum(axis=1).mean()
    stat_std = player_data[stat_cols].sum(axis=1).std()
    stat_min = player_data[stat_cols].sum(axis=1).min()
    stat_max = player_data[stat_cols].sum(axis=1).max()

    stat_cv = stat_std / stat_mean

    return stat_mean, stat_cv, stat_min, stat_max


def trunc_normal(mean_val, sdev, min_sc, max_sc, num_samples=500):

    import scipy.stats as stats

    # create truncated distribution
    lower_bound = (min_sc - mean_val) / sdev, 
    upper_bound = (max_sc - mean_val) / sdev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc=mean_val, scale=sdev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


class PlayerPropsCalculator:
    """
    Vectorized calculator for player props expected values using 
    Poisson and Truncated Normal distributions.
    """
    
    def __init__(self, 
                 max_iterations: int = 20, 
                 max_lambda: float = 50.0):
        """
        Initialize calculator with configuration parameters.
        
        Args:
            max_iterations: Maximum iterations for binary search
            max_lambda: Maximum lambda value for Poisson

        """
        self.max_iterations = max_iterations
        self.max_lambda = max_lambda
        
   
    
    def calculate_discrete_probabilities_vectorized(self, lambda_params: np.ndarray, 
                                                  thresholds: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized calculation of discrete probabilities for Poisson.
        """
        floor_thresholds = np.floor(thresholds).astype(int)
        ceil_thresholds = np.ceil(thresholds).astype(int)
        
        floor_probs = poisson.cdf(floor_thresholds, lambda_params)
        ceil_probs = poisson.cdf(ceil_thresholds, lambda_params)
        weights = ceil_thresholds - thresholds
        
        return {
            'floor_probs': floor_probs,
            'ceil_probs': ceil_probs,
            'weights': weights
        }
    
    def find_poisson_parameters_vectorized(self, points: np.ndarray, 
                                         target_probs: np.ndarray) -> np.ndarray:
        """
        Vectorized binary search to find lambda parameters.
        """
        low = np.zeros_like(points)
        high = np.full_like(points, self.max_lambda)
        
        for _ in range(self.max_iterations):
            mid = (low + high) / 2
            probs = self.calculate_discrete_probabilities_vectorized(mid, points)
            interpolated_probs = (
                probs['floor_probs'] * probs['weights'] + 
                probs['ceil_probs'] * (1 - probs['weights'])
            )
            over_probs = 1 - interpolated_probs
            high = np.where(over_probs > target_probs, mid, high)
            low = np.where(over_probs <= target_probs, mid, low)
        
        return (low + high) / 2

    def calculate_truncnorm_ev_vectorized(self, means: np.ndarray, 
                                        stds: np.ndarray) -> np.ndarray:
        """
        Calculate expected values for truncated normal with configured bounds.
        """
        min_value = self.truncnorm_config['min_value']
        max_value = self.truncnorm_config['max_value']
        
        # Calculate normalized bounds
        a = (min_value - means) / stds
        b = (max_value - means) / stds
        
        # Calculate expected value using truncated normal formula
        expected_values = truncnorm.mean(a, b, loc=means, scale=stds)
        
        return expected_values
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire DataFrame of player props using vectorized operations.
        """
        required_columns = {'point', 'price', 'prop_type', 'week', 'year'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = df.copy()
        result_df['point'] = pd.to_numeric(result_df['point'], errors='coerce')
        result_df['price'] = pd.to_numeric(result_df['price'], errors='coerce')
        
        valid_mask = (
            ~result_df['point'].isna() & 
            ~result_df['price'].isna() & 
            (result_df['price'] > 1) & 
            (result_df['point'] > 0)
        )
        
        # Initialize columns
        result_df['ev_poisson'] = np.nan
        
        if valid_mask.any():
            points = result_df.loc[valid_mask, 'point'].values
            prices = result_df.loc[valid_mask, 'price'].values
            implied_probs = 1 / prices
            
            # Poisson calculations
            lambda_params = self.find_poisson_parameters_vectorized(points, implied_probs)
            poisson_evs = lambda_params
            
            # Assign results
            result_df.loc[valid_mask, 'ev_poisson'] = poisson_evs
        
        return result_df
    
def cleanup_vegas_names(df):
    df['is_defense'] = 0
    df.loc[df.description.str.contains('Defense|D/ST'), 'is_defense'] = 1
    df_def = df[df.is_defense==1].copy()
    df = df[df.is_defense==0].copy()

    df.description = df.description.apply(dc.name_clean)
    df_def.description = df_def.description.apply(dc.name_clean).apply(lambda x: x.split('Defense')[0].split('D/St')[0]).str.strip()
    print('Missing D mappings:', [c for c in df_def.description.unique() if c not in team_d_map.keys()])
    df_def.description = df_def.description.map(team_d_map)

    df = pd.concat([df, df_def], axis=0)
    df = df.drop(['is_defense'], axis=1)
    df = df.rename(columns={'description': 'player'})

    return df

def pull_vegas_stats(prop_types, pos):
    
    df = dm.read(f'''SELECT *
                     FROM Game_Odds
                     WHERE prop_type IN {prop_types}
                           AND name IN ('Over', 'Yes')
                ''', 'Pre_PlayerData')
    
    df.loc[df.prop_type == 'player_anytime_td', 'point'] = 0.5
    df = cleanup_vegas_names(df)

    if pos == 'Defense': 
        pos = 'DST'
        player_col = 'team'
    else: 
        player_col = 'player'
    fp = dm.read(f'''
                    SELECT {player_col} player, week, year
                    FROM FantasyPros
                    WHERE pos='{pos}'
                 ''', 'Pre_PlayerData')
    
    df = pd.merge(df, fp, on=['player', 'week', 'year'], how='inner')

    return df

def get_all_vegas_stats(pos):

    pos_stats = {
    
    'QB': [
            [('player_pass_tds', 0), ['pass_pass_touchdown_sum']],
            [('player_pass_yds', 0), ['pass_yards_gained_sum']],
            [('player_pass_attempts', 0), ['pass_pass_attempt_sum']],
            [('player_pass_completions', 0), ['pass_complete_pass_sum']],
            [('player_pass_interceptions', 0), ['pass_interception_sum']],

            [('player_anytime_td', 'player_tds_over'), ['rush_rush_touchdown_sum', 'rec_pass_touchdown_sum']],
            [('player_rush_yds', 0),['rush_yards_gained_sum']],
            [('player_rush_attempts', 0), ['rush_rush_attempt_sum']]
    ],
    'RB': [
            [('player_anytime_td', 'player_tds_over'), ['rush_rush_touchdown_sum', 'rec_pass_touchdown_sum']],
            [('player_rush_yds', 0), ['rush_yards_gained_sum']],
            [('player_reception_yds', 0), ['rec_yards_gained_sum']],
            [('player_rush_attempts', 0), ['rush_rush_attempt_sum']],
            [('player_receptions', 0), ['rec_complete_pass_sum']]
        ],
    'WR': [
            [('player_anytime_td', 'player_tds_over'), ['rush_rush_touchdown_sum', 'rec_pass_touchdown_sum']],
            [('player_reception_yds', 0), ['rec_yards_gained_sum']],
            [('player_rush_attempts', 0), ['rush_rush_attempt_sum']],
            [('player_receptions', 0), ['rec_complete_pass_sum']],
            [('player_rush_yds', 0), ['rush_yards_gained_sum']]
        ],
    'TE': [
            [('player_anytime_td', 'player_tds_over'), ['rush_rush_touchdown_sum', 'rec_pass_touchdown_sum']],
            [('player_reception_yds', 0), ['rec_yards_gained_sum']],
            [('player_receptions', 0), ['rec_complete_pass_sum']],
            [('player_rush_yds', 0), ['rush_yards_gained_sum']]
        ],

    'Defense': [
        [('player_anytime_td', 'player_tds_over'), ['return_touchdown', 'def_td']]
        ]
    }

    if pos == 'Defense':
        past_data = dm.read(f'''SELECT player, week, year
                                FROM Defense_Data_Week{WEEK-1}
                        ''', f'Model_Features_{YEAR}')
    else:
        past_data = dm.read(f'''SELECT player, week, year
                                FROM Backfill_{pos}_Week{WEEK-1}
                            ''', f'Model_Features_{YEAR}')
    i = 0
    for prop_types, stat_cols in pos_stats[pos]:
        odds = pull_vegas_stats(prop_types, pos)
        _, stat_cv, stat_min, stat_max = get_stat_constraints(past_data, pos, stat_cols)
        odds['ev_trunc_norm'] = np.nan
        for point in odds.point.unique():
            td_dist = trunc_normal(point, point*stat_cv, stat_min, stat_max, num_samples=10000)
            odds.loc[odds.point==point, 'ev_trunc_norm'] = np.percentile(td_dist, 100/odds.loc[odds.point==point, 'price'])

        calculator = PlayerPropsCalculator(max_lambda=odds.point.max())
        odds = calculator.process_dataframe(odds)
        odds = odds.pivot_table(index=['player', 'year', 'week'], columns='prop_type', values=['ev_trunc_norm', 'ev_poisson']).reset_index()
        odds.columns = [f"{c[1]}_{c[0]}" if c[1]!='' else c[0] for c in odds.columns]
        if i == 0: player_vegas_stats = odds.copy()
        else: player_vegas_stats = pd.merge(player_vegas_stats, odds, on=['player', 'week', 'year'], how='outer')
        i += 1

    return player_vegas_stats


def fill_vegas_stats(df, player_vegas_stats, pos):

    if pos=='Defense': 
        player_vegas_stats = player_vegas_stats.rename(columns={'player': 'defTeam'})
        df = pd.merge(df, player_vegas_stats, on=['defTeam', 'week', 'year'], how='left')
    else:
        df = pd.merge(df, player_vegas_stats, on=['player', 'week', 'year'], how='left')

    if pos == 'QB':
        fill_cols = {
            'player_anytime_td_ev_poisson': ['avg_proj_rush_td'],
            'player_anytime_td_ev_trunc_norm': ['avg_proj_rush_td'],
            'player_tds_over_ev_poisson': ['avg_proj_rush_td'],
            'player_tds_over_ev_trunc_norm': ['avg_proj_rush_td'],
            
            'player_pass_tds_ev_poisson': ['avg_proj_pass_td'],
            'player_pass_tds_ev_trunc_norm': ['avg_proj_pass_td'],
            'player_pass_yds_ev_poisson': ['avg_proj_pass_yds'],
            'player_pass_yds_ev_trunc_norm': ['avg_proj_pass_yds'],
            'player_pass_attempts_ev_poisson': ['avg_proj_pass_att'],
            'player_pass_attempts_ev_trunc_norm': ['avg_proj_pass_att'],
            'player_pass_completions_ev_poisson': ['avg_proj_pass_comp'],
            'player_pass_completions_ev_trunc_norm': ['avg_proj_pass_comp'],
            'player_pass_interceptions_ev_poisson': ['avg_proj_pass_int'],
            'player_pass_interceptions_ev_trunc_norm': ['avg_proj_pass_int'],
            
            'player_rush_yds_ev_poisson': ['avg_proj_rush_yds'],
            'player_rush_yds_ev_trunc_norm': ['avg_proj_rush_yds'],
            'player_rush_attempts_ev_poisson': ['avg_proj_rush_att'],
            'player_rush_attempts_ev_trunc_norm': ['avg_proj_rush_att'],
        }

    elif pos == 'Defense':
        fill_cols = {
            'player_anytime_td_ev_poisson': ['avg_proj_dst_td'],
            'player_anytime_td_ev_trunc_norm': ['avg_proj_dst_td'],
            'player_tds_over_ev_poisson': ['avg_proj_dst_td'],
            'player_tds_over_ev_trunc_norm': ['avg_proj_dst_td'],
        }

    else:
        fill_cols = {
            'player_anytime_td_ev_poisson': ['avg_proj_rush_td', 'avg_proj_rec_td'],
            'player_anytime_td_ev_trunc_norm': ['avg_proj_rush_td', 'avg_proj_rec_td'],
            'player_tds_over_ev_poisson': ['avg_proj_rush_td', 'avg_proj_rec_td'],
            'player_tds_over_ev_trunc_norm': ['avg_proj_rush_td', 'avg_proj_rec_td'],
            
            'player_rush_yds_ev_poisson': ['avg_proj_rush_yds'],
            'player_rush_yds_ev_trunc_norm': ['avg_proj_rush_yds'],
            'player_rush_attempts_ev_poisson': ['avg_proj_rush_att'],
            'player_rush_attempts_ev_trunc_norm': ['avg_proj_rush_att'],
            
            'player_reception_yds_ev_poisson': ['avg_proj_rec_yds'],
            'player_reception_yds_ev_trunc_norm': ['avg_proj_rec_yds'],
            'player_receptions_ev_poisson': ['avg_proj_rec'],
            'player_receptions_ev_trunc_norm': ['avg_proj_rec'],
        }

    cols = [c for c in player_vegas_stats.columns if 'ev' in c]
    for c in cols:
        ratio = (df.loc[~df[c].isnull(), c] / (df.loc[~df[c].isnull(), fill_cols[c]].sum(axis=1)+0.1)).median()
        df.loc[df[c].isnull(), c] = ratio * df.loc[df[c].isnull(), fill_cols[c]].sum(axis=1)

    if pos == 'QB':
        df['vegas_proj_points'] = (
            df[['player_anytime_td_ev_trunc_norm', 'player_tds_over_ev_trunc_norm',
                #'player_anytime_td_ev_poisson', 'player_tds_over_ev_poisson'
                ]].mean(axis=1) * 6 +
            df[['player_pass_tds_ev_trunc_norm', #'player_pass_tds_ev_poisson',
                ]].mean(axis=1) * 4 +
            df[['player_pass_yds_ev_poisson', 'player_pass_yds_ev_trunc_norm']].mean(axis=1) * 0.04 +
            df[['player_rush_yds_ev_poisson', 'player_rush_yds_ev_trunc_norm']].mean(axis=1) * 0.1 -
            df[['player_pass_interceptions_ev_poisson', 'player_pass_interceptions_ev_trunc_norm']].mean(axis=1) * 1
        )
    if pos in ('RB', 'WR', 'TE'):
        df['vegas_proj_points'] = (
            df[['player_anytime_td_ev_trunc_norm','player_tds_over_ev_trunc_norm',
                #'player_tds_over_ev_poisson', #'player_anytime_td_ev_poisson'
                ]].mean(axis=1) * 6 + 
            df[['player_rush_yds_ev_poisson', 'player_rush_yds_ev_trunc_norm']].mean(axis=1) * 0.1 + 
            df[['player_reception_yds_ev_poisson', 'player_reception_yds_ev_trunc_norm']].mean(axis=1) * 0.1 + 
            df[['player_receptions_ev_poisson', 'player_receptions_ev_trunc_norm']].mean(axis=1) * 1
        )

    if pos != 'Defense':
        df['avg_vegas_proj_points'] = 0.8*df.avg_proj_points + 0.2*df.vegas_proj_points
        df['good_avg_proj_points'] = df[['etr_proj_points', 'fpts_proj_points', 'vegas_proj_points', 
                                        'fdta_proj_points', 'projected_points', 'nf_proj_points']].mean(axis=1)
    
    return df


#%%

# create the scores and lines table
create_scores_lines_table(WEEK, YEAR)

# get datasets that will be used across positions
opp_defense = defense_for_pos()
team_proj, team_proj_pos = get_team_projections() # add injury removal here in the future
team_stats = get_team_stats()
team_qb = get_max_qb()
team_pos_rank = non_qb_team_pos_rank()
pff_oline = pff_oline_rollup()


#%%
pos = 'QB'
rush_or_pass = ''

def qb_pull(rush_or_pass):

    #-------------------
    # pre-game data
    #-------------------

    # pull all projections and ranks
    df = fantasy_pros_new(pos); print(df.shape[0])
    df = pff_experts_new(df, pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fftoday_proj(df, pos); print(df.shape[0])
    df = fantasy_cruncher(df, pos); print(df.shape[0])
    df = fantasy_data_proj(df, pos); print(df.shape[0])
    df = numberfire_proj(df); print(df.shape[0])
    df = etr_projection(df, pos); print(df.shape[0])
    df = fantasy_points_projection(df, pos); print(df.shape[0])
    df = get_salaries(df, pos); print(df.shape[0])

    # clean up any missing values and engineer data
    df = consensus_fill(df); print(df.shape[0])
    df = fill_ratio_nulls(df); print(df.shape[0])
    
    player_vegas_stats = get_all_vegas_stats(pos)
    df = fill_vegas_stats(df, player_vegas_stats, pos)
    
    df = log_rank_cols(df); print(df.shape[0])
    df = rolling_proj_stats(df); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])

    df = add_fp_rolling(df, pos); print(df.shape[0])
    df = add_pfr_matchup(df); print(df.shape[0])
    df = add_gambling_lines(df); print(df.shape[0])
    df = add_weather(df); print(df.shape[0])

    #--------------------
    # Post-game data
    #--------------------

    # add player stats data
    df = get_player_data(df, pos); print(df.shape[0])

    df = add_rz_stats_qb(df); print(df.shape[0])
    df = add_qbr(df); print(df.shape[0])
    df = add_qb_adv(df); print(df.shape[0])
    df = add_next_gen(df, 'Passing'); print(df.shape[0])

    # merge self team and opposing team stats
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, team_proj, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, opp_defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = def_pts_allowed(df); print(df.shape[0])
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])

    df = attach_y_act(df, pos, rush_or_pass=rush_or_pass)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df = df.dropna(axis=1, thresh=df.shape[0]-100).dropna().reset_index(drop=True); print(df.shape[0])

    df = one_qb_per_week(df); print(df.shape[0])

    df = remove_non_uniques(df)
    df = df[(df.ProjPts > 10) & (df.projected_points > 10)].reset_index(drop=True)
    df = drop_duplicate_players(df)
    df = replace_inf_values(df)
    df = remove_low_corrs(df, corr_cut=0.03)

    print('Total Rows:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())

    dm.write_to_db(df.iloc[:,:2000], f'Model_Features_{YEAR}', f"QB_Data{rush_or_pass.replace('_', '')}_Week{WEEK}", if_exist='replace', create_backup=True)
    if df.shape[1] > 2000:
        dm.write_to_db(df.iloc[:,2000:], f'Model_Features_{YEAR}', f"QB_Data{rush_or_pass.replace('_', '')}_Week{WEEK}_v2", if_exist='replace')

    return df

qb_both = qb_pull('')
# # qb_rush = qb_pull('_rush')
# # qb_pass = qb_pull('_pass')

#%%
for pos in ['RB', 'WR', 'TE']:

    #----------------
    # Pre game data
    #----------------

    # pull all projections and ranks
    df = fantasy_pros_new(pos); print(df.shape[0])
    df = pff_experts_new(df, pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fftoday_proj(df, pos); print(df.shape[0])
    df = fantasy_cruncher(df, pos); print(df.shape[0])
    df = fantasy_data_proj(df, pos); print(df.shape[0])
    df = numberfire_proj(df); print(df.shape[0])
    df = etr_projection(df, pos); print(df.shape[0])
    df = fantasy_points_projection(df, pos); print(df.shape[0])
    df = get_salaries(df, pos); print(df.shape[0])

    # clean up any missing values and engineer data
    df = consensus_fill(df); print(df.shape[0])
    df = fill_ratio_nulls(df); print(df.shape[0])

    player_vegas_stats = get_all_vegas_stats(pos)
    df = fill_vegas_stats(df, player_vegas_stats, pos)

    df = log_rank_cols(df); print(df.shape[0])
    df = rolling_proj_stats(df); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])

    # add additional matchup and other pre-game data
    df = add_fp_rolling(df, pos); print(df.shape[0])
    df = add_pfr_matchup(df); print(df.shape[0])
    df = add_gambling_lines(df); print(df.shape[0])
    df = add_weather(df); print(df.shape[0])
    if pos == 'WR': df = cb_matchups(df); print(df.shape[0])
    if pos == 'TE': df = te_matchups(df); print(df.shape[0])

    #-----------------------
    # Post-Game Data
    #----------------------

    # add player stats
    df = get_player_data(df, pos); print(df.shape[0])

    # add advanced stats
    df = add_rz_stats(df); print(df.shape[0])
    df = advanced_rec_stats(df)
    if pos in ('WR', 'TE'):
        df = add_next_gen(df, 'Receiving'); print('next_gen', df.shape[0])
    if pos == 'RB': 
        df = advanced_rb_stats(df)
        df = add_next_gen(df, 'Rushing'); print(df.shape[0])

    df = def_pts_allowed(df); print(df.shape[0])

    # merge self team and opposing team stats
    df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, team_proj, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, team_proj_pos, on=['pos', 'team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, opp_defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])

    # projection market share
    df = proj_market_share(df, 'team_proj_'); print(df.shape[0])
    df = proj_market_share(df, 'pos_proj_'); print(df.shape[0])
    
    # calc actual market share and trailing stats for pos ranks in a team
    df = calc_market_share(df); print(df.shape[0])
    df = pos_rank_stats(df, team_pos_rank, pos); print(df.shape[0])

    df = attach_y_act(df, pos)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df =  df.dropna(axis=1, thresh=df.shape[0]-100).dropna().reset_index(drop=True); print(df.shape[0])
    df = df.fillna({'fdta_pass_int': 0})
    df.loc[df.fd_salary.isnull(), 'fd_salary'] = df.loc[df.fd_salary.isnull(), 'dk_salary']
    df = remove_non_uniques(df)
    df = drop_duplicate_players(df)
    df = replace_inf_values(df)
    df = remove_low_corrs(df, corr_cut=0.03)

    print('Total Rows:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())

    dm.write_to_db(df.iloc[:, :2000], f'Model_Features_{YEAR}', f'{pos}_Data_Week{WEEK}', if_exist='replace')
    if df.shape[1] > 2000:
        dm.write_to_db(df.iloc[:, 2000:], f'Model_Features_{YEAR}', f'{pos}_Data_Week{WEEK}_v2', if_exist='replace')


#%%

output = pd.DataFrame()
for pos in ['QB', 'RB', 'WR', 'TE']:

    df = fantasy_pros_new(pos); print(df.shape[0])
    df = pff_experts_new(df, pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_Projections', pos); print(df.shape[0])
    df = ffa_compile(df, 'FFA_RawStats', pos); print(df.shape[0])
    df = fftoday_proj(df, pos); print(df.shape[0])
    df = fantasy_cruncher(df, pos); print(df.shape[0])
    df = fantasy_data_proj(df, pos); print(df.shape[0])
    df = numberfire_proj(df); print(df.shape[0])
    df = etr_projection(df, pos); print(df.shape[0])
    df = fantasy_points_projection(df, pos); print(df.shape[0])

    df = consensus_fill(df); print(df.shape[0])
    df = fill_ratio_nulls(df); print(df.shape[0])

    player_vegas_stats = get_all_vegas_stats(pos)
    df = fill_vegas_stats(df, player_vegas_stats, pos)

    df = log_rank_cols(df); print(df.shape[0])
    df, _ = add_injuries(df, pos); print(df.shape[0])

    df = get_salaries(df, pos); print(df.shape[0])
    df.loc[(df.fd_salary < 100) | (df.fd_salary > 10000), 'fd_salary'] = np.nan

    df = add_weather(df); print(df.shape[0])
    df = add_gambling_lines(df); print(df.shape[0])
    
    # merge self team and opposing team stats
    if pos!= 'QB': df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])
    df = pd.merge(df, team_stats, on=['team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, team_proj, on=['team', 'week', 'year']); print( df.shape[0])
    
    if pos != 'QB': df = pd.merge(df, team_proj_pos, on=['pos', 'team', 'week', 'year']); print( df.shape[0])
    df = pd.merge(df, opp_defense, on=['defTeam', 'week', 'year']); print(df.shape[0])
    df = pd.merge(df, pff_oline, on=['team', 'week', 'year']); print(df.shape[0])

    # projection market share
    df = proj_market_share(df, 'team_proj_')
    if pos != 'QB': df = proj_market_share(df, 'pos_proj_')

    df = attach_y_act(df, pos)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])
    df = projected_pts_vs_predicted(df); print(df.shape[0])
    df = def_pts_allowed(df); print(df.shape[0])
    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df =  df.dropna(axis=1, thresh=df.shape[0]-100)
    df = df.fillna({'fdta_pass_int': 0})
    df.loc[df.fd_salary.isnull(), 'fd_salary'] = df.loc[df.fd_salary.isnull(), 'dk_salary']
    
    df = df.dropna().reset_index(drop=True); print(df.shape[0])

    df['pos'] = pos
    if pos=='QB': df = one_qb_per_week(df); print(df.shape[0])
    
    df = drop_duplicate_players(df)
    df = replace_inf_values(df)
    df = remove_low_corrs(df, corr_cut=0.03)
    df = remove_non_uniques(df)

    print('Data Size:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())

    dm.write_to_db(df, f'Model_Features_{YEAR}', f'Backfill_{pos}_Week{WEEK}', 'replace')



#%%

def create_self_cols(df):
    self_df = df.copy()
    self_cols = ['team', 'week','year']
    self_cols.extend(['self_'+c for c in self_df.columns if c not in ('team', 'week', 'year')])
    self_df.columns = self_cols
    return self_df

# defense stats that can be added to the offensive player data
defense = fantasy_pros_new('DST')
defense = add_fp_rolling(defense, 'Defense'); print(defense.shape[0])

defense = defense.rename(columns={'player': 'defTeam'})
defense = add_ffa_defense(defense)
defense = defense.drop(['pos', 'position', 'team'], axis=1)

pff_def = add_team_matchups()
defense = pd.merge(defense, pff_def, on=['defTeam', 'year', 'week'])

defense = fantasy_cruncher(defense.rename(columns={'defTeam': 'player'}), 'DST')
defense = defense.rename(columns={'player': 'defTeam'})

defense = fantasy_data_proj_def(defense.rename(columns={'defTeam': 'player'}))
defense = fantasy_points_projection_def(defense)
defense = etr_projection_def(defense)
defense = defense.rename(columns={'player': 'defTeam'})



defense = consensus_fill(defense, is_dst=True)
defense = fill_ratio_nulls(defense)

player_vegas_stats = get_all_vegas_stats('Defense')
defense = fill_vegas_stats(defense, player_vegas_stats, 'Defense')

defense = log_rank_cols(defense)
defense = rolling_proj_stats(defense.rename(columns={'defTeam':'player'}))
defense = defense.rename(columns={'player':'team'})

defense = defense.dropna(axis=1, thresh=defense.shape[0]-100)

all_cols = [c for c in defense.columns if c != 'y_act']
defense = defense.dropna(subset=all_cols)

defense = add_gambling_lines(defense); print(defense.shape[0])
defense = add_weather(defense); print(defense.shape[0])


team_qb_self = create_self_cols(team_qb)
defense = pd.merge(defense, team_qb_self, on=['team', 'week', 'year'])
defense = pd.merge(defense, team_qb.rename(columns={'team': 'offTeam'}), 
                    on=['offTeam', 'week', 'year'], how='left')

team_proj_self = create_self_cols(team_proj)
defense = pd.merge(defense, team_proj_self, on=['team', 'week', 'year'])
defense = pd.merge(defense, team_proj.rename(columns={'team': 'offTeam'}), 
                   on=['offTeam', 'week', 'year']); print( defense.shape[0])

team_stats = get_team_stats()
defense = pd.merge(defense, team_stats.rename(columns={'team': 'offTeam'}), 
                   on=['offTeam', 'week', 'year']); print(defense.shape[0])

d_stats = get_defense_stats()
defense = pd.merge(defense, d_stats, on=['team', 'week', 'year'], how='inner')

pff_oline = pff_oline_rollup()
defense = pd.merge(defense, pff_oline.rename(columns={'team': 'offTeam'}), 
                    on=['offTeam', 'week', 'year']); print(defense.shape[0])

pff_def = pff_defense_rollup()
defense = pd.merge(defense, pff_def, on=['team', 'week', 'year'])

defense = defense.copy().rename(columns={'team': 'player'})
defense = forward_fill(defense)

defense = attach_y_act(defense, pos='Defense', defense=True)
defense = drop_y_act_except_current(defense, WEEK, YEAR); print(defense.shape[0])
defense = defense.dropna(axis=1, thresh=defense.shape[0]-100).dropna(); print(defense.shape[0])

print('Unique player-week-years:', defense[['player', 'week', 'year']].drop_duplicates().shape[0])
print('Team Counts by Week:', defense[['year', 'week', 'player']].drop_duplicates().groupby(['year', 'week'])['player'].count())

defense.columns = [c.replace('_dst', '') for c in defense.columns]
defense = remove_non_uniques(defense)
defense = remove_low_corrs(defense, corr_cut=0.03)

dm.write_to_db(defense, f'Model_Features_{YEAR}', f'Defense_Data_Week{WEEK}', if_exist='replace')


#%%

chk_week = 14
backfill_chk = dm.read(f'''SELECT player 
                           FROM Backfill_QB_Week{WEEK} 
                           WHERE week={chk_week} AND year={YEAR}
                           UNION
                           SELECT player 
                           FROM Backfill_RB_Week{WEEK} 
                           WHERE week={chk_week} AND year={YEAR}
                           UNION
                           SELECT player 
                           FROM Backfill_WR_Week{WEEK} 
                           WHERE week={chk_week} AND year={YEAR}
                           UNION
                           SELECT player 
                           FROM Backfill_TE_Week{WEEK} 
                           WHERE week={chk_week} AND year={YEAR}
                        ''', f'Model_Features_{YEAR}').player.values
sal = dm.read(f"SELECT player, salary FROM Salaries WHERE week={chk_week} AND year={YEAR}", 'Simulation')
sal[~sal.player.isin(backfill_chk)].sort_values(by='salary', ascending=False).iloc[:50]

#%%

from sklearn.metrics import r2_score, mean_squared_error

WEEK = 14
accuracy = []
for pos in ['QB', 'RB', 'WR', 'TE']:
    print(pos)
    df = dm.read(f'''SELECT * FROM Backfill_{pos}_Week{WEEK} WHERE year >= 2024''', f'Model_Features_{YEAR}')
    # player_vegas_stats = get_all_vegas_stats(pos)
    # df = fill_vegas_stats(df, player_vegas_stats, pos)

    pts_cols = ['projected_points', 'fantasyPoints', 'ProjPts', 'ffa_points',# 'fft_proj_pts',
                'fdta_proj_points', 'nf_proj_points', 'etr_proj_points', 'fpts_proj_points', 
                'vegas_proj_points', 'avg_proj_points', 'avg_vegas_proj_points', 'good_avg_proj_points']
    
    pred = dm.read(f'''SELECT player, week, year, pred_fp_per_game
                            FROM Model_Predictions
                            WHERE model_type = 'backfill'
                                --AND year=2024
                                AND std_dev_type = 'spline_class80_q80_matt0_brier1_kfold3'
                                AND pos = '{pos}'
                                AND reg_ens_vers='random_full_stack_newp_sera0_rsq0_mse1_include2_kfold3'
                                AND pred_vers='sera0_rsq0_mse1_brier1_matt0_optuna_tpe_numtrials100_higherkb'
                        ''', 'Simulation')
    
    pred = pd.merge(pred, df, on=['player', 'week', 'year'])
    pred['good_avg_proj_points'] = pred[['etr_proj_points', 'fpts_proj_points', 'vegas_proj_points', 
                                         'fdta_proj_points', 'projected_points', 'nf_proj_points', 
                                         ]].mean(axis=1)
    # pred = pred.loc[(pred.year<=2024) & (pred.week!=WEEK)].reset_index(drop=True)
    pts_cols.append('pred_fp_per_game')
    for c in pts_cols:
        
        rsq = r2_score(pred.y_act, pred[c])
        msq = np.sqrt(mean_squared_error(pred.y_act, pred[c]))
        accuracy.append([pos, c, rsq, msq])

accuracy = pd.DataFrame(accuracy).sort_values(by=1, ascending=False)
accuracy.columns = ['pos', 'pred_source', 'r2', 'rmse']
accuracy['pos_r2_rank'] = accuracy.groupby('pos')['r2'].rank(ascending=False)
accuracy['pos_rmse_rank'] = accuracy.groupby('pos')['rmse'].rank(ascending=True)
accuracy['total_r2'] = accuracy.groupby('pred_source')['r2'].transform('sum')

print(accuracy.groupby('pred_source')['pos_r2_rank'].mean().sort_values())
print(accuracy.groupby('pred_source')['r2'].sum().sort_values(ascending=False))
print(accuracy.groupby('pred_source')['rmse'].sum().sort_values(ascending=True))

accuracy = accuracy.sort_values(by=['pos', 'r2'], ascending=[True, False])
accuracy
#%%

from skmodel import SciKitModel
from hyperopt import Trials
from sklearn.metrics import r2_score
import datetime as dt


def year_week_to_date(x):
    return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))


def get_cv_time_input(df, back_weeks, test_year, test_week):
    df = df[(df.year < test_year) | \
            (
              (df.year==test_year) & \
              (df.week<=test_week)
            )]
    max_date = str(df.game_date.max())
    year = int(max_date[:4])
    week = int(max_date[-2:])

    for i in range(back_weeks):
        if week > 1:
            week -= 1
        else:
            year -= 1
            week = 17
    cv_time_input = int(dt.datetime(year, 1, week).strftime('%Y%m%d'))
    cv_time_input = np.max([cv_time_input, 20200114])
    print(f'Begin Validation Using {cv_time_input}')
    return cv_time_input


def create_game_date(df, pos, test_year, test_week):
    
    # set up the date column for sorting
    df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
    cv_time_input = get_cv_time_input(df, 32, test_year, test_week)
    train_time_split = int(dt.datetime(test_year, 1, test_week).strftime('%Y%m%d'))

    return df, cv_time_input, train_time_split



pos = 'QB'
test_week = 11
test_year = 2024

model_obj = 'reg'
alpha = 0.8
if model_obj =='class': proba = True
else: proba = False

# Xy = dm.read(f'''SELECT * FROM Backfill_{pos}_Week{test_week}''', f'Model_Features_{test_year}')
Xy1 = dm.read(f'''SELECT * FROM {pos}_Data_Week{test_week}''', f'Model_Features_{test_year}')
Xy2 = dm.read(f'''SELECT * FROM {pos}_Data_Week{test_week}_v2''', f'Model_Features_{test_year}')
Xy = pd.concat([Xy1, Xy2], axis=1)
Xy = Xy.sort_values(by=['year', 'week']).reset_index(drop=True)

Xy, cv_time_input, train_time_split = create_game_date(Xy, pos, test_year, test_week)
train = Xy[Xy.game_date < train_time_split]
pred = Xy[Xy.game_date >= train_time_split]

preds = []
actuals = []

skm = SciKitModel(train, model_obj=model_obj, alpha=alpha, hp_algo='atpe')
to_drop = list(train.dtypes[train.dtypes=='object'].index)
X, y = skm.Xy_split('y_act', to_drop = to_drop)

if proba:
    p = 'select_perc_c'
    kb = 'k_best_c'
    m = 'lr_c'
else:
    p = 'select_perc'
    kb = 'k_best'
    m = 'lgbm'

pipe = skm.model_pipe([skm.piece('random_sample'),
                        skm.piece('std_scale'), 
                        # skm.piece(p),
                        # skm.feature_union([
                        #                skm.piece('agglomeration'), 
                        #                 skm.piece(f'{kb}_fu'),
                        #                 skm.piece('pca')
                        #                 ]),
                        skm.piece(kb),
                        skm.piece(m)
                    ])

params = skm.default_params(pipe, 'bayes')

# pipe.steps[-1][-1].set_params(**{'loss_function': f'Quantile:alpha={alpha}'})
best_models, oof_data, param_scores, _ = skm.time_series_cv(pipe, X, y, params, n_iter=10,
                                                                col_split='game_date',n_splits=5,
                                                                time_split=cv_time_input, alpha=alpha,
                                                                bayes_rand='bayes', proba=proba,
                                                                sample_weight=False, trials=Trials(),
                                                                random_seed=64893)

print('R2 score:', r2_score(oof_data['full_hold']['y_act'], oof_data['full_hold']['pred']))
oof_data['full_hold'].plot.scatter(x='pred', y='y_act')
# try: show_calibration_curve(oof_data['full_hold'].y_act, oof_data['full_hold'].pred, n_bins=6)
# except: pass

oof_data['full_hold'].sort_values(by='pred', ascending=False).iloc[:50]
# oof_data['full_hold'][(oof_data['full_hold'].pred >15) & (oof_data['full_hold'].y_act < 7)]

# %%

try: 
    pred['pred'] = best_models[-1].fit(X,y).predict_proba(pred[X.columns].fillna(pred.mean()))[:,1]
except: 
    pred['pred'] = best_models[-1].fit(X,y).predict(pred[X.columns].fillna(pred.mean()))

pred[['player', 'year', 'pred']].sort_values(by='pred', ascending=False).iloc[:35]

# %%


import matplotlib.pyplot as plt

for i in range(4):
    pipeline = best_models[i]
    pipeline.fit(X,y)
    # Extract the coefficients
    log_reg = pipeline.named_steps[m]

    try:
        log_reg.coef_.shape[1]
        coefficients = log_reg.coef_[0]
    except: 
        try:
            coefficients = log_reg.coef_
        except:
            coefficients = log_reg.feature_importances_

    # Get the feature names from SelectKBest
    rand_features = pipeline.steps[0][1].columns
    X_out = X[rand_features]
    selected_features = pipeline.named_steps[kb].get_support(indices=True)

    coef = pd.Series(coefficients, index=X_out.columns[selected_features])
    coef = coef[np.abs(coef) > 0.01].sort_values()
    coef.plot(kind = 'barh', figsize=(8, len(coef)/3))
    plt.show()
# %%


# %%
