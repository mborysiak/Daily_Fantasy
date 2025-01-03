#%%

YEAR = 2024
WEEK = 18

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
                          AND year >= 2021
                          AND (week <= 17 OR year={YEAR})
                         ''', 'Pre_PlayerData')
    fp = name_cleanup(fp)
    if pos == 'DST': fp = fp.drop('player', axis=1).rename(columns={'team': 'player'})

    return fp


def numberfire_proj(df):
    nf = dm.read(f"SELECT * FROM NumberFire", 'Pre_PlayerData')
    df = pd.merge(df, nf, on=['player', 'week','year'], how='left')

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
    
    fd = dm.read(f'''SELECT player as team, * 
                     FROM FantasyData_Defense
                    ''', 'Pre_PlayerData')
    
    fd = fd.drop(['fdta_rank'], axis=1)

    fd['fdta_rank'] = fd.groupby(['year', 'week'])['fdta_proj_points'].rank(ascending=False, method='min')
    
    df = pd.merge(df, fd, on=['player', 'week', 'year'], how='left')

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


def get_salaries(df, pos):
    sal = dm.read(f'''SELECT player, week, year, 
                             dk_salary, fd_salary, yahoo_salary
                    FROM Daily_Salaries
                    WHERE Position='{pos}'
                    ''', 'Pre_PlayerData')
    df = pd.merge(df, sal, on=['player', 'week', 'year'])
    df.loc[(df.fd_salary < 100) | (df.fd_salary > 10000), 'fd_salary'] = np.nan

    return df

def get_salaries_defense(df):
    sal = dm.read(f'''SELECT DISTINCT team player, week, year, 
                                      dk_salary, fd_salary, yahoo_salary
                    FROM Daily_Salaries
                    ''', 'Pre_TeamData')
    df = pd.merge(df, sal, on=['player', 'week', 'year'])
    df.loc[(df.fd_salary < 100) | (df.fd_salary > 10000), 'fd_salary'] = np.nan

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
    
    else:
        y_act = dm.read(f'''SELECT player, team, week, season year, fantasy_pts y_act
                            FROM {pos}_Stats
                            WHERE season >= 2020''', 'FastR')
                            
        snaps = get_snap_data()
        proj = dm.read('''SELECT player, week, year, projected_points
                          FROM FantasyPros''', 'Pre_PlayerData')

        y_act = pd.merge(y_act, snaps, on=['player', 'week', 'year'], how='left')
        y_act = pd.merge(y_act, proj, on=['player', 'week', 'year'], how='left')

        y_act = y_act[~((y_act.projected_points > 8) & \
                        (y_act.snap_pct < y_act.avg_snap_pct*0.5) & \
                        (y_act.snap_pct <= 0.4) & \
                        (y_act.snap_pct > 0))].drop(['snap_pct', 'avg_snap_pct', 'projected_points'], axis=1)
        
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

def get_all_vegas_stats(pos):
    df = dm.read(f'''SELECT * FROM Vegas_Clean_{pos} ''', 'Pre_PlayerData')
    return df

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



#############################

def log_rank_cols(df):
    rank_cols = [c for c in df.columns if 'rank' in c or 'expert' in c]
    for c in rank_cols:
        df['log_' + c] = np.log(df[c]+1)
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
        tp = get_salaries(tp, pos)
        tp = numberfire_proj(tp)
        tp = fantasy_data_proj(tp, pos)
        tp = etr_projection(tp, pos)
        tp = fantasy_points_projection(tp, pos)

        tp = consensus_fill(tp)
        tp = log_rank_cols(tp)
        
        player_vegas_stats = get_all_vegas_stats(pos)
        tp = fill_vegas_stats(tp, player_vegas_stats, pos)

        tp_this_week = tp.copy()
        tp_this_week = tp_this_week[(tp_this_week.week==WEEK) & (tp_this_week.year==YEAR)]
        tp_other_weeks = tp.copy()
        tp_other_weeks = tp_other_weeks[~((tp_other_weeks.week==WEEK) & (tp_other_weeks.year==YEAR))]

        tp = pd.concat([tp_this_week, tp_other_weeks], axis=0).reset_index(drop=True)
        tp = drop_duplicate_players(tp)

        team_proj = pd.concat([team_proj, tp], axis=0)

    team_proj = create_pos_rank(team_proj)
    team_proj = forward_fill(team_proj)
    team_proj = team_proj.fillna(0)

    cnts = team_proj.groupby(['team', 'week', 'year']).agg({'avg_proj_points': 'count'})
    print('Team counts that do not equal 7:', cnts[cnts.avg_proj_points!=7])

    cols = [
        'projected_points', 'avg_proj_points', 'log_rankadj_fp_rank', 'log_playeradj_fp_rank', 'log_fp_rank', 'dk_salary', 'fd_salary',
      # 'ffa_points', 'ffa_rush_yds', 'ffa_rush_tds','ffa_rec', 'ffa_rec_yds','ffa_rec_tds',
        'fdta_rush_yds', 'fdta_rush_td', 'fdta_rec', 'fdta_rec_yds', 'fdta_rec_td', 'fdta_proj_points',
        'etr_proj_points', 'etr_proj_ceiling', 'etr_proj_floor', 'etr_rush_yds', 'etr_rush_tds', 'etr_rec', 'etr_rec_yds', 'etr_rec_td',
        'nf_rushing_yds', 'nf_rushing_tds', 'nf_receiving_rec', 'nf_receiving_yds', 'nf_receiving_tds',
        'fpts_rush_yds', 'fpts_rec', 'fpts_rush_td', 'fpts_rec_td', 'fpts_rush_yds', 'fpts_rec_yds',
        'avg_proj_rush_att', 'avg_proj_rush_td', 'avg_proj_rec', 'avg_proj_rec_tgts','avg_proj_rec_yds', 'avg_proj_rec_td', 
        'avg_vegas_proj_points', 'vegas_proj_points',
        'player_rush_yds_ev_trunc_norm', 'player_rush_attempts_ev_trunc_norm', 'player_receptions_ev_trunc_norm', 'player_reception_yds_ev_trunc_norm', 
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



def get_max_qb():

    df = fantasy_pros_new('QB')
    df = fantasy_data_proj(df, 'QB')
    df = numberfire_proj(df)
    df = etr_projection(df, 'QB')
    df = fantasy_points_projection(df, 'QB')
    df = get_salaries(df, 'QB')

    df = consensus_fill(df)

    player_vegas_stats = get_all_vegas_stats('QB')
    df = fill_vegas_stats(df, player_vegas_stats, 'QB')
    df = log_rank_cols(df)

    qb_cols = [
               'team', 'week', 'year', 
              #'ffa_points', 'ffa_pass_yds', 'ffa_pass_tds','ffa_pass_int','ffa_rush_yds', 'ffa_rush_tds',
               'fdta_pass_yds', 'fdta_pass_td', 'fdta_rush_yds','fdta_rush_td',
               'etr_proj_points', 'etr_proj_ceiling', 'etr_proj_floor', 'etr_pass_yds', 'etr_pass_tds',
               'etr_rush_yds', 'etr_rush_tds', 'etr_pass_att', 'etr_pass_cmp',
               'nf_passing_yds', 'nf_passing_tds', 'nf_rushing_att', 'nf_rushing_tds', 'nf_rushing_yds',
               'avg_proj_pass_yds', 'avg_proj_pass_td',  'avg_proj_pass_int', 'avg_proj_pass_att', 'avg_proj_pass_comp',
               'avg_proj_rush_yds',  'avg_proj_rush_att', 'avg_proj_rush_td', 
               'projected_points', 'avg_proj_points',
               'log_avg_proj_rank', 'log_playeradj_fp_rank','fp_rank','log_rankadj_fp_rank', 'dk_salary',
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

def rolling_stats(df, gcols, rcols, period, agg_type='mean'):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    rolls = df.groupby(gcols)[rcols].rolling(period, min_periods=1).agg(agg_type).reset_index(drop=True)
    rolls.columns = [f'r{agg_type}{period}_{c}' for c in rolls.columns]

    return rolls

def add_rolling_stats(df, gcols, rcols, perform_check=True):

    df = df.sort_values(by=[gcols[0], 'year', 'week']).reset_index(drop=True)

    if perform_check:
        cnt_check = df.groupby([gcols[0], 'year'])['week'].count()
        print(f'Counts of Groupby Category Over 17: {cnt_check[cnt_check>17]}')

    rolls3_mean = rolling_stats(df, gcols, rcols, 3, agg_type='mean')
    rolls3_max = rolling_stats(df, gcols, rcols, 3, agg_type='max')
    rolls5_std = rolling_stats(df, gcols, rcols, 5, agg_type='std')

    rolls8_mean = rolling_stats(df, gcols, rcols, 8, agg_type='mean')
    rolls8_max = rolling_stats(df, gcols, rcols, 8, agg_type='max')
    rolls8_std = rolling_stats(df, gcols, rcols, 8, agg_type='std')

    df = pd.concat([df, 
                    rolls8_mean, rolls8_max, rolls8_std,
                    rolls3_mean, rolls3_max, rolls5_std
                    ], axis=1)

    return df

def rolling_proj_stats(df):
    df = forward_fill(df)
    proj_cols = [c for c in df.columns if 'proj' in c]
    df = add_rolling_stats(df, ['player'], proj_cols)

    for c in proj_cols:
        df[f'trend_diff_{c}'] = df[f'rmean3_{c}'] - df[f'rmean8_{c}']
        df[f'trend_chg_{c}'] = df[f'trend_diff_{c}'] / (df[f'rmean8_{c}']+5)

    return df

#%%

# create the scores and lines table
create_scores_lines_table(WEEK, YEAR)

team_proj, team_proj_pos = get_team_projections() # add injury removal here in the future
team_qb = get_max_qb()

#%%

pos = 'QB'
full_model = True

for full_model in [True, False]:

    # pull all projections and ranks
    df = fantasy_pros_new(pos); print(df.shape[0])
    df = fantasy_data_proj(df, pos); print(df.shape[0])
    df = numberfire_proj(df); print(df.shape[0])
    df = etr_projection(df, pos); print(df.shape[0])
    df = fantasy_points_projection(df, pos); print(df.shape[0])
    df = get_salaries(df, pos); print(df.shape[0])

    # clean up any missing values and engineer data
    df = consensus_fill(df); print(df.shape[0])

    player_vegas_stats = get_all_vegas_stats(pos)
    df = fill_vegas_stats(df, player_vegas_stats, pos)

    if full_model:
        df =  rolling_proj_stats(df); print(df.shape[0])
        df = pd.merge(df, team_proj, on=['team', 'week', 'year']); print( df.shape[0])

    df = attach_y_act(df, pos)
    df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])

    # fill in missing data and drop any remaining rows
    df = forward_fill(df)
    df = df.dropna(axis=1, thresh=df.shape[0]-100).dropna().reset_index(drop=True); print(df.shape[0])

    df = one_qb_per_week(df); print(df.shape[0])

    df = remove_non_uniques(df)
    df = df[df.projected_points > 7.5].reset_index(drop=True)
    df = drop_duplicate_players(df)
    df = replace_inf_values(df)
    df = remove_low_corrs(df, corr_cut=0.02)

    print('Total Rows:', df.shape[0])
    print('Unique player-week-years:', df[['player', 'week', 'year']].drop_duplicates().shape[0])
    print('Team Counts by Week:', df[['year', 'week', 'team']].drop_duplicates().groupby(['year', 'week'])['team'].count())

    if full_model: table_name = f'QB_Data_Quick_Week{WEEK}'
    else: table_name = f'Backfill_QB_Data_Quick_Week{WEEK}'
    dm.write_to_db(df, f'Model_Features_{YEAR}', table_name, if_exist='replace', create_backup=True)


#%%

for pos in ['RB', 'WR', 'TE']:

    for full_model in [True, False]:

        # pull all projections and ranks
        df = fantasy_pros_new(pos); print(df.shape[0])
        df = fantasy_data_proj(df, pos); print(df.shape[0])
        df = numberfire_proj(df); print(df.shape[0])
        df = etr_projection(df, pos); print(df.shape[0])
        df = fantasy_points_projection(df, pos); print(df.shape[0])
        df = get_salaries(df, pos); print(df.shape[0])

        # clean up any missing values and engineer data
        df = consensus_fill(df); print(df.shape[0])

        player_vegas_stats = get_all_vegas_stats(pos)
        df = fill_vegas_stats(df, player_vegas_stats, pos)
        
        if full_model:
            df =  rolling_proj_stats(df); print(df.shape[0])
            df = pd.merge(df, team_qb, on=['team', 'week', 'year'], how='left'); print(df.shape[0])
            df = pd.merge(df, team_proj, on=['team', 'week', 'year']); print( df.shape[0])
            df = pd.merge(df, team_proj_pos, on=['pos', 'team', 'week', 'year']); print( df.shape[0])
            df = proj_market_share(df, 'team_proj_'); print(df.shape[0])
            df = proj_market_share(df, 'pos_proj_'); print(df.shape[0])

        df = attach_y_act(df, pos)
        df = drop_y_act_except_current(df, WEEK, YEAR); print(df.shape[0])

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

        if full_model: table_name = f'{pos}_Data_Quick_Week{WEEK}'
        else: table_name = f'Backfill_{pos}_Data_Quick_Week{WEEK}'
        dm.write_to_db(df, f'Model_Features_{YEAR}', table_name, if_exist='replace')

#%%

def create_self_cols(df):
    self_df = df.copy()
    self_cols = ['team', 'week','year']
    self_cols.extend(['self_'+c for c in self_df.columns if c not in ('team', 'week', 'year')])
    self_df.columns = self_cols
    return self_df

# defense stats that can be added to the offensive player data
defense = fantasy_pros_new('DST')
defense = fantasy_data_proj_def(defense)
defense = fantasy_points_projection_def(defense)
defense = etr_projection_def(defense)
defense = get_salaries_defense(defense)

defense = consensus_fill(defense, is_dst=True)
defense = defense.dropna(axis=1, thresh=defense.shape[0]-100)

all_cols = [c for c in defense.columns if c != 'y_act']
defense = defense.dropna(subset=all_cols)

defense = add_gambling_lines(defense); print(defense.shape[0])
defense = pd.merge(defense, 
                   team_qb.rename(columns={'team': 'opp'}), 
                   on=['opp', 'week', 'year'], how='left')
defense = pd.merge(defense, 
                   team_proj.rename(columns={'team': 'opp'}), 
                   on=['opp', 'week', 'year'], how='left')
defense = forward_fill(defense)

defense = attach_y_act(defense, pos='Defense', defense=True)
defense = drop_y_act_except_current(defense, WEEK, YEAR); print(defense.shape[0])
defense = defense.dropna(axis=1, thresh=defense.shape[0]-100).dropna(); print(defense.shape[0])

print('Unique player-week-years:', defense[['player', 'week', 'year']].drop_duplicates().shape[0])
print('Team Counts by Week:', defense[['year', 'week', 'player']].drop_duplicates().groupby(['year', 'week'])['player'].count())

defense.columns = [c.replace('_dst', '') for c in defense.columns]
defense = remove_non_uniques(defense)
defense = drop_duplicate_players(defense)
defense = remove_low_corrs(defense, corr_cut=0.03)

dm.write_to_db(defense, f'Model_Features_{YEAR}', f'Defense_Data_Quick_Week{WEEK}', if_exist='replace')


#%%

chk_week = 18
backfill_chk = dm.read(f'''SELECT player 
                           FROM Backfill_QB_Data_Quick_Week{WEEK}
                           WHERE week={chk_week} AND year={YEAR}
                           UNION
                           SELECT player 
                           FROM Backfill_RB_Data_Quick_Week{WEEK}
                           WHERE week={chk_week} AND year={YEAR}
                           UNION
                           SELECT player 
                           FROM Backfill_WR_Data_Quick_Week{WEEK}
                           WHERE week={chk_week} AND year={YEAR}
                           UNION
                           SELECT player 
                           FROM Backfill_TE_Data_Quick_Week{WEEK}
                           WHERE week={chk_week} AND year={YEAR}
                        ''', f'Model_Features_{YEAR}').player.values
sal = dm.read(f"SELECT player, salary FROM Salaries WHERE week={chk_week} AND year={YEAR}", 'Simulation')
sal[~sal.player.isin(backfill_chk)].sort_values(by='salary', ascending=False).iloc[:50]


#%%

from skmodel import SciKitModel
import optuna
from sklearn.metrics import r2_score
import datetime as dt

def show_calibration_curve(y_true, y_pred, n_bins=10):

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    
    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='quantile')
    plt.plot(y, x, marker = '.', label = 'Quantile')

    # Plot model's calibration curve
    x, y = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
    plt.plot(y, x, marker = '+', label = 'Uniform')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.show()

    print('Brier Score:', brier_score_loss(y_true, y_pred))

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

def select_main_slate_teams(df, set_pos):

    import datetime as dt

    good_teams = dm.read(f'''
                    SELECT away_team team, gametime, week, year 
                    FROM Gambling_Lines 
                    WHERE year >= 2020
                    UNION
                    SELECT home_team team, gametime, week, year 
                    FROM Gambling_Lines
                    WHERE year >= 2020
                ''', 'Pre_TeamData')


    good_teams.gametime = pd.to_datetime(good_teams.gametime)
    good_teams['day_of_week'] = good_teams.gametime.apply(lambda x: x.weekday())
    good_teams['hour_in_day'] = good_teams.gametime.apply(lambda x: x.hour)

    good_teams = good_teams[(good_teams.day_of_week==6) & (good_teams.hour_in_day <= 17) & (good_teams.hour_in_day > 10)]
    good_teams = good_teams[['team', 'week', 'year']]

    if set_pos == 'Defense': 
        good_teams = good_teams.rename(columns={'team': 'player'})
        df = pd.merge(df, good_teams, on=['player', 'week', 'year'])

    else:
        df = pd.merge(df, good_teams, on=['team', 'week', 'year'])

    return df


def add_sal_columns(df):

    for c in df.columns:
        if 'proj' in c: df[c+'_salary'] = df[c] / df.dk_salary
     
    return df

def predict_million_df(df, pos, pred_mil_or_roi):

    if pred_mil_or_roi=='mil': table_name = 'Top_Players'
    elif pred_mil_or_roi=='roi': table_name = 'Top_Players_ROI'
    else: raise ValueError('pred_mil_or_roi must be mil or roi')

    df = select_main_slate_teams(df, pos)

    df = df.drop('y_act', axis=1)
    top_players = dm.read(f"SELECT player, week, year, y_act FROM {table_name}", "DK_Results")

    df = pd.merge(df, top_players, on=['player', 'week', 'year'], how='left')
    df = df.fillna({'y_act': 0})
    df = add_sal_columns(df)
    df = add_ownership_cols(df)

    return df

#%%

def add_ownership_cols(df):
    ownership = dm.read(f'''SELECT player, 
                                    week, 
                                    year, 
                                    pred_ownership,
                                    std_dev std_dev_ownership,
                                    min_score min_score_ownership,
                                    max_score max_score_ownership
                            FROM Predicted_Ownership_Only
                            WHERE ownership_vers = 'standard_ln'
                        ''', 'Simulation')

    ownership['pred_ownership_exp'] = np.exp(ownership.pred_ownership)*100
    ownership['min_score_ownership_exp'] = np.exp(ownership.min_score_ownership)*100
    ownership['max_score_ownership_exp'] = np.exp(ownership.max_score_ownership)*100
    
    df = pd.merge(df, ownership[['player', 'week', 'year']], on=['player', 'week', 'year'])

    # df = pd.merge(df, ownership, on=['player', 'week', 'year'])
    # df['avg_proj_points_per_own'] = df.avg_proj_points / df.pred_ownership_exp
    # df['avg_vegas_proj_points_per_own'] = df.avg_vegas_proj_points / df.pred_ownership_exp
    # df['good_avg_proj_points_per_own'] = df.good_avg_proj_points / df.pred_ownership_exp
    # df['avg_proj_points_times_own'] = df.avg_proj_points * -df.pred_ownership
    # df['avg_vegas_proj_points_times_own'] = df.avg_vegas_proj_points * -df.pred_ownership
    # df['good_avg_proj_points_times_own'] = df.good_avg_proj_points * -df.pred_ownership

    return df

#%%
pos = 'WR'
test_week = 18
test_year = 2024

model_obj = 'class'
pred_mil_or_roi = 'mil'

if model_obj =='class': proba = True
else: proba = False
alpha = None

Xy = dm.read(f'''SELECT * FROM {pos}_Data_Quick_Week{test_week}''', f'Model_Features_{test_year}')
Xy = Xy.sort_values(by=['year', 'week']).reset_index(drop=True)

if pred_mil_or_roi is not None:
    Xy = predict_million_df(Xy, pos, pred_mil_or_roi)

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
    m = 'cb_c'
else:
    p = 'select_perc'
    kb = 'k_best'
    m = 'cb'

trials = optuna.create_study(
            storage=f"sqlite:///optuna/experiments/{m}_{pos}_{model_obj}_{pred_mil_or_roi}.sqlite3", 
        )

pipe = skm.model_pipe([skm.piece('random_sample'),
                       # skm.piece('std_scale'), 
                        # skm.piece(p),
                        # skm.feature_union([
                        #                skm.piece('agglomeration'), 
                        #                 skm.piece(f'{kb}_fu'),
                        #                 skm.piece('pca')
                        #                 ]),
                        skm.piece(kb),
                        skm.piece(m)
                    ])

params = skm.default_params(pipe, 'optuna')

# pipe.steps[-1][-1].set_params(**{'loss_function': f'Quantile:alpha={alpha}'})
best_models, oof_data, param_scores, _ = skm.time_series_cv(pipe, X, y, params, n_iter=20,
                                                                col_split='game_date',n_splits=5,
                                                                time_split=cv_time_input, alpha=alpha,
                                                                bayes_rand='optuna', proba=proba,
                                                                sample_weight=False, trials=trials,
                                                                random_seed=64893, optuna_timeout=60*5)

print('R2 score:', r2_score(oof_data['full_hold']['y_act'], oof_data['full_hold']['pred']))
oof_data['full_hold'].plot.scatter(x='pred', y='y_act')
try: show_calibration_curve(oof_data['full_hold'].y_act, oof_data['full_hold'].pred, n_bins=6)
except: pass

oof_data['full_hold'].sort_values(by='pred', ascending=False).iloc[:50]

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
