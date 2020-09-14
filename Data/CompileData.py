# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

conn = sqlite3.connect('/Users/Mark/Documents/Github/Daily_Fantasy/Data/Weekly_Stats.sqlite3')


def find_first_name(df1, df2):
    name_map = {}
    players = df1.player.sort_values().unique()
    dvoa_players = df2.player.sort_values().unique()
    for p in players:
        n_split = p.replace(' II', '').replace(' Jr.', '').split(' ')
        key = n_split[0][0] + '.' + ' '.join(n_split[1:])
        key2 = n_split[0][0] + n_split[0][1] + '.' + ' '.join(n_split[1:])
        try:
            key3 = n_split[0][0] + n_split[0][1] + n_split[0][2] + '.' + ' '.join(n_split[1:])
        except:
            pass
        if key in dvoa_players:
            name_map[key] = p
        if key2 in dvoa_players:
            name_map[key2] = p
        try:
            if key3 in dvoa_players:
                name_map[key3] = p
        except:
            pass
    df2.player = df2.player.map(name_map)

    return df2


def set_to_float(df):
    '''
    Loop through each column and attempt to set the column as float
    '''
    for c in df.columns:
        try:
            df[c] = df[c].astype('float')
        except:
            pass
        
    return df


def strip_and_float(df, c, to_strip, replace_with=''):
    '''
    Function that strips characters and converts to string for specific columns
    '''
    df[c] = df[c].apply(lambda x: float(x.replace(to_strip, replace_with)))
    
    return df


def clean_rz(df):
    '''
    Clean up the names from the redzone data files
    '''
    df.player = df.player.apply(lambda x: x.split(' ')[-1] + ' ' + ' '.join(x.split(' ')[:-1]))
    df.player = df.player.apply(lambda x: x.rstrip().lstrip())
    df.loc[df.player == 'LeVeon Bell', 'player'] = "Le'Veon Bell"
    
    return df


def add_dummies(df, cols, prefix=None):
    '''
    Add one-hot-encoded dummy columns for each column passed into the function
    '''
    dummies = pd.get_dummies(df[cols], prefix=prefix)
    df = pd.concat([df, dummies], axis=1).drop(cols, axis=1)
    return df


def get_per_game_avg(df, col):
    '''
    Calculate the per game average for a given total column
    '''
    df[col + '_per_game'] = df[col] / df.games
    
    return df


def rolling_stats(df, gcols, rcol, period):
    '''
    Calculate rolling mean stats over a specified number of previous weeks
    '''
    
    # calculate the mean over a given groupby for a given period
    gmean = df.groupby(gcols)[rcol].rolling(period).mean().reset_index().drop('level_' + str(len(gcols)), axis=1)
    gmean = gmean.rename(columns={rcol: 'mean_' + rcol + '_' + str(period)})

    # calculate the max week (necessary for accounting for bye weeks)
    # and concat the weeks with the rolling mean results
    weeks = df.groupby(gcols)['week'].rolling(period).max().reset_index().drop('level_' + str(len(gcols)), axis=1)
    to_merge = pd.concat([gmean, weeks.week], axis=1).dropna()
    
    # merge the rolling stats back to the original dataframe
    gcols.append('week')
    df = pd.merge(df, to_merge, how='left', left_on=gcols, right_on=gcols)

    return df


# # Team Stats

# ## Team Offense

# ## Pre-Game

# +
#----------
# Split out the O-line and D-line matchups
#----------

# read in the oline-dline matchup table
ol_dl = pd.read_sql_query('SELECT * FROM PFF_Oline_Dline_Matchups', conn)

# pull out the offensive line statistics
ol = ol_dl[['team', 'pressure_rate_allow', 'sack_conversion_allow', 'yds_before_contact_generate',
            'runs_inside_5_generate', 'td_pct_inside_5_generate', 'pass_block_advantage',
            'run_block_advantage', 'opponent', 'week', 'year']]

# pull out the defensive line statistics and re-arrange names for merging
dl = ol_dl[['team', 'opponent', 'pressure_rate_generate', 'sack_conversion_generate', 
            'yds_before_contact_allow', 'runs_inside_5_allow', 'td_pct_inside_5_allow', 'week', 'year']]
dl = dl.rename(columns = {'team': 'opponent', 'opponent': 'team'})

#----------
# Convert the gambling odds to long format and calculate implied score
#----------

# pull in the gambling lines and split out the home and away teams
lines = pd.read_sql_query('SELECT * FROM Gambling_Lines', conn)

home = lines[['home_team', 'home_line', 'home_moneyline', 'over_under', 'week', 'year']]
home.columns = ['team', 'line', 'moneyline', 'over_under', 'week', 'year']

away = lines[['away_team', 'away_line', 'away_moneyline', 'over_under', 'week', 'year']]
away.columns = ['team', 'line', 'moneyline', 'over_under', 'week', 'year']

# concatenate the home and away data
lines = pd.concat([home, away], axis=0)
lines.line = lines.line.astype('float')
lines.over_under= lines.over_under.astype('float')

# calculate the implied for and against scores based on vegas
lines['implied_for'] = (lines.over_under / 2) - (lines.line / 2)
lines['implied_against'] = lines.over_under - lines.implied_for

#----------
# Add weather stats and convert to flags
#----------

# pull in weather query
weather = pd.read_sql_query('SELECT * FROM GameWeather', conn)

# merge offensive line stats with weather stats
off_pre = pd.merge(ol, weather, how='left', 
                   left_on=['team', 'week', 'year'], right_on=['team', 'week', 'year'])

# Because weather is only listed for one team, drop the null values from the joined dataframe,
# and then re-merge with the "opponent" column to ensure that every line in the original
# dataframe has a weather listing.
off_2 = off_pre.loc[off_pre.temp_high.isnull()].dropna(axis=1)
off_pre = off_pre.dropna(axis=0, thresh=off_pre.shape[1]-3)
off_2 = pd.merge(off_2, weather.rename(columns={'team':'opponent'}), how='inner', 
                 left_on=['opponent', 'week', 'year'], right_on=['opponent', 'week', 'year'])

# re-arrange the second dataframe and merge back to the original
off_2 = off_2[off_pre.columns]
off_pre = pd.concat([off_pre, off_2], axis=0).sort_values(by=['year', 'week']).reset_index(drop=True)

# add in the gambling lines to the team offense dataframe
off_pre = pd.merge(off_pre, lines, how='inner', 
                   left_on=['team', 'week', 'year'], right_on=['team', 'week', 'year'])

# fix the weather data to create flags
off_pre = add_dummies(off_pre, ['precip_type'])
off_pre.loc[(off_pre.precip_prob < 0.5) | (off_pre.precip_intensity < 0.01), 'precip_type_rain'] = 0
off_pre.loc[(off_pre.precip_prob < 0.5) | (off_pre.precip_intensity < 0.01), 'precip_type_snow'] = 0
off_pre['high_winds'] = 0
off_pre.loc[(off_pre.wind_speed > 20) & (off_pre.wind_gust > 35), 'high_winds'] = 1
off_pre['hot_temp'] = 0
off_pre.loc[off_pre.temp_high > 95, 'hot_temp'] = 1

#----------
# Create rolling variable flags
#----------

off_pre_roll_cols = ['pressure_rate_allow', 'sack_conversion_allow','yds_before_contact_generate', 
                     'runs_inside_5_generate', 'td_pct_inside_5_generate', 'pass_block_advantage', 
                     'run_block_advantage', 'over_under', 'implied_for', 'implied_against']

for c in off_pre_roll_cols:
    off_pre = rolling_stats(off_pre, gcols=['team', 'year'], rcol=c, period=3)
# -

# ## Post-Game

# +
off_post = pd.read_sql_query('''
    SELECT * FROM Team_Pass_Stats a
    INNER JOIN (SELECT * FROM Team_Rush_Stats) b on a.team = b.team and a.week = b.week and a.year = b.year
    LEFT JOIN (SELECT * FROM Team_Oline_Stats) c on a.team = c.team and a.week = c.week and a.year = c.year
    INNER JOIN (SELECT * FROM Team_Rush_Stats) d on a.team = d.team and a.week = d.week and a.year = d.year
    INNER JOIN (SELECT * FROM Team_Off_Stats) e on a.team = e.team and a.week = e.week and a.year = e.year
    INNER JOIN (SELECT * FROM DVOA_Team_Off) f on a.team = f.team and a.week = f.week and a.year = f.year
    INNER JOIN (SELECT * FROM DVOA_OL) g on a.team = g.team and a.week = g.week and a.year = g.year
''', conn)

# ensure that no columns have duplicated names
off_post = off_post.loc[:, ~off_post.columns.duplicated()]

# try to set all columns to float, if possible
off_post = set_to_float(off_post)

# for the remaining columns, strip out the characters that cause errors and convert to float
for c in ['team_pass_yds', 'team_rush_yds', 'rush_yds']:
    off_post = strip_and_float(off_post, c, ',')
    
for c in ['long_pass', 'team_long_rush']:
    off_post = strip_and_float(off_post, c, 'T')
    
for c in ['pct_short_yd_left', 'pct_short_yd_center', 'pct_short_yd_right']:
    off_post = strip_and_float(off_post, c, '--', '0')

# for any column that has null values, fill them using a rolling backfill method
nulls = off_post.isnull().sum()[off_post.isnull().sum() > 0].index
for n in nulls:
    off_post[n] = off_post.groupby('team')[n].apply(lambda x: x.fillna(method='bfill'))
    
# set the week forward by one, since these are actual stats that will be used to predict
# the upcoming week's games and performance
off_post['week'] = off_post['week'] + 1


# -

def convert_to_per_game(df, col):
    


off_post.loc[off_post.team == 'SEA', 'team_rush_att'].shift(1).fillna(0)

off_post[off_post.team == 'SEA'].sort_values(by='week').iloc[:, 15:30]

off_post[off_post.team == 'SEA'].iloc[:, :15]

off = pd.merge(off_pre, off_post, how='inner', 
               left_on=['team', 'week', 'year'], right_on=['team', 'week', 'year'])
off = off.sort_values(by=['team', 'year', 'week'], ascending=[True, True, True]).reset_index(drop=True)

# # Defense

# ## Pre-Game

# +
def_pre = pd.read_sql_query('''
    SELECT * FROM FantasyPros_Rankings a
    INNER JOIN (SELECT team, dk_salary, yahoo_salary, week, year FROM Daily_salaries where position = 'DST') b
        ON a.team = b.team and a.week = b.week and a.year = b.year
    WHERE a.position='DST'
''', conn)

# remove duplicated columns
def_pre = def_pre.loc[:, ~def_pre.columns.duplicated()]

# drop the full team name in column "player"
def_pre = def_pre.drop(['player', 'Opponent'], axis=1)

#
def_pre = pd.merge(def_pre, dl, how='inner', 
                   left_on=['team', 'week', 'year'], right_on=['team', 'week', 'year'])
def_pre = def_pre.loc[:, ~def_pre.columns.duplicated()]
# -

# ## Post-Game

# +
def_post = pd.read_sql_query('''
    SELECT * FROM DEF_Allowed_QB a
    JOIN (SELECT * FROM DEF_Allowed_RB) b ON a.team = b.team and a.week = b.week and a.year = b.year
    JOIN (SELECT * FROM DEF_Allowed_WR) c ON a.team = c.team and a.week = c.week and a.year = c.year
    JOIN (SELECT * FROM DEF_Allowed_TE) d ON a.team = d.team and a.week = d.week and a.year = d.year
    JOIN (SELECT * FROM DEF_Basic_Stat_Results) e ON a.team = e.team and a.week = e.week and a.year = e.year
    JOIN (SELECT * FROM DVOA_Team_Def) f ON a.team = f.team and a.week = f.week and a.year = f.year
    JOIN (SELECT * FROM DVOA_DL) g ON a.team = g.team and a.week = g.week and a.year = g.year
   -- JOIN (SELECT * FROM Team_Def_Stats) h ON a.team = h.team and a.week = h.week and a.year = h.year
   -- JOIN (SELECT * FROM Team_Def_Sacks) i ON a.team = i.team and a.week = i.week and a.year = i.year
''', conn)

# remove duplicated columns
def_post = def_post.loc[:, ~def_post.columns.duplicated()]

# shift forward one week to attach to current week's predictions
def_post['week'] = def_post['week'] + 1
# -

defense = pd.merge(def_pre, def_post, how='inner', 
                   left_on=['team', 'week', 'year'], right_on=['team', 'week', 'year'])

# # QB

# ## Pre-Game

# +
qb_pre = pd.read_sql_query('''
    SELECT * FROM FantasyPros_Rankings a
    INNER JOIN (SELECT * FROM Daily_salaries) b ON a.player = b.player and a.week = b.week and a.year = b.year
    LEFT JOIN (SELECT * FROM NumberFire_Rankings) c ON a.player = c.player and a.week = c.week and a.year = c.year
    INNER JOIN (SELECT * FROM QB_FantasyPros_Matchups) d ON a.player = d.player and a.week = d.week and a.year = d.year
    INNER JOIN (SELECT * FROM QB_PRF_Matchups) e ON a.player = e.player and a.week = e.week and a.year = e.year
    INNER JOIN (SELECT * FROM Roto_Grinders_Projections) f ON a.player = f.player and a.week = f.week and a.year = f.year
    WHERE a.position = 'QB'


''', conn)

# remove duplicated columns
qb_pre = qb_pre.loc[:, ~qb_pre.columns.duplicated()]

# fill in null values due to bad data gathering from numbefire for initial weeks
nulls = qb_pre.isnull().sum()[qb_pre.isnull().sum() > 0].index
for n in nulls:
    qb_pre[n] = qb_pre.groupby('player')[n].apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
# -

# ## Post-Game

# +
qb_post = pd.read_sql_query('''
    SELECT * FROM QB_Basic_Stat_Results a
    LEFT JOIN (SELECT * FROM QBR) b ON a.player = b.player and a.week = b.week and a.year = b.year
    LEFT JOIN (SELECT * FROM PFR_Redzone_Pass) c ON a.player = c.player and a.week = c.week and a.year = c.year
    INNER JOIN (SELECT * FROM PFR_QB_Stats) d ON a.player = d.player and a.week = d.week and a.year = d.year
    INNER JOIN (SELECT * FROM PFR_QB_Adv) e ON a.player = e.player and a.week = e.week and a.year = e.year
    INNER JOIN (SELECT * FROM Snap_Counts) g ON a.player = g.player and d.week = g.week and a.year = g.year
''', conn)

# remove duplicated columns
qb_post = qb_post.loc[:, ~qb_post.columns.duplicated()]

# +
qbr_nulls = ['rank', 'pts_added', 'pass_added', 'run_added', 'penalty_added', 'total_epa', 'qb_plays', 
             'raw_qbr', 'total_qbr']
for n in qbr_nulls:
    qb_post[n] = qb_post.groupby('player')[n].apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    
qb_post = qb_post.fillna(0)

# +
df1 = pd.read_sql_query('''SELECT * FROM QB_Basic_Stat_Results''', conn)
df2 = pd.read_sql_query('''SELECT * FROM DVOA_QB''', conn)
dvoa_qb = find_first_name(df1, df2)

qb_post = pd.merge(qb_post, dvoa_qb, how='inner', 
                   left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])

qb_post['week'] = qb_post['week'] + 1

qb = pd.merge(qb_pre, qb_post, how='inner', 
              left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])
# -

# # RB

# ## Pre-Game

# +
rb_pre = pd.read_sql_query('''
    SELECT * FROM FantasyPros_Rankings a
    INNER JOIN (SELECT * FROM Daily_salaries) b ON a.player = b.player and a.week = b.week and a.year = b.year
    LEFT JOIN (SELECT * FROM NumberFire_Rankings) c ON a.player = c.player and a.week = c.week and a.year = c.year
    INNER JOIN (SELECT * FROM RB_FantasyPros_Matchups) d ON a.player = d.player and d.week = b.week and a.year = d.year
    INNER JOIN (SELECT * FROM RB_PRF_Matchups) e ON a.player = e.player and d.week = e.week and a.year = e.year
    INNER JOIN (SELECT * FROM Roto_Grinders_Projections) f ON a.player = f.player and d.week = f.week and a.year = f.year
    WHERE a.position = 'RB'
''', conn)

# remove duplicated columns
rb_pre = rb_pre.loc[:, ~rb_pre.columns.duplicated()]
# -

# ## Post-Game

# +
rb_post = pd.read_sql_query('''
    SELECT * FROM RB_Basic_Stat_Results a
    LEFT JOIN (SELECT * FROM Air_Yards) b ON a.player = b.player and a.week = b.week and a.year = b.year
    LEFT JOIN (SELECT * FROM PRF_Rec_Stats) c ON a.player = c.player and a.week = c.week and a.year = c.year
    INNER JOIN (SELECT * FROM PFR_RB_Stats) d ON a.player = d.player and a.week = d.week and a.year = d.year
    LEFT JOIN (SELECT * FROM PFR_Redzone_Rec) e ON a.player = e.player and a.week = e.week and a.year = e.year
    LEFT JOIN (SELECT * FROM Snap_Counts) f ON a.player = f.player and a.week = f.week and a.year = f.year

''', conn)

# remove duplicated columns
rb_post = rb_post.loc[:, ~rb_post.columns.duplicated()]

# +
rz_tar = clean_rz(pd.read_sql_query('SELECT * FROM RedZone_Targets', conn))
rb_post = pd.merge(rb_post, rz_tar, how='left', 
                   left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])
rb_post = rb_post.fillna(0)

df1 = pd.read_sql_query('''SELECT * FROM RB_Basic_Stat_Results''', conn)
df2 = pd.read_sql_query('''SELECT * FROM DVOA_RB''', conn)
df3 = pd.read_sql_query('''SELECT * FROM DVOA_RB_Rec''', conn)
dvoa_rb = find_first_name(df1, df2)
dvoa_rb_rec = find_first_name(df1, df3)

rb_post = pd.merge(rb_post, dvoa_rb, how='left', 
                   left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])
rb_post = pd.merge(rb_post, dvoa_rb_rec, how='left', 
                   left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])


rb_post = rb_post.dropna(thresh=int(0.5*rb_post.shape[0]), axis=1)
nulls = rb_post.isnull().sum()[rb_post.isnull().sum() > 0].index
for n in nulls:
    rb_post[n] = rb_post.groupby('player')[n].apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

rb_post = rb_post.fillna(0)
# -

rb_post['week'] = rb_post['week'] + 1
rb = pd.merge(rb_pre, rb_post, how='inner', 
              left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])

# # WR

# ## Pre-Game

# +
wr_pre = pd.read_sql_query('''
    SELECT * FROM FantasyPros_Rankings a
    INNER JOIN (SELECT * FROM Daily_salaries) b ON a.player = b.player and a.week = b.week and a.year = b.year
    LEFT JOIN (SELECT * FROM NumberFire_Rankings) c ON a.player = c.player and a.week = c.week and a.year = c.year
    INNER JOIN (SELECT * FROM WR_FantasyPros_Matchups) d ON a.player = d.player and d.week = b.week and a.year = d.year
    INNER JOIN (SELECT * FROM WR_PRF_Matchups) e ON a.player = e.player and d.week = e.week and a.year = e.year
    INNER JOIN (SELECT * FROM Roto_Grinders_Projections) f ON a.player = f.player and d.week = f.week and a.year = f.year
    INNER JOIN (SELECT * FROM PFF_WR_CB_Matchups) g ON a.player = g.player and d.week = g.week and a.year = g.year
    WHERE a.position = 'WR'
''', conn)

# remove duplicated columns
wr_pre = wr_pre.loc[:, ~wr_pre.columns.duplicated()]

wr_pre = wr_pre.dropna(thresh=int(0.5*wr_pre.shape[0]), axis=1)
nulls = wr_pre.isnull().sum()[wr_pre.isnull().sum() > 0].index
for n in nulls:
    wr_pre[n] = wr_pre.groupby('player')[n].apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
# -

# ## Post-Game

# +
wr_post = pd.read_sql_query('''
    SELECT * FROM WR_Basic_Stat_Results a
    INNER JOIN (SELECT * FROM Air_Yards) b ON a.player = b.player and a.week = b.week and a.year = b.year
    LEFT JOIN (SELECT * FROM PRF_Rec_Stats) c ON a.player = c.player and a.week = c.week and a.year = c.year
    LEFT JOIN (SELECT * FROM PFR_Redzone_Rec) e ON a.player = e.player and a.week = e.week and a.year = e.year
    LEFT JOIN (SELECT * FROM Snap_Counts) f ON a.player = f.player and a.week = f.week and a.year = f.year
''', conn)

# remove duplicated columns
wr_post = wr_post.loc[:, ~wr_post.columns.duplicated()]
# -

wr_post = pd.merge(wr_post, rz_tar, how='left', 
                   left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])
wr_post = wr_post.fillna(0)

# +
df1 = pd.read_sql_query('''SELECT * FROM WR_Basic_Stat_Results''', conn)
df2 = pd.read_sql_query('''SELECT * FROM DVOA_WR''', conn)
dvoa_wr = find_first_name(df1, df2)

wr_post = pd.merge(wr_post, dvoa_wr, how='inner', 
                   left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])
# -

# ## Clean-up

wr_post['week'] = wr_post['week'] + 1
wr = pd.merge(wr_pre, wr_post, how='inner', 
              left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])

# # TE

# ## Pre-Game

# +
te_pre = pd.read_sql_query('''
    SELECT * FROM FantasyPros_Rankings a
    INNER JOIN (SELECT * FROM Daily_salaries) b ON a.player = b.player and a.week = b.week and a.year = b.year
    INNER JOIN (SELECT * FROM NumberFire_Rankings) c ON a.player = c.player and a.week = c.week and a.year = c.year
    INNER JOIN (SELECT * FROM TE_FantasyPros_Matchups) d ON a.player = d.player and d.week = b.week and a.year = d.year
    INNER JOIN (SELECT * FROM TE_PRF_Matchups) e ON a.player = e.player and d.week = e.week and a.year = e.year
    INNER JOIN (SELECT * FROM Roto_Grinders_Projections) f ON a.player = f.player and d.week = f.week and a.year = f.year
    INNER JOIN (SELECT * FROM PFF_TE_Matchups) g ON a.player = g.player and d.week = g.week and a.year = g.year
    WHERE a.position = 'TE'
''', conn)

# remove duplicated columns
te_pre = te_pre.loc[:, ~te_pre.columns.duplicated()]

nulls = te_pre.isnull().sum()[te_pre.isnull().sum() > 0].index
for n in nulls:
    te_pre[n] = te_pre.groupby('player')[n].apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
# -

# ## Post-Game

# +
te_post = pd.read_sql_query('''
    SELECT * FROM TE_Basic_Stat_Results a
    INNER JOIN (SELECT * FROM Air_Yards) b ON a.player = b.player and a.week = b.week and a.year = b.year
    LEFT JOIN (SELECT * FROM PRF_Rec_Stats) c ON a.player = c.player and a.week = c.week and a.year = c.year
    LEFT JOIN (SELECT * FROM PFR_Redzone_Rec) e ON a.player = e.player and a.week = e.week and a.year = e.year
    LEFT JOIN (SELECT * FROM Snap_Counts) f ON a.player = f.player and a.week = f.week and a.year = f.year
''', conn)

# remove duplicated columns
te_post = te_post.loc[:, ~te_post.columns.duplicated()]

# +
te_post = pd.merge(te_post, rz_tar, how='left', 
                   left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])

te_post = te_post.fillna(0)

# +
df1 = pd.read_sql_query('''SELECT * FROM TE_Basic_Stat_Results''', conn)
df2 = pd.read_sql_query('''SELECT * FROM DVOA_TE''', conn)
dvoa_te = find_first_name(df1, df2)

te_post = pd.merge(te_post, dvoa_te, how='inner', 
                   left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])
# -

# ## Clean-up

te_post['week'] = te_post['week'] + 1
te = pd.merge(te_pre, te_post, how='inner', 
              left_on=['player', 'week', 'year'], right_on=['player', 'week', 'year'])
