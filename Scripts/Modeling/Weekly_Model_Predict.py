#%%
# core packages
from random import Random
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import datetime as dt
import matplotlib.pyplot as plt
from ff.db_operations import DataManage
from ff import general as ffgeneral
import ff.data_clean as dc
from skmodel import SciKitModel
from Fix_Standard_Dev import *

import pandas_bokeh
pandas_bokeh.output_notebook()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

pd.set_option('display.max_columns', 999)

from sklearn import set_config
set_config(display='diagram')

splines = {}
for k, p in zip([1, 2, 2, 2, 2], ['QB', 'RB', 'WR', 'TE', 'Defense']):
    print(f'Checking Splines for {p}')
    spl_sd, spl_perc = get_std_splines(p, show_plot=True, k=k)
    splines[p] = [spl_sd, spl_perc]

#%%
#==========
# General Setting
#==========

# set the root path and database management object
root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)
np.random.seed(1234)

# set year to analyze
set_year = 2021
set_week = 16

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 10

met = 'y_act'

# full_model or backfill
model_type = 'backfill'
vers = 'standard'

if model_type == 'full_model': positions =  ['QB', 'RB', 'WR', 'TE',  'Defense']
elif model_type == 'backfill': positions = ['QB', 'RB', 'WR', 'TE']

for set_pos in positions:

    def_cuts = [33, 75, 90]
    off_cuts = [33, 80, 95]
    if set_pos == 'Defense': cuts = def_cuts
    else: cuts = off_cuts

    #-------------
    # Set up output dataset
    #-------------

    all_vars = [set_pos, set_year, set_week]

    pkey = f'{set_pos}_year{set_year}_week{set_week}_{model_type}{vers}'
    db_output = {'set_pos': set_pos, 'set_year': set_year, 'set_week': set_week}
    db_output['pkey'] = pkey

    model_output_path = f'{root_path}/Model_Outputs/{set_year}/{pkey}/'
    if not os.path.exists(model_output_path): os.makedirs(model_output_path)

    def load_pickle(path, fname):
        with gzip.open(f"{path}/{fname}.p", 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object

    def get_class_data(df, cut):

        # set up the training and prediction datasets for the classification 
        df_train_class = df[df.game_date < train_time_split].reset_index(drop=True)
        df_predict_class = df[df.game_date == train_time_split].reset_index(drop=True)

        # set up the target variable to be categorical based on Xth percentile
        cut_perc = df_train_class.groupby('game_date')['y_act'].apply(lambda x: np.percentile(x, cut))
        df_train_class = pd.merge(df_train_class, cut_perc.reset_index().rename(columns={'y_act': 'cut_perc'}), on=['game_date'])
        
        if cut > 33:
            df_train_class['y_act'] = np.where(df_train_class.y_act >= df_train_class.cut_perc, 1, 0)
        else: 
            df_train_class['y_act'] = np.where(df_train_class.y_act <= df_train_class.cut_perc, 1, 0)

        df_train_class = df_train_class.drop('cut_perc', axis=1)

        return df_train_class, df_predict_class

    #==========
    # Pull and clean compiled data
    #==========

    # load data and filter down
    if model_type=='full_model': df = dm.read(f'''SELECT * FROM {set_pos}_Data''', 'Model_Features')
    elif model_type=='backfill': df = dm.read(f'''SELECT * FROM Backfill WHERE pos='{set_pos}' ''', 'Model_Features')

    if df.shape[1]==2000:
        df2 = dm.read(f'''SELECT * FROM {set_pos}_Data2''', 'Model_Features')
        df = pd.concat([df, df2], axis=1)

    drop_cols = list(df.dtypes[df.dtypes=='object'].index)
    print(drop_cols)

    # set up the date column for sorting
    def year_week_to_date(x):
        return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))

    df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
    cv_time_input = int(dt.datetime(val_year_min, 1, val_week_min).strftime('%Y%m%d'))
    train_time_split = int(dt.datetime(set_year, 1, set_week).strftime('%Y%m%d'))

    # # get the train / predict dataframes and output dataframe
    df_train = df[df.game_date < train_time_split].reset_index(drop=True)
    df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)

    #  # test log    
    # df_train = df_train[df_train.y_act > 2].reset_index(drop=True)


    df_predict = df[df.game_date == train_time_split].reset_index(drop=True)
    output_start = df_predict[['player', 'dk_salary', 'fantasyPoints', 'projected_points', 'ProjPts']].copy()

    # get the minimum number of training samples for the initial datasets
    min_samples = int(df_train[df_train.game_date < cv_time_input].shape[0])  
    print('Shape of Train Set', df_train.shape)
    #------------
    # Make the Class Predictions
    #------------

    pred_class = load_pickle(model_output_path, 'class_pred')
    actual_class = load_pickle(model_output_path, 'class_actual')
    models_class = load_pickle(model_output_path, 'class_models')
    scores_class = load_pickle(model_output_path, 'class_scores')

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict_class = pd.DataFrame()
    for cut in cuts:

        print(f"\n--------------\nPercentile {cut}\n--------------\n")

        df_train_class, df_predict_class = get_class_data(df, cut)

        skm_class_final = SciKitModel(df_train_class, model_obj='class')
        X_stack_class, y_stack_class = skm_class_final.X_y_stack('class', pred_class, actual_class)
        X_class_final, y_class_final = skm_class_final.Xy_split(y_metric='y_act', to_drop=drop_cols)
        
        for k, v in models_class.items():
            if str(cut) in k:
                m = skm_class_final.ensemble_pipe(v)
                m.fit(X_class_final, y_class_final)
                cur_pred = pd.Series(m.predict_proba(df_predict_class[X_class_final.columns])[:,1], name=k)
                X_predict_class = pd.concat([X_predict_class, cur_pred], axis=1)

    pred = load_pickle(model_output_path, 'reg_pred')
    actual = load_pickle(model_output_path, 'reg_actual')
    models = load_pickle(model_output_path, 'reg_models')
    scores = load_pickle(model_output_path, 'reg_scores')

    #------------
    # Make the Regression Predictions
    #------------

    df_predict_stack = df_predict.copy()
    df_predict_stack = df_predict_stack.drop('y_act', axis=1).fillna(0)
    skm_stack = SciKitModel(df_train)

    # get the X and y values for stack training for the current metric
    X_stack, y_stack = skm_stack.X_y_stack(met, pred, actual)
    X_stack = pd.concat([X_stack, X_stack_class], axis=1)


    # # test log
    # y_stack = np.log1p(y_stack)


    best_models = []
    final_models = [
                    'ridge',
                    'lasso',
                    'bridge',
                    'lgbm', 
                    'xgb', 
                    'rf', 
                    # 'gbm' ,
                    # 'knn'
                    ]
    for final_m in final_models:

        print(f'\n{final_m}')

        # get the model pipe for stacking setup and train it on meta features
        stack_pipe = skm_stack.model_pipe([
                                skm_stack.feature_union([
                                                skm_stack.piece('agglomeration'), 
                                                skm_stack.piece('k_best'),
                                                skm_stack.piece('pca')
                                                ]),
                                skm_stack.piece('k_best'), 
                                skm_stack.piece(final_m)
                            ])
        stack_params = skm_stack.default_params(stack_pipe)
        stack_params['k_best__k'] = range(1, X_stack.shape[1])

        best_model, stack_score, adp_score = skm_stack.best_stack(stack_pipe, stack_params,
                                                                X_stack, y_stack, n_iter=50, 
                                                                run_adp=True, print_coef=False)
        best_models.append(best_model)

    # get the final output:
    X_fp, y_fp = skm_stack.Xy_split(y_metric='y_act', to_drop=drop_cols)

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, v in models.items():
        m = skm_stack.ensemble_pipe(v)
        m.fit(X_fp, y_fp)
        X_predict = pd.concat([X_predict, pd.Series(m.predict(df_predict[X_fp.columns]), name=k)], axis=1)

    X_predict = pd.concat([X_predict, X_predict_class], axis=1)

    predictions = pd.DataFrame()
    val_predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):
        val_predict = pd.Series(skm_stack.cv_predict(bm, X_fp, y_fp), name=f'pred_{met}_{fm}')
        val_predictions = pd.concat([val_predictions, val_predict], axis=1)
        prediction = pd.Series(np.round(bm.predict(X_predict), 2), name=f'pred_{met}_{fm}')
        predictions = pd.concat([predictions, prediction], axis=1)

    #------------
    # Scatter and Metrics for Overall Results
    #------------
    plt.scatter(val_predictions.mean(axis=1), y_fp)
    plt.xlabel('predictions');plt.ylabel('actual')
    plt.show()
    from sklearn.metrics import r2_score
    print('Total R2:', r2_score(y_fp, val_predictions.mean(axis=1)))

    val_high_score = pd.concat([val_predictions.mean(axis=1), y_fp], axis=1)
    val_high_score.columns = ['predictions','y_act']
    val_high_score = val_high_score[val_high_score.predictions >= \
                                    np.percentile(val_high_score.predictions, 75)]

    val_high_score.plot.scatter(x='predictions', y='y_act')
    plt.show()
    print('High Score R2:', r2_score(val_high_score.y_act, val_high_score.predictions))

    #===================
    # Create Outputs
    #===================

    output = output_start[['player', 'dk_salary', 'fantasyPoints', 'ProjPts', 'projected_points']].copy()

    # output = pd.concat([output, predictions], axis=1)
    output['pred_fp_per_game'] = predictions.mean(axis=1)
    _, recent = rolling_max_std(set_pos)
    output = pd.merge(output, recent, on='player', how='left')
    output.roll_std = output.roll_std.fillna(recent.roll_std.median())
    output.roll_max = output.roll_max.fillna(recent.roll_std.median())

    output = create_sd_max_metrics(output)

    output['std_dev'] = splines[set_pos][0](output.sd_metric)
    output['max_score'] = splines[set_pos][1](output.max_metric)

    output = output.sort_values(by='dk_salary', ascending=False)
    output['dk_rank'] = range(len(output))
    output = output.sort_values(by='pred_fp_per_game', ascending=False).reset_index(drop=True)
    print(output.iloc[:50])

    output['pos'] = set_pos
    output['version'] = vers
    output['model_type'] = model_type
    output['min_score'] = df_train.y_act.min()
    output['week'] = set_week
    output['year'] = set_year

    output = output[['player', 'dk_salary', 'sd_metric', 'max_metric', 'pred_fp_per_game', 'std_dev',
                     'dk_rank', 'pos', 'version', 'model_type', 'max_score', 'min_score',
                     'week', 'year']]

    del_str = f'''pos='{set_pos}' 
                AND version='{vers}' 
                AND week={set_week} 
                AND year={set_year}
                AND model_type='{model_type}'
                '''
    dm.delete_from_db('Simulation', 'Model_Predictions', del_str)
    dm.write_to_db(output, 'Simulation', f'Model_Predictions', 'append')

# %%

preds = dm.read(f'''SELECT * 
                    FROM Model_Predictions 
                    WHERE version='{vers}'
                          AND week = '{set_week}'
                          AND year = '{set_year}' 
                          AND player != 'Ryan Griffin'
            ''', 'Simulation')

preds['weighting'] = 1
preds.loc[preds.model_type=='full_model', 'weighting'] = 1

score_cols = ['pred_fp_per_game', 'std_dev', 'max_score']
for c in score_cols: preds[c] = preds[c] * preds.weighting

# Groupby and aggregate with namedAgg [1]:
preds = preds.groupby(['player', 'pos'], as_index=False).agg({'pred_fp_per_game': 'sum', 
                                                              'std_dev': 'sum',
                                                              'max_score': 'sum',
                                                              'min_score': 'sum',
                                                              'weighting': 'sum'})



for c in score_cols: preds[c] = preds[c] / preds.weighting
preds = preds.drop('weighting', axis=1)

drop_teams = ['SF', 'TEN', 'ARI', 'GB','CLE', 'IND', 'WAS', 'DAL', 'MIA', 'NO']

teams = dm.read(f'''SELECT * FROM Player_Teams''', 'Simulation')
preds = pd.merge(preds, teams, on=['player'])
preds = preds[~preds.team.isin(drop_teams)].drop('team', axis=1).reset_index(drop=True)

preds.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50]


#%%

def create_distribution(player_data, num_samples=1000):

    import scipy.stats as stats

    # create truncated distribution
    lower, upper = player_data.min_score,  player_data.max_score
    lower_bound = (lower - player_data.pred_fp_per_game) / player_data.std_dev, 
    upper_bound = (upper - player_data.pred_fp_per_game) / player_data.std_dev
    trunc_dist = stats.truncnorm(lower_bound, upper_bound, loc= player_data.pred_fp_per_game, scale= player_data.std_dev)
    
    estimates = trunc_dist.rvs(num_samples)

    return estimates


def create_sim_output(output, num_samples=1000):
    sim_out = pd.DataFrame()
    for _, row in output.iterrows():
        cur_out = pd.DataFrame([row.player, row.pos]).T
        cur_out.columns=['player', 'pos']
        dists = pd.DataFrame(create_distribution(row, num_samples)).T
        cur_out = pd.concat([cur_out, dists], axis=1)
        sim_out = pd.concat([sim_out, cur_out], axis=0)
    
    return sim_out


def plot_distribution(estimates, exponent=False):

    from IPython.core.pylabtools import figsize
    import seaborn as sns
    import matplotlib.pyplot as plt

    print('\n', estimates.player)
    estimates = estimates.iloc[2:]

    if exponent:
        estimates = estimates.apply(lambda x: np.exp(x))

    # Plot all the estimates
    plt.figure(figsize(8, 8))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws = {'linewidth' : 4},
                 label = 'Estimated Dist.')

    # Plot the mean estimate
    plt.vlines(x = estimates.mean(), ymin = 0, ymax = 0.01, 
                linestyles = '--', colors = 'red',
                label = 'Pred Estimate',
                linewidth = 2.5)

    plt.legend(loc = 1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel('Grade'); plt.ylabel('Density');

    # Prediction information
    sum_stats = (np.percentile(estimates, 5), np.percentile(estimates, 95), estimates.std() /estimates.mean())
    print('Average Estimate = %0.4f' % estimates.mean())
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f    Std Error = %0.4f' % sum_stats) 

sim_dist = create_sim_output(preds).reset_index(drop=True)

#%%

idx = sim_dist[sim_dist.player=="Davante Adams"].index[0]
plot_distribution(sim_dist.iloc[idx])
# plot_distribution(sim_dist.iloc[idx], exponent=True)
# %%

dm.write_to_db(sim_dist, 'Simulation', f'week{set_week}_year{set_year}', 'replace')


# %%

for i in range(12):

    players = dm.read('''SELECT * FROM Best_Lineups WHERE week=7''', 'Results').iloc[i, :9].values

    cur_team = preds[preds.player.isin(players)].copy()

    cur_team['variance'] = cur_team.std_dev ** 2
    sum_variance = np.sum(cur_team.variance)
    sum_mean_var = np.var(cur_team.pred_fp_per_game)

    team_var = np.sqrt(sum_variance + sum_mean_var)
    team_mean = cur_team.pred_fp_per_game.sum()

    import seaborn as sns
    estimates = np.random.normal(team_mean, team_var, 10000)
    
    print(i, team_mean, team_var, np.percentile(estimates, 80), np.percentile(estimates, 99))
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws = {'linewidth' : 4},
                 label = 'Estimated Dist.')
    
# %%

players = dm.read('''SELECT * FROM Best_Lineups''', 'Results')
players = players.loc[1:, ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'week', 'year']]
players['lineup_num'] = range(len(players))
best_lineups = pd.melt(players, id_vars=['week', 'year', 'lineup_num'])
best_lineups = best_lineups.rename(columns={'value': 'player'}).drop('variable', axis=1)

y_act = dm.read(f'''SELECT defTeam player, week, season year, fantasy_pts y_act
                    FROM Defense_Stats
                    WHERE season >= 2020''', 'FastR')

for p in ['QB', 'RB', 'WR', 'TE']:
    y_act_cur = dm.read(f'''SELECT player, week, season year, fantasy_pts y_act
                            FROM {p}_Stats
                            WHERE season >= 2020''', 'FastR')
    y_act = pd.concat([y_act, y_act_cur], axis=0)

best_lineups = pd.merge(best_lineups, y_act, on=['player', 'week', 'year'])
team_scores = best_lineups.groupby('lineup_num').agg({'y_act': 'sum'}).y_act
for i in range(50, 100, 5):
    print(f'Percentile {i}:', np.percentile(team_scores, i))

print(f'Percentile 99:', np.percentile(team_scores, 99))
sns.distplot(team_scores, hist = True, kde = True, bins = 15,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws = {'linewidth' : 4},
                 label = 'Estimated Dist.')


# %%


df = dm.read('''SELECT * FROM Model_Predictions''', 'Simulation')


 # pull in the salary and actual results data
dk_sal = dm.read('''SELECT team player, team, week, year, projected_points sd_metric
                    FROM FantasyPros
                    WHERE pos='DST' ''', 'Pre_PlayerData')

stats = dm.read(f'''SELECT defTeam player, defTeam team, week, season year, fantasy_pts y_act
                    FROM {pos}_Stats ''', 'FastR')

# pull in the salary and actual results data
dk_sal = dm.read('''SELECT player, offTeam team, week, year, projected_points, fantasyPoints, ProjPts
                    FROM PFF_Proj_Ranks
                    JOIN (SELECT player, team offTeam, week, year, projected_points 
                            FROM FantasyPros)
                            USING (player, offTeam, week, year)
                    JOIN (SELECT player, offTeam, week, year, `Proj Pts` ProjPts 
                            FROM PFF_Expert_Ranks)
                            USING (player, offTeam, week, year)''', 'Pre_PlayerData')
# %%
