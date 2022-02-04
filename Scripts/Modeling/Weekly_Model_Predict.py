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
set_week = 18

# set the earliest date to begin the validation set
val_year_min = 2020
val_week_min = 10

met = 'y_act'

# full_model or backfill
model_type = 'backfill'
vers = 'standard'

if model_type == 'full_model': positions = ['QB', 'RB', 'WR', 'TE',  'Defense']
elif model_type == 'backfill': positions =  ['QB']#, 'RB', 'WR', 'TE']

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

    def load_data(model_type, set_pos):

        # load data and filter down
        if model_type=='full_model': df = dm.read(f'''SELECT * FROM {set_pos}_Data''', 'Model_Features')
        elif model_type=='backfill': df = dm.read(f'''SELECT * FROM Backfill WHERE pos='{set_pos}' ''', 'Model_Features')

        if df.shape[1]==2000:
            df2 = dm.read(f'''SELECT * FROM {set_pos}_Data2''', 'Model_Features')
            df = pd.concat([df, df2], axis=1)

        drop_cols = list(df.dtypes[df.dtypes=='object'].index)
        print(drop_cols)

        return df, drop_cols

    def year_week_to_date(x):
        return int(dt.datetime(x[0], 1, x[1]).strftime('%Y%m%d'))

    def create_game_date(df):
        
        # set up the date column for sorting
        df['game_date'] = df[['year', 'week']].apply(year_week_to_date, axis=1)
        cv_time_input = int(dt.datetime(val_year_min, 1, val_week_min).strftime('%Y%m%d'))
        train_time_split = int(dt.datetime(set_year, 1, set_week).strftime('%Y%m%d'))

        return df, cv_time_input, train_time_split

    
    def train_predict_split(df, train_time_split, cv_time_input):

        # # get the train / predict dataframes and output dataframe
        df_train = df[df.game_date < train_time_split].reset_index(drop=True)
        df_train = df_train.dropna(subset=['y_act']).reset_index(drop=True)

        df_predict = df[df.game_date == train_time_split].reset_index(drop=True)
        output_start = df_predict[['player', 'dk_salary', 'fantasyPoints', 'projected_points', 'ProjPts']].copy()

        # get the minimum number of training samples for the initial datasets
        min_samples = int(df_train[df_train.game_date < cv_time_input].shape[0])  
        print('Shape of Train Set', df_train.shape)

        return df_train, df_predict, output_start, min_samples


    def get_class_predictions(df, models_class):

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

        return X_stack_class, X_predict_class


    def optimize_reg_model(final_m, X_stack, y_stack):
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

        return best_model, stack_score, adp_score


    #==========
    # Pull and clean compiled data
    #==========

    
    df, drop_cols = load_data(model_type, set_pos)
    df, cv_time_input, train_time_split = create_game_date(df)
    df_train, df_predict, output_start, min_samples = train_predict_split(df, train_time_split, cv_time_input)

    #------------
    # Make the Class Predictions
    #------------

    pred_class = load_pickle(model_output_path, 'class_pred')
    actual_class = load_pickle(model_output_path, 'class_actual')
    models_class = load_pickle(model_output_path, 'class_models')
    scores_class = load_pickle(model_output_path, 'class_scores')

    X_stack_class, X_predict_class = get_class_predictions(df, models_class)

    pred = load_pickle(model_output_path, 'reg_pred')
    actual = load_pickle(model_output_path, 'reg_actual')
    models = load_pickle(model_output_path, 'reg_models')
    scores = load_pickle(model_output_path, 'reg_scores')
#%%
    # get the X and y values for stack training for the current metric
    skm_stack = SciKitModel(df_train)
    X_stack, y_stack = skm_stack.X_y_stack(met, pred, actual)
    X_stack = pd.concat([X_stack, X_stack_class], axis=1)

    # df_predict_stack = df_predict.copy()
    # df_predict_stack = df_predict_stack.drop('y_act', axis=1).fillna(0)

    best_models = []
    final_models = ['ridge',
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
        best_model, stack_score, adp_score = optimize_reg_model(final_m, X_stack, y_stack)
        
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

drop_teams = ['GB', 'CLE', 'MIN', 'PIT']

teams = dm.read(f'''SELECT * FROM Player_Teams''', 'Simulation')
preds = pd.merge(preds, teams, on=['player'])
preds = preds[~preds.team.isin(drop_teams)].drop('team', axis=1).reset_index(drop=True)

preds.sort_values(by='pred_fp_per_game', ascending=False).iloc[:50]


#%%
df_train_class, df_predict_class = get_class_data(df, cuts[0])
skm_class_final = SciKitModel(df_train_class, model_obj='class')
X_class_final, y_class_final = skm_class_final.Xy_split(y_metric='y_act', to_drop=drop_cols)

cv_time = skm_class_final.cv_time_splits('game_date', X_class_final, cv_time_input)
predictions = skm_class_final.cv_predict_time(models['y_act_lgbm'][0], X_class_final, y_class_final, cv_time)
# %%
