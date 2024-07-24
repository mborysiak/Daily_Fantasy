#%%
from ff.db_operations import DataManage
from ff import general as ffgeneral
import pandas as pd
import numpy as np
import os
import pickle
import datetime as dt
import gzip
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

from warnings import simplefilter 
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)



def load_pickle(path, fname):
        with gzip.open(f"{path}/{fname}.p", 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object
        
def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

def save_pickle(obj, path, fname, protocol=-1):
    with gzip.open(f"{path}/{fname}.p", 'wb') as f:
        pickle.dump(obj, f, protocol)

    print(f'Saved {fname} to path {path}')

class FoldPredict:

    def __init__(self, save_path, retrain=True):
        self.save_path = save_path
        self.retrain = retrain

    def cross_fold_train(self, model_type, model, params, X, y, n_iter=10):

        for i, (train_idx, test_idx) in enumerate(KFold(n_splits=4, shuffle=True).split(X)):
            print(f'Fold {i+1}')
            X_train, _ = X.iloc[train_idx], X.iloc[test_idx]
            y_train, _ = y.iloc[train_idx], y.iloc[test_idx]

            grid = RandomizedSearchCV(model, params, n_iter=n_iter, scoring='neg_mean_squared_error', n_jobs=2, cv=4)
            grid.fit(X_train,y_train)
            
            scores = pd.concat([pd.DataFrame(grid.cv_results_['params']), 
                                pd.DataFrame(np.sqrt(-grid.cv_results_['mean_test_score']))], axis=1).sort_values(by=0, ascending=False)
            print(scores)

            best_model = grid.best_estimator_
            print(best_model)
    
            if not os.path.exists(self.save_path): os.makedirs(self.save_path)
            save_pickle(best_model, self.save_path, f'{model_type}_fold{i}')

    def cross_fold_predict(self, model_type, X, y):

        predictions = pd.DataFrame()
        for _, (train_idx, test_idx) in enumerate(KFold(n_splits=4, shuffle=True).split(X)):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            cur_predict = pd.DataFrame()
            for i in range(4):
                model = load_pickle(self.save_path, f'{model_type}_fold{i}')
                if self.retrain: model.fit(X_train, y_train)
                model_i_predict = pd.DataFrame(model.predict(X_test), index=test_idx, columns=[f'score_{i}'])
                cur_predict = pd.concat([cur_predict, model_i_predict], axis=1)
            
            cur_predict = pd.DataFrame(cur_predict.mean(axis=1), columns=[f'{model_type}_pred'])
            print('MAE:', np.round(np.mean(np.abs(cur_predict[f'{model_type}_pred'] - y_test)), 5))
            cur_predict = pd.concat([cur_predict, y_test], axis=1)
            predictions = pd.concat([predictions, cur_predict], axis=0)
                
        predictions = pd.merge(predictions, X,left_index=True, right_index=True)

        return predictions
    



#%%

best_trials = dm.read('''SELECT *
                         FROM Entry_Optimize_Results
                         WHERE trial_num >= 460
                         ''', 'Results')

best_trials.loc[best_trials.avg_winnings > 10000, 'avg_winnings'] = 10000

best_trials['non8_winnings'] = best_trials.avg_winnings
best_trials.loc[(best_trials.week == 8) & (best_trials.year==2022), 'non8_winnings'] = 0 

best_trials['avg_winnings_sqrt_pre'] = best_trials.avg_winnings ** 0.5

best_trials = (
    best_trials
    .groupby(['pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers',
              'model_notes', 'manual_adjust', 'trial_num', 'week', 'year'])
    .agg({'avg_winnings': 'mean',
          'avg_winnings_sqrt_pre': 'mean',
          'non8_winnings': 'mean',
          #'perc80': lambda x: np.percentile(x, 80)
          })
    .reset_index()
)

best_trials['avg_winnings_sqrt_post'] = best_trials.avg_winnings ** 0.5
best_trials = (
    best_trials
    .groupby(['pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers', 'model_notes', 'manual_adjust', 'trial_num'])
    .agg({'avg_winnings': 'sum',
          'avg_winnings_sqrt_pre': 'sum',
          'avg_winnings_sqrt_post': 'sum',
          'non8_winnings': 'sum'})
)

winnings_cols = ['avg_winnings', 'avg_winnings_sqrt_pre', 'avg_winnings_sqrt_post', 'non8_winnings']
for c in winnings_cols:
    best_trials[c+'_rank'] = best_trials[c].rank(ascending=False)

winnings_cols = [c+'_rank' for c in winnings_cols]
best_trials['avg_rank'] = best_trials[winnings_cols].mean(axis=1)
best_trials = best_trials.sort_values(by='avg_rank').reset_index()
# best_trials['avg_winnings'] = best_trials.avg_winnings ** 2

best_trials.iloc[:25]

#%%

params = dm.read('''SELECT *
                    FROM Entry_Optimize_Params
                    WHERE trial_num >= 460
                    ''', 'Results')
param_vars = list(params.param.unique())
params['param'] = params.param.astype('str') + '_' + params.param_option.astype('str')

params = params.pivot_table(index=['trial_num'], columns='param', values='option_value').reset_index().fillna(0)

results = dm.read('''SELECT *
                    FROM Entry_Optimize_Results
                    WHERE trial_num >= 460
                         -- AND NOT (week=8 AND year=2022)
                          AND reg_ens_vers = 'random_full_stack_sera0_rsq0_mse1_include2_kfold3'
                          AND million_ens_vers IN (
                                                   'random_full_stack_team_stats_matt0_brier1_include2_kfold3',
                                                   'random_kbest_team_stats_matt0_brier1_include2_kfold3',
                                                   'random_kbest_matt0_brier1_include2_kfold3'
                          )
                          AND std_dev_type iN (
                                                    'spline_class80_q80_matt0_brier1_kfold3',
                                                    'spline_pred_class80_q80_matt0_brier1_kfold3',
                                                    'spline_pred_class80_matt0_brier1_kfold3'
                          )
                    ''', 'Results')

results.loc[results.avg_winnings > 10000, 'avg_winnings'] = 10000
results.loc[~((results.week==8)&(results.year==2022)), 'avg_winnings'] = results.loc[~((results.week==8)&(results.year==2022)), 'avg_winnings'] * 5

results = results.groupby(['pred_vers', 'reg_ens_vers', 'std_dev_type', 'million_ens_vers', 
                           'ownership_vers', 'entry_type', 'trial_num', 'repeat_num']).agg({'avg_winnings': 'sum'}).reset_index()

results = pd.merge(results, params, on='trial_num')
results = results.drop(['trial_num', 'repeat_num'], axis=1)
for c in results.columns:
    if results.dtypes[c] == 'object': 
        results[c] = results[c].astype('category')

# results.week = results.week.astype('category')
# results.year = results.year.astype('category')

X = results.drop('avg_winnings', axis=1)
y = results.avg_winnings

#%%


params = {
    'n_estimators': range(100, 400, 10),
    'num_leaves': range(20, 300, 10),
    'min_child_samples': range(10, 70, 2),
    'learning_rate': np.arange(0.01, 0.35, 0.02),
    'subsample': np.arange(0.8, 1, 0.02)
}

retrain = False
model = LGBMRegressor(n_jobs=16)
fp = FoldPredict(f'{root_path}/Model_Outputs/Final_LGBM/', retrain=retrain)
fp.cross_fold_train('winnings', model, params, X, y, n_iter=20)

#%%

model = load_pickle(fp.save_path, 'winnings_fold1')
import shap
shap_values = shap.TreeExplainer(model).shap_values(X)
shap.summary_plot(shap_values, X, feature_names=X.columns, plot_size=(8,10), max_display=30, show=False)

#%%
more_opt_columns = [ 'qb_min_iter', 'matchup_seed', 'max_team_type', 'matchup_drop', 'qb_set_max_team', 'ownership_vers', 'use_ownership']
X[[c for c in X.columns if 'max_team_type' in c]].value_counts()

#%%

shap.dependence_plot("ownership_vers_standard_ln", shap_values, X)




#%%

extra_cutoff_val = -20

counts_base = 0
while counts_base < 7 * 1000000:
    for cnt_cutoff in range(-200, -30, 10):
        cnt_cutoff = abs(cnt_cutoff)

        drop_cols = []
        for pv in param_vars:
            for c in X.columns:
                if pv in c:
                    drop_cols.append(c)
        base_cols = [c for c in X.columns if c not in drop_cols]
        counts_base = len(X[base_cols].drop_duplicates())

        for pv in param_vars:
            cur_cols = [c for c in X.columns if pv in c]

            for moc in more_opt_columns:
                if moc in cur_cols[0]: 
                    extra_cutoff = extra_cutoff_val
                    break
                else: 
                    extra_cutoff = 0

            if X[cur_cols].value_counts().max() < cnt_cutoff:
                cur_cnt_cutoff = X[cur_cols].value_counts().max()
            else:
                cur_cnt_cutoff = cnt_cutoff

            cur_cnts = X[cur_cols].value_counts()
            # num_above_cutoff = len(cur_cnts[cur_cnts>=(cur_cnt_cutoff+extra_cutoff)])
            # if num_above_cutoff > 3: c_multiplier = 3
            # else: c_multiplier = len(cur_cnts[cur_cnts>=(cur_cnt_cutoff+extra_cutoff)])
            c_multiplier = len(cur_cnts[cur_cnts>=(cur_cnt_cutoff+extra_cutoff)])
            counts_base = counts_base * c_multiplier
        print(cnt_cutoff, counts_base/1000000)

        if counts_base > 7 * 1000000: break


cnt_cutoff = cnt_cutoff + 10

#%%

drop_cols = []
for pv in param_vars:
    for c in X.columns:
        if pv in c:
            drop_cols.append(c)
base_cols = [c for c in X.columns if c not in drop_cols]
X_predict = X[base_cols].drop_duplicates()
X_predict['cross_idx'] = 1

for pv in param_vars:
    cur_cols = [c for c in X.columns if pv in c]
    cur_cnts = X[cur_cols].value_counts()

    for moc in more_opt_columns:
        if moc in cur_cols[0]: 
            extra_cutoff = extra_cutoff_val
            break
        else: 
            extra_cutoff = 0

    if X[cur_cols].value_counts().max() < cnt_cutoff:
        cur_cnt_cutoff = X[cur_cols].value_counts().max()
    else:
        cur_cnt_cutoff = cnt_cutoff

    cur_col = X[cur_cols].value_counts()[X[cur_cols].value_counts()>=(cur_cnt_cutoff+extra_cutoff)].reset_index().drop(0, axis=1)
    print(cur_col)
    cur_col['cross_idx'] = 1
    X_predict = pd.merge(X_predict, cur_col, on='cross_idx')
    print(pv, X_predict.shape)

X_predict = X_predict.drop('cross_idx', axis=1)
X_predict = X_predict[X.columns]

#%%
winnings_pr = fp.cross_fold_predict('winnings', X_predict, y=pd.Series(np.array([260*32]*len(X_predict)), name='entry'))
grp_cols = [c for c in winnings_pr.columns if c not in ('winnings_pred', 'avg_winnings', 'week', 'year')]
for c in grp_cols:
    winnings_pr[c] = winnings_pr[c].astype('str')

winnings_pr = winnings_pr.sort_values(by='winnings_pred', ascending=False).drop_duplicates().reset_index(drop=True)
winnings_pr = winnings_pr.reset_index().rename(columns={'index': 'param_rank'})

model_notes = 'only_reg_full_stack_non8_times_5_more_options'
date_run = dt.datetime.now().strftime('%Y-%m-%d')
winnings_pr = winnings_pr.assign(model_notes=model_notes, date_run=date_run)

#%%

winnings_pr.loc[:500, [c for c in winnings_pr.columns if 'covar_type' in c]].value_counts()

#%%

try:
    dm.delete_from_db('SimParams', 'Entry_Optimize_Hyperparams', f"model_notes='{model_notes}' AND date_run='{date_run}'", create_backup=False)
    dm.write_to_db(winnings_pr, 'SimParams','Entry_Optimize_Hyperparams', 'append')

except:
    old_df = dm.read('''SELECT * FROM Entry_Optimize_Hyperparams''', 'SimParams')
    winnings_pr = pd.concat([ winnings_pr, old_df], axis=0).reset_index(drop=True)
    winnings_pr = winnings_pr.fillna(0)
    dm.write_to_db(winnings_pr, 'SimParams','Entry_Optimize_Hyperparams', 'replace')
