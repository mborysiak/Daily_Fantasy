#%%

#=====================
# Regression Baseline Model
#=====================
import pandas as pd 
import numpy as np
from ff.db_operations import DataManage
from ff import general as ffgeneral
from skmodel.run_models import SciKitModel

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#-----------------
# Format Data for Baseline Projections
#-----------------

baseline = dm.read('''SELECT playerName player_name,
                             week,
                             fantasyPoints, 
                             fantasyPointsRank,
                             salary,
                             `Proj Pts` ProjPts,
                             expertConsensus,
                             expertNathanJahnke,
                             expertKevinCole,
                             expertAndrewErickson,
                             expertIanHartitz
                      FROM PFF_Proj_Ranks a
                      JOIN (SELECT Name playerName, *
                            FROM PFF_Expert_Ranks )
                            USING (playerName, week)
                      WHERE a.position='wr' 
                      ''', 'Pre_PlayerData')

baseline = baseline.fillna(baseline.median())

baseline_m = pd.merge(baseline, 
                      df[['week', 'player_name', 'y_act']], 
                      on=['week', 'player_name'])

#-----------------
# Run Baseline Model
#-----------------

baseline_m = baseline_m.sort_values(by='week')

skm = SciKitModel(baseline_m)
X_base, y_base = skm.Xy_split(y_metric='y_act', to_drop=['player_name'])
cv_time_base = skm.cv_time_splits('week', X_base, 3)

model_base = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('k_best'),
                        skm.piece('rf')])

params = skm.default_params(model_base)
params['k_best__k'] = range(1, X_base.shape[1])

best_model_base = skm.random_search(model_base, X_base, y_base, params, cv=cv_time_base, n_iter=50)
_, _ = skm.val_scores(best_model_base, X_base, y_base, cv_time_base)

imp_cols = X_base.columns[best_model_base['k_best'].get_support()]
skm.print_coef(best_model_base, imp_cols)

base_predict = skm.cv_predict_time(best_model_base, X_base, y_base, cv_time_base)
base_predict = pd.Series(base_predict, name='base_predict')

pred_labels = skm.return_labels(['player_name', 'week'], 'time').reset_index(drop=True)
pred_labels = pd.concat([pred_labels, base_predict], axis=1)

pred_labels = pd.merge(pred_labels, baseline, on=['player_name', 'week'])


#%%

df_m = pd.merge(df, baseline, on=['player_name', 'week'])
df_m = df_m.sort_values(by='week')

skm = SciKitModel(df_m)
X, y = skm.Xy_split(y_metric='y_act', to_drop=['player_name', 'posteam'])
cv_time = skm.cv_time_splits('week', X, 3)

model = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('select_perc'),
                        skm.feature_union([
                                           skm.piece('agglomeration'), 
                                           skm.piece('k_best'), 
                                           skm.piece('pca')
                                           ]),
                        skm.piece('k_best', label_rename='k_best2'),
                        skm.piece('ridge')])

params = skm.default_params(model)
best_model = skm.random_search(model, X, y, params, cv=cv_time, n_iter=50)
_, _ = skm.val_scores(best_model, X, y, cv_time)

#%%

df_m = pd.merge(df, pred_labels, on=['player_name', 'week'])
df_m = df_m.sort_values(by='week')

df_m['y_act'] = np.where(#(df_m.y_act > df_m.base_predict * 1.5) & \
                             (df_m.y_act > 21), 1, 0)                 
# df_m['y_act'] = np.where((df_m.y_act > df_m.ProjPts * 1.15) & (df_m.y_act > 15), 1, 0)                 

skm = SciKitModel(df_m, 'class')
X, y = skm.Xy_split(y_metric='y_act', to_drop=['player_name', 'posteam'])
cv_time = skm.cv_time_splits('week', X, 4)

model = skm.model_pipe([skm.piece('std_scale'), 
                        skm.piece('select_perc_c'),
                        skm.feature_union([
                                           skm.piece('agglomeration'), 
                                           skm.piece('k_best_c')
                                           ]),
                        skm.piece('k_best_c', label_rename='k_best2'),
                        skm.piece('lr_c')])

params = skm.default_params(model)
best_model = skm.random_search(model, X, y, params, cv=cv_time, n_iter=50)
_, _ = skm.val_scores(best_model, X, y, cv_time)

#%%
imp_cols = X.columns[best_model['k_best2'].get_support()]
skm.print_coef(best_model, imp_cols)
