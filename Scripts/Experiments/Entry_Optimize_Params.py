#%%
from ff.db_operations import DataManage
from ff import general as ffgeneral
import pandas as pd
import numpy as np


root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)


#%%

def entry_optimize_params(df, max_adjust, one_hot=False):

    adjust_winnings = df.groupby(['trial_num']).agg(max_lineup_num=('lineup_num', 'max')).reset_index()
    adjust_winnings['lineups_per_param'] = 30 / (adjust_winnings.max_lineup_num+1)

    df = pd.merge(df, adjust_winnings, on='trial_num')
    df.winnings = df.winnings / df.max_lineup_num

    df.loc[df.winnings >= max_adjust, 'winnings'] = max_adjust
    str_cols = ['week', 'year', 'pred_vers', 'ensemble_vers', 'std_dev_type']
    
    if one_hot:
        str_cols.extend( ['player_drop_multiple','top_n_choices', 'matchup_drop', 'adjust_pos_counts', 
                         'full_model_weight', 'max_lineup_num', 'use_ownership', 'own_neg_frac',
                         'num_top_players', 'static_top_players',
                         'qb_min_iter', 'qb_solo_start', 'qb_set_max_team'])

    df[str_cols] = df[str_cols].astype('str')
    df = df.drop(['trial_num', 'lineup_num', 'max_lineup_num'], axis=1)
    
    df.max_salary_remain = df.max_salary_remain.fillna(5000).astype('float').astype('int').astype('str')
    for c in df.columns:
        if df.dtypes[c] == 'object': 
            df = pd.concat([df, pd.get_dummies(df[c], prefix=c)], axis=1).drop(c, axis=1)

    X = df.drop(['repeat_num', 'winnings'], axis=1)
    y = df.winnings

    return X, y

df = dm.read('''SELECT *  
                FROM Entry_Optimize_Params_Detail 
                JOIN (
                     SELECT week, year, pred_vers, ensemble_vers, std_dev_type, sim_type, trial_num, repeat_num
                      FROM Entry_Optimize_Results
                      ) USING (week, year, trial_num, repeat_num)
                WHERE trial_num > 90
                     -- AND week NOT IN (1,3)
                ''', 'Results')

X, y = entry_optimize_params(df, max_adjust=5000, one_hot=False)



# %%

from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
 
# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)
 
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	yhat, _ = surrogate(model, X)
	best = max(yhat)
	
    # calculate mean and stdev via surrogate function
	mu, std = surrogate(model, Xsamples)
	mu = mu[:, 0]
	
    # calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs
 
# optimize the acquisition function
def opt_acquisition(X, y, model):

	# random search, generate random samples
	Xsamples = random(100)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)

	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	
    # locate the index of the largest scores
	ix = argmax(scores)
	return Xsamples[ix, 0]
 

# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)

# perform the optimization process
for i in range(100):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	
    # sample the point
	actual = objective(x)
	
    # summarize the finding
	est, _ = surrogate(model, [[x]])
	print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	
    # add the data to the dataset
	X = vstack((X, [[x]]))
	y = vstack((y, [[actual]]))
	# update the model
	model.fit(X, y)
 
# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))



#%%

params = dm.read("SELECT * FROM Entry_Optimize_Params", 'Results')
params['param'] = params.param + '_' + params.param_option
params = pd.pivot_table(params, index='trial_num', columns='param').fillna(0)
params.columns = [c[1] for c in params.columns]

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
selector.fit(params)
params = params.loc[:, selector.get_support()].reset_index()
params

# %%

# %%
