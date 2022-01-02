#%%
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
#%%

X, y = make_regression(n_samples=5000, n_features=100, n_informative=10, noise=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=4000)

#%%
# First, set up the scikit-learn workflow, wrapped within a function.
def train(params):
  """
  This is our main training function which we pass to Hyperopt.
  It takes in hyperparameter settings, fits a model based on those settings,
  evaluates the model, and returns the loss.

  :param params: map specifying the hyperparameter settings to test
  :return: loss for the fitted model
  """
  # We will tune 2 hyperparameters:
  #  regularization and the penalty type (L1 vs L2).
  
  params['rf__n_estimators'] = int(params['rf__n_estimators'])
  params['rf__max_depth'] = int(params['rf__max_depth'])
  
  # Turn up tolerance for faster convergence
  m = Pipeline([('std_scale', StandardScaler()),
                ('rf', RandomForestRegressor(n_jobs=2))])
  m.set_params(**params)
  m.fit(X_train, y_train)
  score = r2_score(y_test, m.predict(X_test))

  return {'loss': -score, 'status': STATUS_OK}

# Next, define a search space for Hyperopt.
search_space = {
  'rf__n_estimators': hp.quniform('rf__n_estimators', 20, 200, 10),
  'rf__max_depth': hp.quniform('rf__max_depth', 2, 30, 2)
}


#%%
# We can distribute tuning across our Spark cluster
# by calling `fmin` with a `SparkTrials` instance.
spark_trials = SparkTrials(parallelism=8)
best_hyperparameters = fmin(
  fn=train,
  space=search_space,
  algo=tpe.suggest,
  trials=spark_trials,
  max_evals=32)
best_hyperparameters

# %%
# Turn up tolerance for faster convergence
m = Pipeline([('std_scale', StandardScaler()),
                ('rf', RandomForestRegressor(n_jobs=2))])
m.set_params(**best_hyperparameters)
# %%
