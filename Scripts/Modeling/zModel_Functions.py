import pickle
from ff.db_operations import DataManage
from ff import general
from skmodel import SciKitModel
import pandas as pd
import numpy as np
import shap
import gzip
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def load_pickle(path, fname):
    with gzip.open(f"{path}/{fname}.p", 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def load_all_pickles(model_output_path, label):
    pred = load_pickle(model_output_path, f'{label}_pred')
    actual = load_pickle(model_output_path, f'{label}_actual')
    models = load_pickle(model_output_path, f'{label}_models')
    scores = load_pickle(model_output_path, f'{label}_scores')
    try: full_hold = load_pickle(model_output_path, f'{label}_full_hold')
    except: full_hold = None
    return pred, actual, models, scores, full_hold

def get_class_predictions(models_class, df_train_class, df_predict_class):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict_class = pd.DataFrame()

    skm_class_final = SciKitModel(df_train_class, model_obj='class')
    X_class_final, y_class_final = skm_class_final.Xy_split(y_metric='y_act', to_drop=['player', 'team', 'pos'])
    
    for k, v in models_class.items():
        m = skm_class_final.ensemble_pipe(v)
        m.fit(X_class_final, y_class_final)
        cur_pred = pd.Series(m.predict_proba(df_predict_class[X_class_final.columns])[:,1], name=k)
        X_predict_class = pd.concat([X_predict_class, cur_pred], axis=1)

    return X_predict_class


def X_y_stack_old(met, pred, actual):

    X = pd.DataFrame([v for k,v in pred.items() if met in k]).T
    X.columns = [k for k,_ in pred.items() if met in k]
    y = pd.Series(actual[X.columns[0]], name='y_act')

    return X, y


def X_y_stack_full(met, full_hold):
    i = 0
    for k, v in full_hold.items():
        if i == 0:
            df = v.copy()
            df = df.rename(columns={'pred': k})
        else:
            df_cur = v.rename(columns={'pred': k}).drop('y_act', axis=1)
            df = pd.merge(df, df_cur, on=['player', 'team', 'week','year'])
        i+=1

    X = df[[c for c in df.columns if met in c or 'y_act_' in c]].reset_index(drop=True)
    y = df['y_act'].reset_index(drop=True)
    return X, y, df


def X_y_stack(met, full_hold, pred, actual):
    if full_hold is not None:
        X_stack, y_stack, df = X_y_stack_full(met, full_hold)
    else:
        X_stack, y_stack = X_y_stack(met, pred, actual)

    return X_stack, y_stack, df


def get_quant_predictions(df_predict, models, X, y):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict_quant = pd.DataFrame()
    for k, ind_models in models.items():

        predictions = pd.DataFrame()

        for m in ind_models:
            m.fit(X, y)
            cur_pred = m.predict(df_predict[X.columns])
            predictions = pd.concat([predictions, pd.Series(cur_pred)], axis=1)
            
        predictions = predictions.mean(axis=1)
        predictions = pd.Series(predictions, name=k)
        X_predict_quant = pd.concat([X_predict_quant, predictions], axis=1)
    
    return X_predict_quant


def get_reg_predict_features(df_predict, models, X, y):

    # create the full stack pipe with meta estimators followed by stacked model
    X_predict = pd.DataFrame()
    for k, ind_models in models.items():
        
        predictions = pd.DataFrame()

        for m in ind_models:

            m.fit(X, y)
            reg_predict = m.predict(df_predict[X.columns])
            predictions = pd.concat([predictions, pd.Series(reg_predict)], axis=1)
            
        predictions = predictions.mean(axis=1)
        predictions = pd.Series(predictions, name=k)
        X_predict = pd.concat([X_predict, predictions], axis=1)

    return X_predict

def stack_predictions(X_predict, best_models, final_models, model_obj='reg'):
    predictions = pd.DataFrame()
    for bm, fm in zip(best_models, final_models):
        if model_obj=='reg': cur_prediction = np.round(bm.predict(X_predict), 2)
        elif model_obj=='class': cur_prediction = np.round(bm.predict_proba(X_predict)[:,1], 3)
        cur_prediction = pd.Series(cur_prediction, name=fm)
        predictions = pd.concat([predictions, cur_prediction], axis=1)

    return predictions

def best_average_models(skm_stack, scores, final_models, y_stack, stack_val_pred, predictions):

    from sklearn.metrics import r2_score

    n_scores = []
    for i in range(len(scores)):
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:i+1]
        model_idx = np.array(final_models)[top_n]
        
        n_score = -r2_score(y_stack, stack_val_pred[model_idx].mean(axis=1))
        # n_score = skm_stack.custom_score(y_stack, stack_val_pred[model_idx].mean(axis=1))
        n_scores.append(n_score)

    print('All Average Scores:', np.round(n_scores, 3))
    best_n = np.argmin(n_scores[1:])
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:best_n+2]
    
    model_idx = np.array(final_models)[top_n]
    best_val = stack_val_pred[model_idx]

    print('Top Models:', model_idx)
    best_predictions = predictions[model_idx]

    return best_val, best_predictions


def show_scatter_plot(y_pred, y, label='Total', r2=True):
    plt.scatter(y_pred, y)
    plt.xlabel('predictions');plt.ylabel('actual')
    plt.show()

    from sklearn.metrics import r2_score
    if r2: print(f'{label} R2:', r2_score(y, y_pred))
    else: print(f'{label} Corr:', np.corrcoef(y, y_pred)[0][1])


def top_predictions(y_pred, y, r2=False):

    val_high_score = pd.concat([pd.Series(y_pred), pd.Series(y)], axis=1)
    val_high_score.columns = ['predictions','y_act']
    val_high_score = val_high_score[val_high_score.predictions >= \
                                    np.percentile(val_high_score.predictions, 75)]
    show_scatter_plot(val_high_score.predictions, val_high_score.y_act, label='Top', r2=r2)



def shap_plot(best_models, X, model_num=0):

    m = best_models[model_num]
    transformer = Pipeline(m.steps[:-1])
    X_shap = transformer.transform(X)

    if len([i[0] for i in m.get_params()['steps'] if i[0]=='random_sample']) > 0:
        cols = m['random_sample'].columns
    elif len([i[0] for i in m.get_params()['steps'] if i[0]=='k_best']) > 0:
        cols = X.columns[m['k_best'].get_support()]
    else: cols = X.columns
    X_shap = pd.DataFrame(X_shap, columns=cols)

    try: shap_values = shap.TreeExplainer(m.steps[-1][1]).shap_values(X_shap)
    except: shap_values = shap.explainers.Linear(m.steps[-1][1], X_shap).shap_values(X_shap)
    shap.summary_plot(shap_values, X_shap, feature_names=X_shap.columns, plot_size=(8,15), max_display=30, show=False)


def get_sd_cols(df_train, df_predict, X, best_models, num_cols=5):
    
    sd_cols = {}
    for model_num in range(len(best_models)):
        m = best_models[model_num]
        transformer = Pipeline(m.steps[:-1])
        cols = X.columns[transformer['k_best'].get_support()]

        coef_results = pd.DataFrame(m.steps[-1][1].coef_, cols)
        best_cols = abs(coef_results).sort_values(by=0).iloc[-num_cols:].index
        coef_results = coef_results[coef_results.index.isin(best_cols)]

        for c, v in coef_results.iterrows():
            if v[0] < 0:
                df_train['minus_' + c] = np.log(df_train[c].max() - df_train[c] + 1)
                df_predict['minus_' + c] = np.log(df_predict[c].max() - df_predict[c] + 1)
                if 'minus_' + c not in sd_cols.keys(): sd_cols['minus_' + c] = []
                sd_cols['minus_' + c].append(-v[0])
            else:
                if c not in sd_cols.keys():sd_cols[c] = []
                sd_cols[c].append(v[0])

    for k, v in sd_cols.items():
        sd_cols[k] = np.sum(v)
    
    return sd_cols, df_train, df_predict

def assign_sd_max(output, df_predict, sd_df, sd_cols, sd_spline, max_spline, min_spline):
    
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(sd_df[list(sd_cols.keys())])

    df_predict = df_predict.set_index('player')
    df_predict = df_predict.reindex(index=output['player'])
    df_predict = df_predict.reset_index()

    pred_sd_max = pd.DataFrame(sc.transform(df_predict[list(sd_cols.keys())])) * list(sd_cols.values())
    pred_sd_max = pred_sd_max.mean(axis=1)

    output['std_dev'] = sd_spline(pred_sd_max)
    output['max_score'] = max_spline(pred_sd_max)
    output['min_score'] = min_spline(pred_sd_max)
    
    return output