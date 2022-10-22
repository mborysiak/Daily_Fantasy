#%%
from zSim_Helper_Covar import *

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#===============
# Settings and User Inputs
#===============

# set the model version
set_weeks = [5]

set_years = [
      2022
]

pred_versions = [   
               'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate',
]

ensemble_versions = [
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2',
]

std_dev_types = [
                'pred_spline_class80_matt0_brier1', 
]


sim_types = [
             'ownership_ln_prob',
]

contests = [
            'Million',
            'Million',
            'Million',
            'Million'
]

all_winnings = []
iter_cats = zip(set_weeks, set_years, pred_versions, ensemble_versions, std_dev_types, sim_types, contests)
for week, year, pred_vers, ensemble_vers, std_dev_type, sim_type, contest in iter_cats:

    salary_cap = 50000
    pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
    num_iters = 50
    TOTAL_LINEUPS = 10
    adjust_winnings = True

    print(f'\nWeek {week} PredVer: {pred_vers} EnsVer: {ensemble_vers} SDType:{std_dev_type} SimType:{sim_type} Contest:{contest}\n===============\n')

    set_max_team = None

    if 'ownership' in sim_type:
        use_ownership=True
    else:
        use_ownership=False
    

    def get_stats(pos):
        if pos=='Defense': colname='defTeam'
        else: colname='player'
        return dm.read(f'''SELECT {colname} AS player, fantasy_pts
                            FROM {pos}_Stats
                            WHERE week={week}
                                AND season={year}''', 'FastR')

    def calc_winnings(to_add):
        results = pd.DataFrame(to_add, columns=['player'])
        results = pd.merge(results, points, on='player')
        total_pts = results.fantasy_pts.sum()
        idx_match = np.argmin(abs(prizes.Points - total_pts))
        prize_money = prizes.loc[idx_match, 'prize']

        return np.round(total_pts,1), prize_money

    def rand_drop_selected(total_add, drop_multiplier):
        to_drop = []
        total_selections = dict(Counter(total_add))
        for k, v in total_selections.items():
            prob_drop = (v * drop_multiplier) / TOTAL_LINEUPS
            drop_val = np.random.uniform() * prob_drop
            if  drop_val >= 0.5:
                to_drop.append(k)
        return to_drop

    def get_my_results(week, contest):
        path = f'//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/MBorysiak/DK/Results/{year}/{contest}/week{week}.csv'
        my_results = pd.read_csv(path, low_memory=False)
        my_results = my_results.loc[my_results.EntryName.str.contains('mborysi'), 'Points'].values

        actual_winnings = []
        for mr in my_results:
            idx_match = np.argmin(abs(prizes.Points - mr))
            prize_money = prizes.loc[idx_match, 'prize']
            actual_winnings.append(prize_money)

        if len(actual_winnings) == 0: actual_winnings=[0]
        if len(my_results) == 0: my_results=[100]

        return actual_winnings, my_results


    def summary_results(winnings, points, adjust_winnings=True):

        frac = 1
        if adjust_winnings: 
            if len(winnings) != 10:
                frac = 10 / (len(winnings)+1)

        total_winnings = np.sum(winnings) * frac
        mean_points = np.mean(points)
        max_points = np.max(points)
        max_winnings = np.max(winnings)
        num_placed = len([i for i in winnings if i>0]) * frac

        results = pd.DataFrame([[num_placed, total_winnings, max_winnings, mean_points, max_points]],
                                columns=['NumberPlaced', 'TotalWinnings', 'MaxWinnings', 'MeanPoints', 'MaxPoints'])
        results = round(results, 1)
        return results

    def get_matchups():
        df = dm.read(f'''SELECT away_team, home_team
                         FROM Gambling_Lines 
                         WHERE week={week} 
                               and year={year} 
                    ''', 'Simulation')

        matchups = []
        for away, home in df.values:
            matchups.append([home, away])

        return matchups

    def rand_drop_teams(matchups, drop_matchups):
        drop_teams = matchups[np.random.choice(matchups.shape[0], drop_matchups, replace=False), :][0]
        return list(player_teams.loc[player_teams.team.isin(drop_teams), 'player'].values)

    player_teams = dm.read(f'''SELECT DISTINCT player, team 
                               FROM Covar_Means
                               WHERE week={week}
                                     AND year={year}
                                     AND pred_vers='{pred_vers}'
                                     ''', 'Simulation')
    unique_teams = player_teams.team.unique()
    matchups = np.array([m for m in get_matchups() if m[0] in unique_teams and m[1] in unique_teams])

    points = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
        points = pd.concat([points, get_stats(pos)])

    prizes = dm.read(f'''SELECT Rank, Points, prize
                         FROM Contest_Results
                         WHERE week={week}
                            AND year={year}
                            AND Contest='{contest}' ''', 'DK_Results')

    min_prize_pts = round(prizes.loc[prizes.prize > 0, 'Points'].min(),1)
    mean_prize_pts = round(prizes.loc[prizes.prize > 0, 'Points'].mean(),1)
    max_prize_pts = round(prizes.loc[prizes.prize > 0, 'Points'].max(),1)
    print(f'Min Prize Points: {min_prize_pts}\nMean Prize Points: {mean_prize_pts}\nMax Prize Points: {max_prize_pts}')

    actual_winnings, actual_points = get_my_results(week, contest)
    actual_results = summary_results(actual_winnings, actual_points)
    actual_results.columns = ['My'+ar for ar in actual_results]
    print(actual_results)


    from itertools import product

    def dict_configs(d):
        for vcomb in product(*d.values()):
            yield dict(zip(d.keys(), vcomb))

    G = {
        'adjust_pos_counts': [True, False], 
        'drop_player_multiple': [0, 4], 
        'drop_team_frac': [0, 1],
        'top_n_choices': [0, 4],
        'full_model_rel_weight': [0.2, 1, 5],
        'covar_type': ['no_covar', 'team_points_trunc'],
        'min_player_same_team': ['Auto'],
        'iter': [0, 1, 2],
        }

    params = []
    for config in dict_configs(G):
        params.append(list(config.values()))

    for i in range(len(params)):
        params[i].append(i)
        params[i].append(use_ownership)
    
    def sim_winnings(adjust_select, player_drop_multiplier, team_drop_frac, top_n_choices, 
                     full_model_rel_weight, covar_type, min_players_same_team, iter_num, param_iter,
                     use_ownership):
        
        if covar_type=='no_covar': use_covar=False
        else: use_covar=True

        sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                                    pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                                    covar_type=covar_type, full_model_rel_weight=full_model_rel_weight, 
                                    use_covar=use_covar, use_ownership=use_ownership)

        winnings = []
        points_record = []
        total_add = []
        to_drop_selected = []
        lineups = []
        for t in range(TOTAL_LINEUPS):

            to_add = []

            if team_drop_frac > 0: to_drop = rand_drop_teams(matchups, team_drop_frac)
            else: to_drop = []
            to_drop.extend(to_drop_selected)

            for i in range(9):
                results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, 
                                         min_players_opp_team_input='Auto', adjust_select=adjust_select)

                prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
                selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
                to_add.append(selected_player)

            to_add.append(param_iter)
            lineups.append(to_add)

            total_pts, prize_money = calc_winnings(to_add)
            winnings.append(prize_money); points_record.append(total_pts)

            total_add.extend(to_add)
            to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier)

        sim_results = summary_results(winnings, points_record, adjust_winnings)
        print(sim_results)
        sim_results = list(sim_results.values[0])
        
        return sim_results, lineups

    # for adj, pdm, tdf, tn, fmw, ct, mpst, i, param_i, uo in params[:1]:
        # sim_winnings(adj, pdm, tdf, tn, fmw, ct, mpst, i, param_i, uo)

    from joblib import Parallel, delayed

    par_out = Parallel(n_jobs=-1, verbose=10)(delayed(sim_winnings)(adj, pdm, tdf, tn, fmw, ct, mpst, it_n, param_i, uo) for \
                                                                    adj, pdm, tdf, tn, fmw, ct, mpst, it_n, param_i, uo in params)
    
    lineups = []
    for o in par_out:
        lineups.extend(o[1])
    lineups = pd.DataFrame(lineups).rename(columns={9: 'param_iter'})
    lineups = lineups.assign(week=week, year=year, pred_vers=pred_vers, 
                             ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                             sim_type=sim_type)

    results = [list(o[0]) for o in par_out]
    output = pd.concat([pd.DataFrame(params), pd.DataFrame(results)], axis=1)
    output = pd.concat([output, actual_results], axis=1).fillna(method='ffill')

    cols = list(G.keys())
    cols.extend(['param_iter', 'use_ownership', 'lineups_placed', 'total_winnings', 'max_winnings', 'avg_points', 'max_points', 
                 'my_number_placed', 'my_total_winnings', 'my_max_winnings', 'my_mean_points', 'my_max_points'])
    
    output.columns = cols
    output = output.assign(min_prize_points=min_prize_pts, mean_prize_points=mean_prize_pts, max_prize_points=max_prize_pts,
                           week=week, year=year, pred_vers=pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                           num_iters=num_iters)


    #%%
    output['sim_type'] = sim_type
    output['drop_team_frac'] = output.drop_team_frac / len(matchups)

    output['pred_proba'] = 0
    output.loc[output.pred_vers.str.contains('proba'), 'pred_proba'] = 1

    output['pred_sera'] = 0
    output.loc[output.pred_vers.str.contains('sera'), 'pred_sera'] = 1

    output['pred_brier'] = 0
    output.loc[output.pred_vers.str.contains('brier'), 'pred_brier'] = 1

    output['pred_lowsample'] = 0
    output.loc[output.pred_vers.str.contains('lowsample'), 'pred_lowsample'] = 1

    output['ens_sample_weights'] = 0
    output.loc[output.ensemble_vers.str.contains('yes_weight'), 'ens_sample_weights'] = 1

    output['ens_kbest'] = 0
    output.loc[output.ensemble_vers.str.contains('yes_kbest'), 'ens_kbest'] = 1

    output['ens_randsample'] = 0
    output.loc[output.ensemble_vers.str.contains('randsample'), 'ens_randsample'] = 1

    output['ens_sera'] = 0
    output.loc[output.ensemble_vers.str.contains('sera'), 'ens_sera'] = 1

    output['bridge'] = 0
    output.loc[output.std_dev_type.str.contains('bridge'), 'bridge'] = 1

    output['std_spline'] = 0
    output.loc[output.std_dev_type.str.contains('spline'), 'std_spline'] = 1

    output['min_best_models'] = 3

    output.loc[output.std_dev_type=='spline', 'std_experts'] = 0.75
    output.loc[output.std_dev_type=='spline', 'std_actuals'] = 0.25

    output.loc[output.std_dev_type=='spline_actuals', 'std_experts'] = 0.5
    output.loc[output.std_dev_type=='spline_actuals', 'std_actuals'] = 0.5

    output.loc[output.std_dev_type=='spline_splquant_actuals', 'std_splquantile'] = 0.5
    output.loc[output.std_dev_type=='spline_splquant_actuals', 'std_actuals'] = 0.5

    output.loc[output.std_dev_type=='spline_pred_actuals', 'std_predictions'] = 0.5
    output.loc[output.std_dev_type=='spline_pred_actuals', 'std_actuals'] = 0.5

    output.loc[output.std_dev_type=='spline_pred', 'std_predictions'] = 1

    output.loc[output.std_dev_type=='spline_quantile', 'std_splquantile'] = 1

    output.loc[output.std_dev_type=='spline_proj_only', 'std_experts'] = 0.5
    output.loc[output.std_dev_type=='spline_proj_only', 'std_splquantile'] = 0.33
    output.loc[output.std_dev_type=='spline_proj_only', 'std_predictions'] = 0.17

    output.loc[output.std_dev_type=='spline_all', 'std_experts'] = 0.33
    output.loc[output.std_dev_type=='spline_all', 'std_splquantile'] = 0.22
    output.loc[output.std_dev_type=='spline_all', 'std_predictions'] = 0.11
    output.loc[output.std_dev_type=='spline_all', 'std_actuals'] = 0.33

    output.loc[output.std_dev_type=='spline_all_no_perc', 'std_experts'] = 0.43
    output.loc[output.std_dev_type=='spline_all_no_perc', 'std_predictions'] = 0.14
    output.loc[output.std_dev_type=='spline_all_no_perc', 'std_actuals'] = 0.43

    output.loc[output.std_dev_type=='spline_proj_only_no_perc', 'std_experts'] = 0.75
    output.loc[output.std_dev_type=='spline_proj_only_no_perc', 'std_predictions'] = 0.25

    output.loc[output.pred_vers.str.contains('fixed_model_clone'), 'proper_ensemble'] = 1
    output.loc[output.std_dev_type.str.contains('coef'), 'std_coef'] = 1
    output.loc[output.std_dev_type.str.contains('adp'), 'std_experts'] = 1

    output.loc[output.std_dev_type.str.contains('isotonic'), 'std_isotonic'] = 1
    output['pred_perc'] = 0
    output.loc[~output.pred_vers.str.contains('lowsample'), 'pred_perc'] = 1

    output.loc[output.std_dev_type.isin(['pred_isotonic', 'pred_spline', 'pred_isotonic_spline']), 'std_predictions'] = 1

    output['std_class'] = 0
    output.loc[output.std_dev_type.str.contains('class'), 'std_class'] = 1

    output['std_calibrate'] = 0
    output.loc[output.std_dev_type.str.contains('calibrate'), 'std_calibrate'] = 1

    output['pred_calibrate'] = 0
    output.loc[output.pred_vers.str.contains('calibrate'), 'pred_calibrate'] = 1

    lineups['sim_type'] = sim_type
    output['Contest'] = contest
    output['NumPlayers'] = 9

    def get_objective_wts(col, val):
        col = col.split('_')
        val_word = [c for c in col if val in c]

        try: val_int = int(val_word[0].replace(val, ''))
        except: val_int = 0
        
        return val_int

    output['ens_sera_wt'] = output.ensemble_vers.apply(lambda x: get_objective_wts(x, 'sera'))
    output['ens_rsq_wt'] = output.ensemble_vers.apply(lambda x: get_objective_wts(x, 'rsq'))

    output['pred_sera_wt'] = output.pred_vers.apply(lambda x: get_objective_wts(x, 'sera'))
    output['pred_rsq_wt'] = output.pred_vers.apply(lambda x: get_objective_wts(x, 'rsq'))

    output['pred_brier_wt'] = output.pred_vers.apply(lambda x: get_objective_wts(x, 'brier'))
    output['pred_matt_wt'] = output.pred_vers.apply(lambda x: get_objective_wts(x, 'matt'))

    output['std_matt_wt'] = output.std_dev_type.apply(lambda x: get_objective_wts(x, 'matt'))
    output['std_brier_wt'] = output.std_dev_type.apply(lambda x: get_objective_wts(x, 'brier'))
    
    dm.write_to_db(output, 'Results', 'Winnings_Optimize', 'append')
    dm.write_to_db(lineups, 'Results', 'Lineups_Optimize', 'append')


#%%

# df = dm.read('''SELECT * FROM Winnings_Optimize''', 'Results')

# df['std_calibrate'] = 0
# df.loc[df.ensemble_vers.str.contains('calibrate'), 'std_calibrate'] = 1

# df['pred_calibrate'] = 0
# df.loc[df.pred_vers.str.contains('calibrate'), 'pred_calibrate'] = 1

# df['std_class'] = 0
# df.loc[df.std_dev_type.str.contains('class'), 'std_class'] = 1

# df.loc[df.ensemble_vers.str.contains('calibrate'), 'std_dev_type'] = \
#     df.loc[df.ensemble_vers.str.contains('calibrate'), 'std_dev_type'] + '_calibrate'

# df.loc[df.ensemble_vers.str.contains('calibrate'), 'ensemble_vers'] = \
#     df.loc[df.ensemble_vers.str.contains('calibrate'), 'ensemble_vers'].apply(lambda x: x.replace('_calibrate', ''))


# df.loc[df.ensemble_vers.str.contains('_matt1_brier_1'), 'std_dev_type'] = \
#     df.loc[df.ensemble_vers.str.contains('_matt1_brier_1'), 'std_dev_type'] + '_matt1_brier1'

# df.loc[df.ensemble_vers.str.contains('_matt1_brier_1'), 'ensemble_vers'] = \
#     df.loc[df.ensemble_vers.str.contains('_matt1_brier_1'), 'ensemble_vers'].apply(lambda x: x.replace('_matt1_brier_1', ''))

# def get_objective_wts(col, val):
#     col = col.split('_')
#     val_word = [c for c in col if val in c]

#     try: val_int = int(val_word[0].replace(val, ''))
#     except: val_int = 0
    
#     return val_int

# df['ens_sera_wt'] = df.ensemble_vers.apply(lambda x: get_objective_wts(x, 'sera'))
# df['ens_rsq_wt'] = df.ensemble_vers.apply(lambda x: get_objective_wts(x, 'rsq'))
# df.loc[(df.ensemble_vers.str.contains('sera')) & (df.ens_sera_wt==0), 'ens_sera_wt'] = 1

# df['pred_sera_wt'] = df.pred_vers.apply(lambda x: get_objective_wts(x, 'sera'))
# df['pred_rsq_wt'] = df.pred_vers.apply(lambda x: get_objective_wts(x, 'rsq'))
# df.loc[(df.pred_vers.str.contains('sera')) & (df.pred_sera_wt==0), 'pred_sera_wt'] = 1

# df['pred_brier_wt'] = df.pred_vers.apply(lambda x: get_objective_wts(x, 'brier'))
# df['pred_matt_wt'] = df.pred_vers.apply(lambda x: get_objective_wts(x, 'matt'))
# df.loc[(df.pred_vers.str.contains('brier')) & (df.pred_brier_wt==0), 'pred_brier_wt'] = 1
# df.loc[(df.pred_vers.str.contains('brier')) & (df.pred_matt_wt==0), 'pred_matt_wt'] = 1

# df['std_matt_wt'] = df.std_dev_type.apply(lambda x: get_objective_wts(x, 'matt'))
# df['std_brier_wt'] = df.std_dev_type.apply(lambda x: get_objective_wts(x, 'brier'))

# dm.write_to_db(df, 'Results', 'Winnings_Optimize', 'replace')

# %%
