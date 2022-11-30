#%%
from zSim_Helper_Covar import *

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral
from joblib import Parallel, delayed

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#===============
# Settings and User Inputs
#===============
# set the model version
set_weeks = [
   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
]

set_years = [
      2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022
]


pred_versions = len(set_weeks)*['sera1_rsq0_brier1_matt1_lowsample_perc']

ensemble_versions =[
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    'no_weight_yes_kbest_randsample_sera1_rsq0_include2_kfold3',
                    ]

std_dev_types = [
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                'pred_spline_class80_q80_matt1_brier1_kfold3',
                ]

sim_types = len(set_weeks) * ['ownership_ln_pos_fix_flip']

max_trial_num = dm.read("SELECT max(trial_num) FROM Entry_Optimize_Params", 'Results').values[0][0]
trial_num = max_trial_num + 1

print('Trial #', trial_num, '\n==============\n')
    
salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
adjust_winnings = False


set_max_team = None

def get_stats(pos, week, year):
    if pos=='Defense': colname='defTeam'
    else: colname='player'
    return dm.read(f'''SELECT {colname} AS player, fantasy_pts
                        FROM {pos}_Stats
                        WHERE week={week}
                            AND season={year}''', 'FastR')

def calc_winnings(to_add, points, prizes):
    results = pd.DataFrame(to_add, columns=['player'])
    results = pd.merge(results, points, on='player')
    total_pts = results.fantasy_pts.sum()
    idx_match = np.argmin(abs(prizes.Points - total_pts))
    prize_money = prizes.loc[idx_match, 'prize']

    return prize_money

def rand_drop_selected(total_add, drop_multiplier, lineups_per_param):
    to_drop = []
    total_selections = dict(Counter(total_add))
    for k, v in total_selections.items():
        prob_drop = (v * drop_multiplier) / lineups_per_param
        drop_val = np.random.uniform() * prob_drop
        if  drop_val >= 0.5:
            to_drop.append(k)
    return to_drop


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

def get_matchups(week, year):
    df = dm.read(f'''SELECT away_team, home_team
                    FROM Gambling_Lines 
                    WHERE week={week} 
                        and year={year} 
                ''', 'Simulation')

    matchups = []
    for away, home in df.values:
        matchups.append([home, away])

    return matchups

def rand_drop_teams(matchups, drop_matchups, player_teams):
    drop_teams = matchups[np.random.choice(matchups.shape[0], drop_matchups, replace=False), :][0]
    return list(player_teams.loc[player_teams.team.isin(drop_teams), 'player'].values)



def avg_winnings_contest(par_out):

    results = [list(o[0]) for o in par_out]
    avg_winnings = []
    for _ in range(50):
        
        wt_winnings = []
        for w in results:
            wt_winnings.append(w[1] * np.random.choice([1, 0.15], p=[0.34, 0.66]))
        avg_winnings.append(wt_winnings)

    avg_winnings = np.mean(np.array(avg_winnings), axis=0)

    return avg_winnings


def detailed_param_output(d, weighted_winnings, week, year, trial_num, repeat_num):
    cols = list(d.keys())
    cols.append('lineup_num')
    
    params_output = pd.DataFrame(params, columns=cols)
    params_output = params_output.assign(week=week, year=year, trial_num=trial_num, repeat_num=repeat_num)
    params_output['winnings'] = weighted_winnings

    return params_output

def lineup_output(par_out):
    all_lineup_pts = pd.DataFrame()
    for po in par_out:
        all_lineup_pts = pd.concat([all_lineup_pts, po[1]])
    all_lineup_pts = all_lineup_pts.assign(week=week, year=year)

    return all_lineup_pts


def param_set_output(d):
    output = pd.DataFrame(d).T
    output = pd.melt(output.reset_index(), id_vars='index').dropna().sort_values(by='index')
    lpp = pd.DataFrame({'index': ['lineups_per_param'],
                        'variable': [lineups_per_param],
                        'value': 1})
    output = pd.concat([output, lpp], axis=0)
    output['trial_num'] = trial_num
    output.columns = ['param', 'param_option', 'option_value', 'trial_num']

    return output

def get_prizes(week, year):
    prizes = dm.read(f'''SELECT Rank, Points, prize
                        FROM Contest_Results
                        WHERE week={week}
                            AND year={year}
                            AND Contest='Million' ''', 'DK_Results')
    return prizes

def pull_matchups(week, year, pred_vers):

    player_teams = dm.read(f'''SELECT DISTINCT player, team 
                                FROM Covar_Means
                                WHERE week={week}
                                        AND year={year}
                                        AND pred_vers='{pred_vers}'
                                    ''', 'Simulation')
    unique_teams = player_teams.team.unique()
    matchups = np.array([m for m in get_matchups(week, year) if m[0] in unique_teams and m[1] in unique_teams])
    return matchups

def pull_points(week, year):

    points = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
        points = pd.concat([points, get_stats(pos, week, year)])

    return points

#%%

def sim_winnings(adjust_select, player_drop_multiplier, matchup_drop, top_n_choices, 
                full_model_rel_weight, covar_type, min_players_same_team, 
                min_players_opp_team, num_top_players, qb_min_iter, qb_set_max_team, qb_solo_start,
                static_top_players, use_ownership, own_neg_frac, max_salary_remain, 
                num_iters, lineups_per_param, week, year, pred_vers, ensemble_vers, std_dev_type
                ):
    
    prizes = get_prizes(week, year)
    matchups = pull_matchups(week, year, pred_vers)
    points = pull_points(week, year)

    try: min_players_opp_team = int(min_players_opp_team)
    except: pass

    try: min_players_same_team = float(min_players_same_team)
    except: pass

    if covar_type=='no_covar': use_covar=False
    else: use_covar=True

    sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                                pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                                covar_type=covar_type, full_model_rel_weight=full_model_rel_weight, 
                                use_covar=use_covar, use_ownership=use_ownership, 
                                salary_remain_max=max_salary_remain)

    winnings = []
    total_add = []
    to_drop_selected = []

    for t in range(lineups_per_param):

        to_add = []
        to_drop = []
        to_drop.extend(to_drop_selected)
        
        # if matchup_drop > 0: to_drop = rand_drop_teams(matchups, matchup_drop)
        # else: to_drop = []
        # to_drop.extend(to_drop_selected)

        for i in range(9):
            results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, 
                                    min_players_opp_team_input=min_players_opp_team, 
                                    adjust_select=adjust_select, num_matchup_drop=matchup_drop,
                                    own_neg_frac=own_neg_frac, n_top_players=num_top_players,
                                    static_top_players=static_top_players, qb_min_iter=qb_min_iter,
                                    qb_set_max_team=qb_set_max_team, qb_solo_start=qb_solo_start)
            
            prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
            try: 
                selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
                to_add.append(selected_player)
            except: 
                pass
            
        prize_money = calc_winnings(to_add, points, prizes)
        winnings.append(prize_money)

        total_add.extend(to_add)
        to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier, lineups_per_param)

    return np.mean(winnings)


def objective(params):
    
    iter_cats = zip(set_weeks, set_years, pred_versions, ensemble_versions, std_dev_types)
    input_args = []
    for week, year, pred_vers, ensemble_vers, std_dev_type in iter_cats:
        input_args.append([params['adjust_pos_counts'], 
                           params['player_drop_multiple'], 
                           params['matchup_drop'], 
                           params['top_n_choices'], 
                           params['full_model_weight'], 
                           params['covar_type'], 
                           params['min_player_same_team'], 
                           params['min_players_opp_team'], 
                           params['num_top_players'], 
                           params['qb_min_iter'], 
                           params['qb_set_max_team'], 
                           params['qb_solo_start'], 
                           params['static_top_players'], 
                           params['use_ownership'], 
                           params['own_neg_frac'], 
                           params['max_salary_remain'], 
                           params['num_iters'], 
                           params['lineups_per_param'],
                           week, year, pred_vers, ensemble_vers, std_dev_type])

    # for p1, p2, p3, p4, p5, p6,p7, p8, p9, p10, p11, p12,p13, p14, p15, p16, p17, p18,p19, p20, p21, p22, p23 in input_args[:3]:

    #     winnings = sim_winnings(p1, p2, p3, p4, p5, p6,
    #                 p7, p8, p9, p10, p11, p12,
    #                 p13, p14, p15, p16, p17, p18,
    #                 p19, p20, p21, p22, p23)
    #     print(winnings)
                
    winnings = Parallel(n_jobs=-1, verbose=0)(delayed(sim_winnings)(p1, p2, p3, p4, p5, p6,
                                                                   p7, p8, p9, p10, p11, p12,
                                                                   p13, p14, p15, p16, p17, p18,
                                                                   p19, p20, p21, p22, p23) for 
                                                                   p1, p2, p3, p4, p5, p6,
                                                                   p7, p8, p9, p10, p11, p12,
                                                                   p13, p14, p15, p16, p17, p18,
                                                                   p19, p20, p21, p22, p23 in input_args)
    
    print({'Week'+str(i+1): w for i, w in enumerate(winnings)}) 
    print('Total Winnings:', np.sum(winnings))
    
    return -np.mean(winnings)-np.percentile(winnings, 50)
        
#%%
from hyperopt import fmin, tpe, hp, space_eval

space = {
        'adjust_pos_counts': hp.choice('adjust_pos_counts', [True, False]),
        'player_drop_multiple': hp.choice('player_drop_multiple',  np.arange(0, 5, dtype=int)),
        'matchup_drop': hp.choice('matchup_drop', np.arange(0, 3, dtype=int)),
        'top_n_choices': hp.choice('top_n_choices',  np.arange(0, 4, dtype=int)),
        'full_model_weight': hp.choice('full_model_weight', [0.2, 1, 5]),
        'covar_type': hp.choice('covar_type', ['no_covar', 'team_points_trunc']),
        'min_player_same_team': hp.choice('min_player_same_team', ['Auto', 2, 3]),
        'min_players_opp_team': hp.choice('min_players_opp_team', ['Auto', 1, 2]),
        'num_top_players': hp.choice('num_top_players', np.arange(2, 6, dtype=int)),
        'qb_min_iter': hp.choice('qb_min_iter', [0, 1, 9]),
        'qb_set_max_team': hp.choice('qb_set_max_team', [True, False]),
        'qb_solo_start': hp.choice('qb_solo_start', [True, False]),
        'static_top_players': hp.choice('static_top_players', [True, False]),
        'use_ownership': hp.uniform('use_ownership', 0, 1),
        'own_neg_frac': hp.uniform('own_neg_frac', 0, 1),
        'max_salary_remain': hp.choice('max_salary_remain', np.arange(200, 2100, 100, dtype=int)),
        'num_iters': hp.choice('num_iters', np.arange(50, 175, 25, dtype=int)),
        'lineups_per_param': hp.choice('lineups_per_param', [2,3])
}

fmin_result = fmin(objective, space, algo=tpe.suggest, max_evals=100)
print(space_eval(space, fmin_result))

# %%
fmin_result = {'adjust_pos_counts': 0.74564620653125,
 'covar_type': 0,
 'full_model_weight': 2,
 'lineups_per_param': 0,
 'matchup_drop': 0,
 'max_salary_remain': 4,
 'min_player_same_team': 0,
 'min_players_opp_team': 0,
 'num_iters': 5,
 'num_top_players': 3,
 'own_neg_frac': 0.7065411384909316,
 'player_drop_multiple': 0.9807781565825981,
 'qb_min_iter': 1,
 'qb_set_max_team': 1,
 'qb_solo_start': 1,
 'static_top_players': 1,
 'top_n_choices': 0.18001757324562861,
 'use_ownership': 0.6085929357842961}

# %%
