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
    1, 2, 3, 4, 5, 6
]

set_years = [
      2022,
      2022, 2022, 2022, 2022, 2022
]

pred_versions = [   
               'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate',
               'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate',
               'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate',
               'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate',
               'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate',
               'sera1_rsq0_brier2_matt1_lowsample_perc_calibrate',
]

ensemble_versions = [
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2',
                    'no_weight_yes_kbest_randsample_sera10_rsq1_include2',
]

std_dev_types = [
                'pred_spline_class80_matt0_brier1_calibrate_pipe', 
                'pred_spline_class80_matt0_brier1_calibrate_pipe', 
                'pred_spline_class80_matt0_brier1_calibrate_pipe', 
                'pred_spline_class80_matt0_brier1_calibrate_pipe', 
                'pred_spline_class80_matt0_brier1_calibrate_pipe', 
                'pred_spline_class80_matt0_brier1_calibrate_pipe', 
]


sim_types = [
             'ownership_ln_prob',
             'ownership_ln_prob',
             'ownership_ln_prob',
             'ownership_ln_prob',
             'ownership_ln_prob',
             'ownership_ln_prob',
]

max_trial_num = dm.read("SELECT max(trial_num) FROM Entry_Optimize_Params", 'Results').values[0][0]
trial_num = max_trial_num + 1

print('Trial #', trial_num, '\n==============\n')

for repeat_num in range(10):

    all_winnings = []
    output_results = []
    iter_cats = zip(set_weeks, set_years, pred_versions, ensemble_versions, std_dev_types)
    for week, year, pred_vers, ensemble_vers, std_dev_type in iter_cats:

        salary_cap = 50000
        pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
        num_iters = 50
        adjust_winnings = False

        print(f'\nWeek {week}, Repeat {repeat_num}\n---------\n')

        set_max_team = None

        def get_stats(pos):
            if pos=='Defense': colname='defTeam'
            else: colname='player'
            return dm.read(f'''SELECT {colname} AS player, fantasy_pts
                                FROM {pos}_Stats
                                WHERE week={week}
                                    AND season={year}''', 'FastR')

        def calc_winnings(to_add, lineup_num, unique_lineup_num):
            results = pd.DataFrame(to_add, columns=['player'])
            results = pd.merge(results, points, on='player')
            results = results.assign(lineup_num=lineup_num, repeat_num=repeat_num, trial_num=trial_num, 
                                     unique_lineup_num=unique_lineup_num)

            total_pts = results.fantasy_pts.sum()
            idx_match = np.argmin(abs(prizes.Points - total_pts))
            prize_money = prizes.loc[idx_match, 'prize']

            return results, np.round(total_pts,1), prize_money

        def rand_drop_selected(total_add, drop_multiplier):
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
                                AND Contest='Million' ''', 'DK_Results')

        
        d = {
            'adjust_pos_counts': {
                True: 0.7, 
                False: 0.3
            },

            'player_drop_multiple': {
                0: 0.5, 
                4: 0.5,
                6: 0
            },
                        
            'matchup_drop': {
                0: 1,
                1: 0,
            },

            'top_n_choices': {
                0: 1,
                2: 0,
                4: 0,
            },

            'full_model_weight': {
                0.2: 0.2,
                1: 0.2,
                5: 0.6
            },

            'covar_type': {
                'no_covar': 0.5,
                'team_points_trunc': 0.5,
                'kmeans_trunc': 0
            },

            'min_player_same_team': {
                'Auto': 1,
                2: 0,
                3: 0,
                0: 0
            },

            'min_players_opp_team': {
                0: 0.5,
                'Auto': 0.5
            },

            'use_ownership': {
                True: 1,
                False: 0
            },

            'max_salary_remain': {
                None: 0.5,
                200: 0,
                500: 0.5,
                1000: 0
            }
        }

        lineups_per_param = 3

        params = []
        for i in range(int(30/lineups_per_param)):
            cur_params = []
            for param, param_options in d.items():
                param_vars = list(param_options.keys())
                param_prob = list(param_options.values())
                cur_params.append(np.random.choice(param_vars, p=param_prob))

            cur_params.append(i)
            params.append(cur_params)

        
        def sim_winnings(adjust_select, player_drop_multiplier, matchup_drop, top_n_choices, 
                        full_model_rel_weight, covar_type, min_players_same_team, 
                        min_players_opp_team, use_ownership, max_salary_remain, param_iter,
                        ):
            
            try: min_players_opp_team = int(min_players_opp_team)
            except: pass

            try: min_players_same_team = int(min_players_same_team)
            except: pass

            if covar_type=='no_covar': use_covar=False
            else: use_covar=True

            sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                                        pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                                        covar_type=covar_type, full_model_rel_weight=full_model_rel_weight, 
                                        use_covar=use_covar, use_ownership=use_ownership, 
                                        salary_remain_max=max_salary_remain)

            winnings = []
            points_record = []
            total_add = []
            to_drop_selected = []

            lineup_pts = pd.DataFrame()
            for t in range(lineups_per_param):

                to_add = []

                if matchup_drop > 0: to_drop = rand_drop_teams(matchups, matchup_drop)
                else: to_drop = []
                to_drop.extend(to_drop_selected)

                for i in range(9):
                    results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, 
                                            min_players_opp_team_input=min_players_opp_team, 
                                            adjust_select=adjust_select)

                    prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
                    selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
                    to_add.append(selected_player)

                to_add.append(param_iter)

                unique_lineup_num = (param_iter * lineups_per_param) + t
                cur_lineup_pts, total_pts, prize_money = calc_winnings(to_add, param_iter, unique_lineup_num)
                lineup_pts = pd.concat([lineup_pts, cur_lineup_pts], axis=0)
                winnings.append(prize_money); points_record.append(total_pts)

                total_add.extend(to_add)
                to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier)

            sim_results = summary_results(winnings, points_record, adjust_winnings)
            print(sim_results)
            sim_results = list(sim_results.values[0])
            
            return sim_results, lineup_pts


        par_out = Parallel(n_jobs=-1, verbose=0)(delayed(sim_winnings)(adj, pdm, md, tn, fmw, ct, mpst, mpot, uo, msr, param_i) for \
                                                                       adj, pdm, md, tn, fmw, ct, mpst, mpot, uo, msr, param_i in params)

        weighted_winnings = avg_winnings_contest(par_out)
        cur_week_avg_winnings = np.sum(weighted_winnings)
        print('Average Winnings:', int(cur_week_avg_winnings))

        all_winnings.append(cur_week_avg_winnings)
        print('Total Cumulative Winnings:', int(np.sum(all_winnings)))
        output_results.append([week, year, pred_vers, ensemble_vers, std_dev_type, cur_week_avg_winnings])

        # save out the details of each lineup
        param_output = detailed_param_output(d, weighted_winnings, week, year, trial_num, repeat_num)
        dm.write_to_db(param_output, 'Results', 'Entry_Optimize_Params_Detail', 'append')

        lineup_output_cur = lineup_output(par_out)
        dm.write_to_db(lineup_output_cur, 'Results', 'Entry_Optimize_Lineups', 'append')

    # save out the high level results of the overall week
    output_results = pd.DataFrame(output_results, columns=['week', 'year', 'pred_vers', 'ensemble_vers', 'std_dev_type', 'avg_winnings'])
    output_results['trial_num'] = trial_num
    output_results['repeat_num'] = repeat_num
    dm.write_to_db(output_results, 'Results', 'Entry_Optimize_Results', 'append')

# save out the initial params that were set for randomization
output = param_set_output(d)
dm.write_to_db(output, 'Results', 'Entry_Optimize_Params', 'append')


#%%

# to_delete_num=39
# df = dm.read(f"SELECT * FROM Entry_Optimize_Lineups WHERE trial_num!={to_delete_num}", 'Results')
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Lineups', 'replace')

# df = dm.read(f"SELECT * FROM Entry_Optimize_Params WHERE trial_num!={to_delete_num}", 'Results')
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Params', 'replace')

# df = dm.read(f"SELECT * FROM Entry_Optimize_Params_Detail WHERE trial_num!={to_delete_num}", 'Results')
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Params_Detail', 'replace')

# df = dm.read(f"SELECT * FROM Entry_Optimize_Results WHERE trial_num!={to_delete_num}", 'Results')
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Results', 'replace')


# %%
