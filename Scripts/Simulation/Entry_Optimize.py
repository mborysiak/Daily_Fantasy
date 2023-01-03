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
  # 13, 14, 15, 16, 17,
   1, 2, 3, 4, 5, 6,#7, 8, 9, 10, 11, 12, 13,
   15, 16
]

set_years = [
    #  2021, 2021, 2021, 2021, 2021,
      2022, 2022, 2022, 2022, 2022, 2022, #2022, 2022, 2022, 2022, 2022, 2022, 2022
      2022, 2022
]


pred_vers = 'sera1_rsq0_brier1_matt1_lowsample_perc_ffa_fc'
ensemble_vers ='no_weight_yes_kbest_randsample_sera10_rsq1_include2_kfold3_fullstack'   
std_dev_type = 'pred_spline_class80_q80_matt1_brier1_kfold3'
ownership_vers = 'standard_ln'

max_trial_num = dm.read("SELECT max(trial_num) FROM Entry_Optimize_Params", 'Results').values[0][0]
trial_num = max_trial_num + 1

print('Trial #', trial_num, '\n==============\n')

for repeat_num in range(10):

    all_winnings = []
    output_results = []
    for week, year in zip(set_weeks, set_years):

        salary_cap = 50000
        pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
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

        d_ordering = ['adjust_pos_counts', 'player_drop_multiple', 'matchup_drop', 
                      'top_n_choices', 'full_model_weight', 'covar_type', 'max_team_type',
                      'min_player_same_team', 'min_players_opp_team', 'num_top_players', 
                      'qb_min_iter', 'qb_set_max_team', 'qb_solo_start', 
                      'static_top_players', 'use_ownership', 'own_neg_frac', 
                      'max_salary_remain', 'num_iters']

        d = {'adjust_pos_counts': {True: 0.9552216066203039, False: 0.044778393379696135},
 'player_drop_multiple': {4: 0.3034185794204973,
                          2: 0.3127530113077569,
                          0: 0.3838284092717458},
 'matchup_drop': {1: 0.339499400515529,
                  2: 0.09160956594794034,
                  0: 0.5688910335365307},
 'top_n_choices': {1: 0.23824918294640055,
                   2: 0.11567804477434247,
                   0: 0.646072772279257},
 'full_model_weight': {5: 0.6969851772667797, 0.2: 0.30301482273322033},
 'covar_type': {'no_covar': 0.742707491259824,
                'team_points_trunc': 0.25729250874017595},
                'max_team_type': {'player_points': 1},
 'min_player_same_team': {2: 0.4387686589640436,
                          3: 0.25716311073946735,
                          'Auto': 0.30406823029648905},
 'min_players_opp_team': {1: 0.3973535691796506,
                         2: 0.1841102946623824,
                         'Auto': 0.41853613615796703},
 'num_top_players': {2: 0.4108738640017161,
                     3: 0.4618051682056118,
                     5: 0.1273209677926721},
 'qb_min_iter': {0: 0.959001152916604, 9: 0.04099884708339596},
 'qb_set_max_team': {True: 0.5067779866475088, False: 0.49322201335249116},
 'qb_solo_start': {True: 0.6069897061805674, False: 0.39301029381943264},
 'static_top_players': {True: 0.7654425963061461, False: 0.23455740369385392},
 'use_ownership': {0.9: 0.24782542328528012,
                   0.8: 0.14010327191238073,
                   1: 0.6120713048023392},
 'own_neg_frac': {0.8: 0.9852939468491708, 1: 0.01470605315082918},
 'max_salary_remain': {200: 0.05063923696860667,
                       500: 0.2731977365683341,
                       1000: 0.2631224673711191,
                       1500: 0.41304055909194015},
 'num_iters': {100: 0.7409496716980304, 50: 0.2590503283019696}}
        
        d = {k: d[k] for k in d_ordering}
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
                        full_model_rel_weight, covar_type, max_team_type, min_players_same_team, 
                        min_players_opp_team, num_top_players, qb_min_iter, qb_set_max_team, qb_solo_start,
                        static_top_players, use_ownership, own_neg_frac, max_salary_remain, 
                        num_iters, param_iter
                        ):
            
            try: min_players_opp_team = int(min_players_opp_team)
            except: pass

            try: min_players_same_team = float(min_players_same_team)
            except: pass

            if covar_type=='no_covar': use_covar=False
            else: use_covar=True

            sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                                        pred_vers, ensemble_vers=ensemble_vers, std_dev_type=std_dev_type,
                                        covar_type=covar_type, ownership_vers=ownership_vers,
                                        full_model_rel_weight=full_model_rel_weight, 
                                        use_covar=use_covar, use_ownership=use_ownership, 
                                        salary_remain_max=max_salary_remain)

            winnings = []
            points_record = []
            total_add = []
            to_drop_selected = []

            lineup_pts = pd.DataFrame()
            for t in range(lineups_per_param):

                to_add = []
                to_drop = []
                to_drop.extend(to_drop_selected)
                
                # if matchup_drop > 0: to_drop = rand_drop_teams(matchups, matchup_drop)
                # else: to_drop = []
                # to_drop.extend(to_drop_selected)

                for i in range(9):
                    results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, 
                                            min_players_opp_team_input=min_players_opp_team, max_team_type=max_team_type,
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
        
        # for adj, pdm, md, tn, fmw, ct, mtt, mpst, mpot, ntp, qmi, qsmt, qss, stp, uo, onf, msr, ni, param_i in params:
        #     print([adj, pdm, md, tn, fmw, ct, mtt, mpst, mpot, ntp, qmi, qsmt, qss, stp, uo, onf, msr, ni, param_i])
        #     sim_winnings(adj, pdm, md, tn, fmw, ct, mtt, mpst, mpot, ntp, qmi, qsmt, qss, stp, uo, onf, msr, ni, param_i)
            
    
        par_out = Parallel(n_jobs=-1, verbose=0)(delayed(sim_winnings)(adj, pdm, md, tn, fmw, ct, mtt, mpst, mpot, ntp, qmi, qsmt, qss, stp, uo, onf, msr, ni, param_i) for \
                                                                       adj, pdm, md, tn, fmw, ct, mtt, mpst, mpot, ntp, qmi, qsmt, qss, stp, uo, onf, msr, ni, param_i in params)

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
    output_results['ownership_vers'] = ownership_vers
    output_results['trial_num'] = trial_num
    output_results['repeat_num'] = repeat_num
    dm.write_to_db(output_results, 'Results', 'Entry_Optimize_Results', 'append')

# save out the initial params that were set for randomization
output = param_set_output(d)
dm.write_to_db(output, 'Results', 'Entry_Optimize_Params', 'append')


#%%

to_delete_num=134
df = dm.read(f"SELECT * FROM Entry_Optimize_Lineups WHERE trial_num!={to_delete_num}", 'Results')
dm.write_to_db(df, 'Results', 'Entry_Optimize_Lineups', 'replace')

df = dm.read(f"SELECT * FROM Entry_Optimize_Params WHERE trial_num!={to_delete_num}", 'Results')
dm.write_to_db(df, 'Results', 'Entry_Optimize_Params', 'replace')

df = dm.read(f"SELECT * FROM Entry_Optimize_Params_Detail WHERE trial_num!={to_delete_num}", 'Results')
dm.write_to_db(df, 'Results', 'Entry_Optimize_Params_Detail', 'replace')

df = dm.read(f"SELECT * FROM Entry_Optimize_Results WHERE trial_num!={to_delete_num}", 'Results')
dm.write_to_db(df, 'Results', 'Entry_Optimize_Results', 'replace')


# %%

# df = dm.read(f"SELECT * FROM Entry_Optimize_Params", 'Results')
# add_on = pd.DataFrame({'trial_num': range(df.trial_num.max()+1)})
# add_on = add_on.assign(param='max_team_type', param_option='player_points', option_value=1)
# add_on = add_on[df.columns]

# df = pd.concat([df, add_on], axis=0)
# df = df.sort_values(by='trial_num')

# # df.loc[(df.trial_num.isin([84])) & (df.param=='num_iters'), ['param_option', 'option_value']] = [100, 1]
# dm.write_to_db(df, 'Results', 'Entry_Optimize_Params', 'replace', create_backup=True)

#%%


df = dm.read(f"SELECT * FROM Entry_Optimize_Params_Detail", 'Results')
df['max_team_type'] = 'player_points'
# df['qb_set_max_team'] = False
# df['qb_solo_start'] = False
# df.loc[df.trial_num.isin([84]), 'num_iters'] = 100
dm.write_to_db(df, 'Results', 'Entry_Optimize_Params_Detail', 'replace')

#%%

# for adj, pdm, md, tn, fmw, ct, mpst, mpot, uo, onf, msr, ni, param_i in params:
#     print(param_i)
#     sim_winnings(adj, pdm, md, tn, fmw, ct, mpst, mpot, uo, onf, msr, ni, param_i)

