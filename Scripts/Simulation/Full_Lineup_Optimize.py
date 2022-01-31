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
import sys

for week in [2, 3, 4, 5, 6, 7, 8, 9]:

    print(f'Week {week}\n===============\n')
    year = 2021
    salary_cap = 50000
    pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}
    num_iters = 100

    pred_vers = 'standard'
    covar_type = 'team_points'
    full_model_rel_weight = 1

    adjust_select = True
    TOTAL_LINEUPS = 10
    TEAM_DROP_FRAC = 0.2 # percent of teams to drop from each iteration
    PLAYER_DROP_MULTIPLIER = 0 # higher is more likely to drop previous selected players

    sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, 
                            pred_vers, covar_type, full_model_rel_weight)
    min_players_same_team = 'Auto'
    set_max_team = None

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

    def get_my_results(week):
        path = f'//starbucks/amer/public/CoOp/CoOp831_Retail_Analytics/Pricing/Working/MBorysiak/DK/Results/week{week}.csv'
        my_results = pd.read_csv(path, low_memory=False)
        my_results = my_results.loc[my_results.EntryName.str.contains('mborysi'), 'Points'].values

        actual_winnings = []
        for mr in my_results:
            idx_match = np.argmin(abs(prizes.Points - mr))
            prize_money = prizes.loc[idx_match, 'prize']
            actual_winnings.append(prize_money)

        return actual_winnings, my_results


    def summary_results(winnings, points):
        if len(winnings) != 10:
            frac = 10 / len(winnings)
        else:
            frac = 1
        total_winnings = np.sum(winnings) * frac
        mean_points = np.mean(points)
        max_points = np.max(points)
        max_winnings = np.max(winnings)
        num_placed = len([i for i in winnings if i>0]) * frac

        results = pd.DataFrame([[num_placed, total_winnings, max_winnings, mean_points, max_points]],
                                columns=['NumberPlaced', 'TotalWinnings', 'MaxWinnings', 'MeanPoints', 'MaxPoints'])
        results = round(results, 1)
        return results

    def rand_drop_teams(unique_teams, drop_frac):
        drop_teams = np.random.choice(unique_teams, size=int(drop_frac*len(unique_teams)))
        return list(player_teams.loc[player_teams.team.isin(drop_teams), 'player'].values)


    player_teams = dm.read(f'''SELECT player, team 
                            FROM Covar_Means
                            WHERE week={week}
                                    AND year={year}
                                    AND pred_vers='{pred_vers}'
                                    AND covar_type='{covar_type}'
                                    AND full_model_rel_weight={full_model_rel_weight} ''', 'Simulation')
    unique_teams = player_teams.team.unique()

    points = pd.DataFrame()
    for pos in ['QB', 'RB', 'WR', 'TE', 'Defense']:
        points = pd.concat([points, get_stats(pos)])

    prizes = dm.read(f'''SELECT Rank, Points, prize
                        FROM Million_Results
                        WHERE week={week}
                            AND year={year}''', 'DK_Results')

    min_prize_pts = round(prizes.loc[prizes.prize > 0, 'Points'].min(),1)
    mean_prize_pts = round(prizes.loc[prizes.prize > 0, 'Points'].mean(),1)
    max_prize_pts = round(prizes.loc[prizes.prize > 0, 'Points'].max(),1)
    print(f'Min Prize Points: {min_prize_pts}\nMean Prize Points: {mean_prize_pts}\nMax Prize Points: {max_prize_pts}')

    actual_winnings, actual_points = get_my_results(week)
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
        'drop_team_frac': [0, 0.2],
        'top_n_choices': [0, 4],
        'iters': [0, 1, 2] 
        }

    params = []
    for config in dict_configs(G):
        params.append(list(config.values()))


    sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters, pred_vers, covar_type)
    min_players_same_team = 'Auto'
    set_max_team = None

    def sim_winnings(adjust_select, player_drop_multiplier, team_drop_frac, top_n_choices):

        winnings = []
        points_record = []
        total_add = []
        to_drop_selected = []
        for _ in range(10):
            
            to_add = []
            to_drop = rand_drop_teams(unique_teams, team_drop_frac)
            to_drop.extend(to_drop_selected)

            for i in range(8):
                results, _ = sim.run_sim(to_add, to_drop, min_players_same_team, set_max_team, adjust_select)
                prob = results.loc[i:i+top_n_choices, 'SelectionCounts'] / results.loc[i:i+top_n_choices, 'SelectionCounts'].sum()
                selected_player = np.random.choice(results.loc[i:i+top_n_choices, 'player'], p=prob)
                to_add.append(selected_player)
            
            total_pts, prize_money = calc_winnings(to_add)
            winnings.append(prize_money); points_record.append(total_pts)

            total_add.extend(to_add)
            to_drop_selected = rand_drop_selected(total_add, player_drop_multiplier)

        sim_results = summary_results(winnings, points_record)
        sim_results

        return list(sim_results.values)

#%%
    from joblib import Parallel, delayed
    
    results = Parallel(n_jobs=-1, verbose=10)(delayed(sim_winnings)(adj, pdm, tdf, tn) for adj, pdm, tdf, tn, i in params)
    results = [r[0] for r in results]
    output = pd.concat([pd.DataFrame(params), pd.DataFrame(results)], axis=1)
    print(output)

    output = pd.concat([output, actual_results], axis=1).fillna(method='ffill')
    output = output.assign(min_prize_points=min_prize_pts, mean_prize_points=mean_prize_pts, max_prize_points=max_prize_pts,
                           week=week, year=year, pred_vers=pred_vers, 
                           covar_type=covar_type, full_model_rel_weight=full_model_rel_weight)

    output.columns = ['adjust_pos_counts', 'drop_player_multiple',  'drop_team_frac', 'top_n_choices', 'iter',
                      'lineups_placed', 'total_winnings', 'max_winnings', 'avg_points', 'max_points', 
                      'my_number_placed', 'my_total_winnings', 'my_max_winnings', 'my_mean_points', 'my_max_points',
                      'min_prize_points', 'mean_prize_points', 'max_prize_points',
                      'week', 'year', 'pred_vers', 'covar_type', 'full_model_rel_weight']

    drop_str = f"week={week} AND year={year} AND pred_vers='{pred_vers}' AND covar_type='{covar_type}'AND full_model_rel_weight={full_model_rel_weight}"
    dm.delete_from_db('Simulation', 'Winnings_Optimize', drop_str)
    dm.write_to_db(output, 'Simulation', 'Winnings_Optimize', 'append')

#%%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

df = dm.read("SELECT * FROM Winnings_Optimize", 'Simulation')
X = df[['adjust_pos_counts', 'drop_player_multiple',  'drop_team_frac', 'top_n_choices', 
        'week', 'pred_vers', 'covar_type', 'full_model_rel_weight']]

def one_hot(X):
    for c in ['week', 'pred_vers', 'covar_type']:
        X = pd.concat([X, pd.get_dummies(X[c], prefix=c, drop_first=True)], axis=1).drop(c, axis=1)
    return X

X = one_hot(X)
sc = StandardScaler()
sc.fit(X)
X_scale = sc.transform(X)
y = df.total_winnings

lr = LinearRegression()
lr.fit(X_scale,y)
print(lr.score(X_scale,y))
pd.Series(lr.coef_, index=X.columns).sort_values().plot.barh()

# %%

X_pred = X.iloc[0].copy()
X_pred.adjust_pos_counts = 1
X_pred.drop_player_multiple = 0
X_pred.drop_team_frac = 0
X_pred.top_n_choices = 4
X_pred.week_18 = 1
X_pred = pd.DataFrame(X_pred).T

print('Optimal Avg Winnings:', lr.predict(sc.transform(X_pred))[0])

my_avg_winnings = dm.read("SELECT DISTINCT week, year, my_total_winnings FROM Winnings_Optimize", 'Simulation').my_total_winnings.mean()
print('My Avg Winnings:', my_avg_winnings)
# %%
