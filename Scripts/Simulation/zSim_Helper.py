#%%

# core packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas.io.formats.style
import copy
import os 

# sql packages
import sqlite3

# linear optimization
from cvxopt import matrix
from cvxopt.glpk import ilp
from scipy.stats import skewnorm

class FootballSimulation():

    #==========
    # Creating Player Distributions for Given Settings
    #==========
    
    def __init__(self, conn_sim, set_year, week, iterations):
        
        # create empty dataframe to store player point distributions
        pos_update = {'QB': 'aQB', 'RB': 'bRB', 'WR': 'cWR', 'TE': 'dTE', 'Defense': 'fDEF'}
        self.data = pd.read_sql_query(f'''SELECT * FROM week{week}_year{set_year}''', conn_sim)
        self.data['pos'] = self.data['pos'].map(pos_update)

        # add salaries to the dataframe and set index to player
        salaries = pd.read_sql_query(f'''SELECT player, salary
                                         FROM Salaries
                                         WHERE year={set_year}
                                               AND league={week} ''', conn_sim)

        self.sal = salaries
        self.data = pd.merge(self.data, salaries, how='left', left_on='player', right_on='player')
        self.data.salary = self.data.salary.fillna(3000)
        
        # add flex data
        flex = self.data[self.data.pos.isin(['bRB', 'cWR', 'dTE'])]
        flex.loc[:, 'pos'] = 'eFLEX'
        self.data = pd.concat([self.data, flex], axis=0)
        
        # reset index
        self.data = self.data.sort_values(by=['pos', 'salary'], ascending=[True, False])
        self.data = self.data.set_index('player')

    
    def return_data(self):
        '''
        Returns self.data if necessary.
        '''
        return self.data
    
    
    #==========
    # Running the Simulation for Given League Settings and Keepers
    #==========
    
    def run_simulation(self, league_info, to_drop, to_add, iterations=500):        
        '''
        Method that runs the actual simulation and returns the results.
        
        Input: Projected player data and salaries with variance, the league 
               information (e.g. position requirements, and salary caps), and 
               information about players selected by your team and other teams.
        Returns: The top team results (players selected and salaries), as well
                 as counts of players selected, their salary they were selected at,
                 and the points the scored when selected.
        '''
        #--------
        # Pull Out Data, Salaries, and Inflation for Current Simulation
        #--------
        
        # create a copy of self.data for current iteration settings
        data = self.data.copy()
        league_info = copy.deepcopy(league_info)
       
        # drop other selected players + calculate inflation metrics
        data, drop_proj_sal, drop_act_sal = self._drop_players(data, league_info, to_drop)
        
        # drop your selected players + calculate inflation metrics
        data, league_info, to_add, add_proj_sal, add_act_sal = self._add_players(data, league_info, to_add)
        pos_require = list(league_info['pos_require'].values())
        
        # calculate inflation based on drafted players and their salaries
        inflation = 1

        # determine salaries, skew distributions, and number of players for each position
        data, salaries, pos_counts = self._pull_salary_poscounts(data)
        data = data.drop(['pos', 'salary'], axis=1)

        #--------
        # Initialize Matrix and Results Dictionary for Simulation
        #--------
        
        # generate the A matrix for the simulation constraints
        A = self._Amatrix(pos_counts, league_info['pos_require'])

        # pull out the names of all players and set to names
        names = data.index
        dict_names = list(data.index)
        dict_names.extend(to_add['players'])
        
        # create empty matrices
        results = {}
        results['names'] = []
        results['points'] = []
        results['salary'] = []

        # create empty dictionaries
        counts = {}
        counts['names'] = pd.Series(0, index=dict_names).to_dict()
        counts['points'] = pd.Series(0, index=dict_names).to_dict()
        counts['salary'] = pd.Series(0, index=dict_names).to_dict()
        
        # shuffle the random data--both salary skews and the point projections
        data = self._df_shuffle(data)
                
        #--------
        # Run the Simulation Loop
        #--------
            
        trial_counts = 0
        for i in range(0, iterations):
    
            # every N trials, randomly shuffle each run in salary skews and data
            if i % (iterations / 10) == 0:
                data = self._df_shuffle(data)
            
            # pull out a random selection of points and salaries
            points, salaries_tmp = self._random_select(data, salaries)

            # run linear integer optimization
            x = self._run_opt(A, points, salaries_tmp, league_info['salary_cap'], pos_require)

            # pull out and store the selected names, points, and salaries
            results, self.counts, trial_counts = self._pull_results(x, names, points, salaries_tmp, 
                                                                    to_add, results, counts, trial_counts)
        
        # format the results after the simulation loop is finished
        self.results = self._format_results(results)

        # add self version of variable for output calculations
        self._inflation = inflation
        self._sal = self.data.reset_index()[['player', 'salary']].drop_duplicates().set_index('player')

        return self.counts, inflation, 
    
    #==========
    # Helper Functions for the Simulation Loop
    #==========

    #--------
    # Salary (+Inflation) and Keeper Setup
    #--------
    
    def add_salaries(self, salaries):
        '''
        Input: Salaries for all players in the dataset.
        Return: The self.data dataframe that has salaries appended to it.
        '''
        #--------
        # Merge salaries and points on names to ensure matches
        #--------
        
        # merge the salary and prediction data together on player
        self.data = pd.merge(self.data, salaries, how='inner', left_on='player', right_on='player')
        
        # sort values and move player to the index of the dataframe
        self.data = self.data.sort_values(by=['pos', 'salary'], ascending=[True, False]).set_index('player')
    
    
    @staticmethod
    def _drop_players(data, league_info, to_drop):
        '''
        Drops a list of players that are chosen as by other teams and calculates actual 
        salary vs. expected salary for inflation calculation.
        
        Input: Data for a given simulation run, league information (e.g. total salary cap), 
               and a dictionary of players with their salaries to drop. 
        Return: The players that remain available for the simulation, along with metrics
                for salary inflation.
        '''
        
        #--------
        # Dropping Other Team's Players
        #--------
        
        # find players from data that will be dropped and remove them from other data
        drop_data = data[data.index.isin(to_drop['players'])]
        other_data = data.drop(drop_data.index, axis=0)
        
        # pull out the projected and actual salaries for the players that are being kept
        drop_proj_salary = drop_data.salary.reset_index().drop_duplicates().sum().salary
        drop_act_salary = np.sum(to_drop['salaries'])

        return other_data, drop_proj_salary, drop_act_salary
    
    
    @staticmethod
    def _add_players(data, league_info, to_add):
        '''
        Removes a list of players that are chosen as to_add and calculates inflation based
        on their added salary vs. expected salary.
        
        Input: Data for a given simulation run, league information (e.g. total salary cap), 
               and a dictionary of players with their salaries to keep. 
        Return: The players that remain available for the simulation, the players to be kept,
                and metrics to calculate salary inflation.
        '''
        
        # pull data for players that have been added to your team and split out other players
        add_data = data[data.index.isin(to_add['players'])]
        other_data = data.drop(add_data.index, axis=0)

        # pull out the salaries of your players and sum
        add_proj_salary =  add_data.salary.reset_index().drop_duplicates().sum().salary
        add_act_salary = np.sum(to_add['salaries'])

        # update the salary for your team to subtract out drafted player's salaries
        league_info['salary_cap'] = float(league_info['salary_cap'] - add_act_salary)
        print('Remaining Salary:', league_info['salary_cap'])
        
        # add the mean points scored by the players who have been added
        to_add['points'] = -1.0*(add_data.drop(['pos', 'salary'],axis=1).mean(axis=1).values)
        
        # create list of letters to append to position for proper indexing
        letters = ['a', 'b', 'c', 'd', 'e', 'f']

        # loop through each position in the pos_require dictionary
        for i, pos in enumerate(league_info['pos_require'].keys()):

            # create a unique label based on letter and position
            pos_label = letters[i]+pos

            # loop through each player that has been selected  
            for player in list(add_data[add_data.pos==pos_label].index):

                # if the position is still required to be filled:
                if league_info['pos_require'][pos] > 0:

                    # subtract out the current player from the position count
                    league_info['pos_require'][pos] = league_info['pos_require'][pos] - 1

                    # and remove that player from consideration for filling other positions
                    add_data = add_data[add_data.index != player]
        
        print('Remaining Positions Required:', league_info['pos_require'])
        return other_data, league_info, to_add, add_proj_salary, add_act_salary
    
    
    @staticmethod
    def _calc_inflation(league_info, drop_proj_sal, drop_act_sal, add_proj_sal, add_act_sal):
        '''
        Method to calculate inflation based on players selected and the expected salaries.
        '''
        # add up the total actual and projected salaries for all keepers
        projected_salary = drop_proj_sal + add_proj_sal
        actual_salary = drop_act_sal + add_act_sal
        
        # calculate the salary inflation due to the keepers
        total_cap = league_info['num_teams'] * league_info['initial_cap']
        inflation = (total_cap-actual_salary) / (total_cap-projected_salary)
        
        return inflation, total_cap
        
        
    def _pull_salary_poscounts(self, data):
        '''
        Method to pull salaries from the data dataframe, create salary skews, and determine
        the position counts for the A matrix in the simulation
        
        Input: Data for current simulation and inflation metric
        Return: The data without salary column, the inflated salary numpy array, a dataframe of salaru
                skews for current simulation, and a count of positions in the dataframe 
        '''
        #--------
        # Extract salaries into numpy array and drop salary from points data
        #--------

        # set salaries to numpy array and multiply by inflation
        # CHANGE 2021-07-06: Inflation removed since predicted salaries account for it
        salaries = data.salary.values

        # extract the number of counts for each position for later creating A matrix
        pos_counts = list(data.pos.value_counts().sort_index())
        
        return data, salaries, pos_counts
        
        
    #--------
    # Setting up and Running the Simulation
    #--------
    
    @staticmethod
    def _Amatrix(pos_counts, pos_require):
        '''
        This function creates the A matrix that is critical for the ILP solution being equal
        to the positional constraints specified. I identified the given pattern empirically:
        1. Repeat the vector [1, 0, 0, 0, ...] N times for each player for a given position.
           The number of trailing zeros is equal to p-1 positions to draft.
        2. After the above vector is repeated N times for a given player, append a 0 before
           repeating the same pattern for the next player. Repeat for all players up until the 
           last position.
        3. for the last poition, repeat the pattern N-1 times and append a 1 at the end.
        This pattern allows the b vector, e.g. [1, 2, 2, 1] to set the constraints on the positions
        selected by the ILP solution.
        '''
        
        #--------
        # Initialize the Vector Pattern and Matrix
        #--------
        
        # create A matrix
        vec = [1]
        vec.extend([0]*(len(pos_require)-1))
        
        # intialize A matrix by multiplying length one by vec and appending 0 to start pattern
        A = pos_counts[0]*vec
        A.append(0)

        #--------
        # Repeat the Pattern Until Last Position
        #--------
        
        # repeat the same pattern for the inner position requirements
        for i in range(1, len(pos_counts)-1):

            A.extend(pos_counts[i]*vec)
            A.append(0)

        #--------
        # Finish the Pattern for the Last Position
        #--------
        
        # adjust the pattern slightly for the final position requirement
        A.extend((pos_counts[-1]-1)*vec)
        A.append(1)

        # convert A into a matrix for integer optimization
        A = matrix(A, size=(len(vec), np.sum(pos_counts)), tc='d')

        return A
    
    
    @staticmethod
    def _df_shuffle(df):
        '''
        Input: A dataframe to be shuffled, row-by-row indepedently.
        Return: The same dataframe whose columns have been shuffled for each row.
        '''
        # store the index before converting to numpy
        idx = df.index
        df = df.values

        # shuffle each row separately, inplace, and convert o df
        _ = [np.random.shuffle(i) for i in df]

        return pd.DataFrame(df, index=idx)
    
    
    @staticmethod
    def _run_opt(A, points, salaries, salary_cap, pos_require):
        '''
        This function sets up and solves the integer Linear Programming problem 
        c = n x 1 -- c is the vector of points to be optimized
        G = m x n -- G is the salaries of the corresponding players / points (m=1 in this case)
        h = m x 1 -- h is the salary cap (m=1 in this case)
        A = p x n -- A sparse binary matrix that must be developed so b equals player constraints
        b = p x 1 -- b is a vector with player requirements, e.g. [QB, RB, WR] = [1, 2, 2]

        Solve:
        c'*n -- minimize

        Subject to:
        G*x <= h
        A*x = b
        '''
        
        # generate the c matrix with the point values to be optimized
        c = matrix(points, tc='d')

        # generate the G matrix that contains the salary values for constraining
        G = matrix(salaries, tc='d').T

        # generate the h matrix with the salary cap constraint
        h = matrix(salary_cap, size=(1,1), tc='d')

        # generate the b matrix with the number of position constraints
        b = matrix(pos_require, size=(len(pos_require), 1), tc='d')

        # solve the integer LP problem
        (status, x) = ilp(c, G, h, A=A, b=b, B=set(range(0, len(points))))

        return x
    
    @staticmethod
    def _random_select(data, salaries):
        '''
        Random column selection for trial in simulation
        
        Input: Data, salaries, and salary skews
        Return: Randomly selected array of points and salaries + skews for a given trial
        '''
        # select random number between 0 and sise of distributions
        ran_num = random.randint(2, 1000-1)

        # pull out a random column of points and convert to points per game
        ppg = data.iloc[:, ran_num].values.astype('double')
        ppg = -1.*ppg
        
        # pull out a random skew and add to the original salaries
        salaries_tmp = salaries
        salaries_tmp = salaries_tmp.astype('double')
        
        return ppg, salaries_tmp
    
    
    #==========
    # Formatting and Displaying All Results
    #==========
    
    @staticmethod
    def _pull_results(x, names, points, salaries, to_add, results, counts, trial_counts):
        '''
        This method pulls in each individual result from the simulation loop and stores it in dictionaries.
        
        Input: Names, points, and salaries for the current simulation lineup.
        Return: Dictionaries with full results and overall simulation counts, continuously updated.
        '''
        # find all LP results chosen and equal to 1
        x = np.array(x)[:, 0]==1

        if len(names[x]) != len(np.unique(names[x])):
            return results, counts, trial_counts
        
        trial_counts += 1
        
        names_ = list(names[x])
        names_.extend(to_add['players'])
        
        points_ = list(points[x])
        points_.extend(to_add['points'])
        
        salaries_ = list(salaries[x])
        salaries_.extend(to_add['salaries'])
        
        for i, p in enumerate(names_):

            counts['names'][p] += 1

            if counts['points'][p] == 0:
                counts['points'][p] = []
            counts['points'][p].append(points_[i])

            if counts['salary'][p] == 0:
                counts['salary'][p] = []
            counts['salary'][p].append(salaries_[i])

        # pull out the corresponding names, points, and salaries for chosen players
        # to append to the higher level results dataframes
        results['names'].append(names_)
        results['points'].append(points_)
        results['salary'].append(salaries_)

        return results, counts, trial_counts
    
    
    @staticmethod
    def _format_results(results):
        '''
        After the simulation loop, this method pulls results from the dictionary and formats
        into dataframes.
        
        Input: The results dictionary with all results
        Return: A formatted dataframe with all results
        '''
        
        # create dataframes for the names of selected players, their points scored, and salaries
        name_results = pd.DataFrame(results['names'])
        point_results = pd.DataFrame(results['points'])*-1
        total_points = point_results.sum(axis=1)
        salary_results = pd.DataFrame(results['salary'])
        total_salary = salary_results.sum(axis=1)
        
        # concatenate names, points, and salaries altogether
        results_df = pd.concat([name_results, total_points, total_salary, point_results, salary_results], axis=1)
        
        # rename columns to numbers
        results_df.columns = range(0, results_df.shape[1])
        
        # find the first numeric column that corresponds to points scored and sort by that column
        first_num_col = results_df.dtypes[results_df.dtypes=='float64'].index[0]
        results_df = results_df.sort_values(by=first_num_col, ascending=False)

        return results_df

    #===============
    # Creating Output Visualizations
    #===============
    
    
    def density_plot(self, player):
        '''
        Creates density player showing points scored and salary selected for a given player
        '''
        
        # pull out points and salary for a given player
        sal = np.array(self.counts['salary'][player])
        
        # create and displayjoint distribution plot
        sns.distplot(sal)
        plt.show()

                
    def show_most_selected(self, to_add, iterations, num_show=20):
        '''
        Input: Dictionary containing all the values (counts, salaries, points) of the various
               iterations from the simulation.
        Output: Formatted dataframe that can be printed with pandas bar charts to visualize results.
        '''
        #----------
        # Create data frame of percent drafted and average salary
        #----------

        # create a dataframe of counts drafted for each player
        counts_df = pd.DataFrame.from_dict(self.counts['names'], orient='index').rename(columns={0: 'Percent Drafted'})
        counts_df = counts_df.sort_values(by='Percent Drafted', 
                                    ascending=False)[len(to_add['players']):].head(num_show) / iterations

        # pull out the salary that each player was drafted at from dictionary
        avg_sal = {}
        for key, value in self.counts['salary'].items():
            avg_sal[key] = np.mean(value)

        # pass the average salaries into datframe and merge with the counts dataframe
        avg_sal = pd.DataFrame.from_dict(avg_sal, orient='index').rename(columns={0: 'Average Salary'})
        avg_sal = pd.merge(counts_df, avg_sal, how='inner', left_index=True, 
                        right_index=True).sort_values(by='Percent Drafted', ascending=False)
        
        # pull in the list salary + inflation to calculate drafted salary minus expected
        avg_sal = pd.merge(avg_sal, self._sal, how='inner', left_index=True, right_index=True)
        avg_sal.salary = (avg_sal['Average Salary'] - avg_sal.salary * self._inflation)
        avg_sal = avg_sal.rename(columns={'salary': 'Expected Salary Diff'})

        # format the result with rounding
        avg_sal.loc[:, 'Percent Drafted'] = round(avg_sal.loc[:, 'Percent Drafted'] * 100, 1)
        avg_sal.loc[:, 'Average Salary'] = round(avg_sal.loc[:, 'Average Salary'], 1)
        avg_sal.loc[:, 'Expected Salary Diff'] = round(avg_sal.loc[:, 'Expected Salary Diff'], 1)

        return avg_sal


# %%

# # connection for simulation and specific table
# path = f'c:/Users/{os.getlogin()}/Documents/Github/Daily_Fantasy/'
# conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
# set_year = 2020
# league=15

# # number of iteration to run
# iterations = 1000

# # define point values for all statistical categories
# pass_yd_per_pt = 0.04 
# pass_td_pt = 4
# int_pts = -2
# sacks = -1
# rush_yd_per_pt = 0.1 
# rec_yd_per_pt = 0.1
# rush_rec_td = 7
# ppr = .5

# # creating dictionary containing point values for each position
# pts_dict = {}
# pts_dict['QB'] = [pass_yd_per_pt, pass_td_pt, rush_yd_per_pt, rush_rec_td, int_pts, sacks]
# pts_dict['RB'] = [rush_yd_per_pt, rec_yd_per_pt, ppr, rush_rec_td]
# pts_dict['WR'] = [rec_yd_per_pt, ppr, rush_rec_td]
# pts_dict['TE'] = [rec_yd_per_pt, ppr, rush_rec_td]

# # instantiate simulation class and add salary information to data
# sim =  FootballSimulation(conn_sim, set_year, league, iterations)

# # set league information, included position requirements, number of teams, and salary cap
# league_info = {}
# league_info['pos_require'] = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DEF': 1}
# league_info['num_teams'] = 12
# league_info['initial_cap'] = 50000
# league_info['salary_cap'] = 50000

# to_drop = {}
# to_drop['players'] = []
# to_drop['salaries'] = []

# # input information for players and their associated salaries selected by your team
# to_add = {}
# to_add['players'] = ['Davante Adams', 'Mark Andrews', 'Aaron Jones', 'Ceedee Lamb',
#                      'Kj Hamler', 'Kareem Hunt', 'Josh Allen', 'Darius Slayton', 
#                      'CLE']
# to_add['salaries'] = [9400, 5500, 7300, 4500, 3500, 5600, 7200, 4000, 3000]

# _, _ = sim.run_simulation(league_info, to_drop, to_add, iterations=iterations)
# sim.show_most_selected(to_add, iterations, num_show=30)
# # %%

# from ff.db_operations import DataManage
# from ff import general as ffgeneral

# # set the root path and database management object
# root_path = ffgeneral.get_main_path('Daily_Fantasy')
# db_path = f'{root_path}/Data/Databases/'
# dm = DataManage(db_path)

# set_year = 2020
# league=15

# to_add_tuple = tuple(['Davante Adams', 'Mark Andrews', 'Aaron Jones', 'Ceedee Lamb',
#                       'Kj Hamler', 'Kareem Hunt', 'Josh Allen', 'Darius Slayton', 'CLE'])

# # to_add_tuple = tuple(['Aaron Jones', 'Rob Gronkowski', 'Mike Evans', 'PIT', 'Aj Brown',
# #                       'Michael Gallup', 'Alvin Kamara', 'Matt Ryan', 'Russell Gage'])

# # to_add_tuple = tuple(['Mark Andrews', 'Derrick Henry', 'Michael Gallup', 'Alvin Kamara',
# #                      'Jamison Crowder', 'Ty Hilton', 'Josh Allen', 'Darius Slayton', 'LAR'])

# # to_add_tuple = tuple(['Aaron Jones', 'Rob Gronkowski', 'Deandre Hopkins',
# #                       'PIT', 'Emmanuel Sanders', 'Melvin Gordon', 'Deshaun Watson',
# #                       'Corey Davis', 'Mark Andrews'])

# # to_add_tuple = tuple(['Darren Waller', 'Amari Cooper', 'Chris Carson', 'Cooper Kupp', 
# #                      'Drew Lock', 'Kj Hamler', 'Alvin Kamara', 'Ty Hilton', 'SEA'])

# actuals = dm.read(f'''SELECT * FROM (
#                      SELECT player, y_act, week, year
#                      FROM QB_Data
#                      UNION
#                      SELECT player, y_act, week, year
#                      FROM RB_Data
#                      UNION
#                      SELECT player, y_act, week, year
#                      FROM WR_Data
#                      UNION
#                      SELECT player, y_act,  week, year
#                      FROM TE_Data
#                      UNION
#                      SELECT player, y_act,  week, year
#                      FROM Defense_Data)
#                      WHERE player in {to_add_tuple}
#                            AND week={league}
#                            AND year={set_year}
#                  ''', 'Model_Features')
# actuals
# # %%
