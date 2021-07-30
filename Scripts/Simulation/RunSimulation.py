
#%%

# import sim functions
from zSim_Helper import *
import seaborn as sns
from IPython.core.pylabtools import figsize

#===============
# Settings and User Inputs
#===============

np.random.seed(1234)

#--------
# League Settings
#--------

# connection for simulation and specific table
path = f'c:/Users/{os.getlogin()}/Documents/Github/Daily_Fantasy/'
conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
set_year = 2020
league=15

# number of iteration to run
iterations = 1000

# set league information, included position requirements, number of teams, and salary cap
league_info = {}
league_info['pos_require'] = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DEF': 1}
league_info['num_teams'] = 12
league_info['initial_cap'] = 50000
league_info['salary_cap'] = 50000

flex_pos = ['RB', 'WR', 'TE']

#==================
# Initialize the Simluation Class
#==================

# instantiate simulation class and add salary information to data
sim = FootballSimulation(conn_sim, set_year, league, iterations)

# return the data and set up dataframe for proportion of salary across position
d = sim.return_data()
d = d.rename(columns={'pos': 'Position', 'salary': 'Salary'})
d.Position = d.Position.apply(lambda x: x[1:])

if league == 'nv':
    d.loc[(d.Position=='QB') & (d.Salary==1), 'Salary'] = 3

# pull out the positional salary and rank order in to cut out top players from flex
proport = d.loc[:, ['Position', 'Salary']].reset_index()
proport = proport.sort_values(by=['Position', 'Salary'], ascending=[True, False])
proport['PosCnts'] = proport.groupby('Position').cumcount() + 1

# determine the number of players required at each position
league_req = pd.DataFrame([league_info['pos_require']]).T.reset_index()
league_req.columns =['Position', 'NumReq']
league_req.NumReq = league_req.NumReq * league_info['num_teams']

# merge the number of players required for each pos to cut ot of flex options
proport = pd.merge(proport, league_req, on='Position')
remove_flex = proport.loc[(proport.PosCnts <= proport.NumReq) & (proport.Position.isin(flex_pos)), ['player']]
proport = proport[~((proport.Position=='FLEX') & (proport.player.isin(remove_flex.player)))]

# redetermine the position counts by salary after removing top players to filter to starters
proport['PosCnts'] = proport.groupby('Position').cumcount() + 1
proport = proport[proport.PosCnts <= proport.NumReq].reset_index(drop=True)

# get the proportion of salary by position
proport = proport.groupby('Position').agg({'Salary': 'mean'})
proport['total'] = proport.Salary.sum()
proport['Wts'] = proport.Salary / proport.total
proport = proport.reset_index()
proport = proport[['Position', 'Wts']]

#------------------
# For Beta Keepers
#------------------

# input information for players and their associated salaries selected by other teams
keepers = {}

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html 
import plotly.express as px
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

# set up dash with external stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


#==================
# Expected Points Functions
#==================

def csv_check(df, name):
    return df.to_csv(f'c:/Users/mborysia/Desktop/FF_App/{name}.csv', index=False)


def expected_pts_model(df):
    '''
    Linear Regression model that predicts points based on Position and Salary
    using the sim.return_data() dataset
    '''
    # create dataset that models Pts and Std of Pts based on Salary + Position
    Xy = pd.concat([df.Position, df.Salary, df.iloc[:, 1:999].mean(axis=1), df.iloc[:, 1:999].std(axis=1)], axis=1).reset_index(drop=True)
    Xy = pd.concat([pd.get_dummies(Xy.Position), Xy.Salary, Xy[0], Xy[1]], axis=1)
    Xy.Salary = np.log(Xy.Salary)

    # create X and y variables for training 
    X_train = Xy.dropna().drop([0, 1], axis=1)
    y_mean = Xy[0].dropna()
    y_std = Xy[1].dropna()

    # fit the linear regression models
    from sklearn.linear_model import LinearRegression
    lr_mean = LinearRegression()
    lr_std = LinearRegression()
    lr_mean.fit(X_train, y_mean)
    lr_std.fit(X_train, y_std)

    # save out the template for creating X_test
    cols = X_train.columns
    test_template = pd.DataFrame([len(cols)*[None]], columns=cols)

    return lr_mean, lr_std, test_template


def salary_proportions(my_team, proport, remaining_sal):
    '''
    Function that determines the amount of money to be spread across remaining positions
    '''
    # determine who is already selected and get their points
    selected_list = list(my_team.loc[~(my_team.Player.isnull()) | ~(my_team.Player==''), 'Player'].values)
    selected_df = d[(d.index.isin(selected_list)) & (d.Position!='FLEX')].drop(['Position', 'Salary'], axis=1)
    selected_df.columns = [int(c) for c in selected_df.columns]

    # get the remaining positions from my team and merge with proportion of salary to each
    remain = my_team.loc[(my_team.Player.isnull()) | (my_team.Player==''), 'Position']
    remain = pd.merge(remain, proport, on='Position')

    # readjust the weights to equal 1
    remain['TotalWt'] = remain.Wts.sum()
    remain['AdjustWt'] = remain.Wts / remain.TotalWt

    # multiply the weights for each position by remaining dollars
    remain['Salary'] = remain.AdjustWt * remaining_sal
    remain = remain[['Position', 'Salary']]

    return remain, selected_df


def create_X_test(df, template):
    '''
    Input: Dataframe with Position and Salary columns
           Template with proper one-hot-encoding of positions

    Output: The data from the dataframe formatted properly for modeling based on template
    '''
    df.Salary = df.Salary.astype('int')
    df = pd.concat([pd.get_dummies(df.Position), np.log(df.Salary)], axis=1)
    return pd.concat([template, df], axis=0).dropna(how='all').fillna(0)


def create_pt_dist(X_test, selected, lr_mean, lr_std):
    '''
    Input: LR models + X_test of remaining positions + salary to predict,
           along with the selected player distributions thus far
    
    Output: A dataframe containing distribution of player points
    '''
    # generate model fit
    if len(X_test) > 0:
        
        remain_means = lr_mean.predict(X_test)
        remain_std = lr_std.predict(X_test)
        
        team_dist = pd.DataFrame()
        team_dist = pd.concat([team_dist, selected], axis=0)
        
        np.random.seed(1234)
        for m, s in zip(remain_means, remain_std):
            team_dist = pd.concat([team_dist, pd.DataFrame(np.random.normal(m, s, 1000)).T], axis=0)
    
    else:
        team_dist = selected

    team_dist = team_dist.sum(axis=0)
    team_dist = team_dist * (11/16) + 140 + 250
    team_dist = pd.DataFrame(team_dist, columns=['projection'])

    return team_dist


#==========================
# Submit Data Entry
#==========================

# submit button
red_button_style = {'background-color': 'red',
                    'color': 'white',
                    'height': '20',
                    'width': '100%',
                    'fontSize': 16}
submit_button = html.Button(id='submit-button-state', n_clicks=0, 
                            children='Refresh Top Picks', style=red_button_style)


#============================
# Build out Dash Tables
#============================

main_color = '40, 110, 132'
main_color_rgb = f'rgb({main_color})'
main_color_rgba = f'rgba({main_color}, 0.8)'

#--------------
# Set up dataframe and Data Table for My Team
#--------------

player_list = []
for pl, row in d.sort_values(by='Salary', ascending=False)[['Salary', 'Position']].iterrows():
    if row.Position != 'FLEX':
        player_list.append([row.Position, pl, row.Salary, 0])

pick_df = pd.DataFrame(player_list, columns=['Position', 'Player', 'List Salary', 'Salary'])
pick_df['My Team'] = 'No'

for p, s in keepers.items():
    pick_df.loc[pick_df.Player==p, 'Salary'] = s

# set up all players drafted DataTable
drafted_player_table =  dash_table.DataTable(
                            id='draft-results-table',

                            columns=[{'id': c, 'name': c, 'editable': (c == 'Salary')} for c in pick_df.columns if c != 'My Team'] +
                                     [{'id': c, 'name': c,'presentation': 'dropdown', 'editable': (c == 'My Team')} for c in pick_df.columns if c == 'My Team'],
                            data=pick_df.to_dict('records'),
                            filter_action='native',
                            sort_action='native',
                            style_table={
                                            'height': '800px',
                                            'overflowY': 'auto',
                                        },
                            style_cell={'textAlign': 'left', 'fontSize':14, 'font-family':'sans-serif'},
                            dropdown={
                                    'My Team': {
                                        'options': [
                                            {'label': i, 'value': i} for i in ['Yes', 'No']
                                        ]
                                    }
                                    # 'style': {'backgroundColor': 'white', 'color': 'black'}
                                    },
                            style_data_conditional=[{
                                        'if': {'column_editable': False},
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                    }],
                            style_header={
                                        'backgroundColor': main_color_rgb,
                                        'fontWeight': 'bold',
                                        'color': 'white'
                                    }
                        )


#--------------
# Set up dataframe and Data Table for My Team
#--------------

my_team_list = []
for k, v in league_info['pos_require'].items():
    for i in range(v):
        my_team_list.append([k, None, 0, 0])
my_team_df = pd.DataFrame(my_team_list, columns=['Position', 'Player', 'Salary', 'Points Added'])

 # set up my team  drafted DataTable
my_team_table =  dash_table.DataTable(
                            id='my-team-table',
                            columns=[{"name": i, "id": i, 'editable': False} for i in my_team_df.columns],
                            data=my_team_df.to_dict('records'),
                            style_cell={'textAlign': 'left', 'fontSize':14, 'font-family':'sans-serif'},
                            style_header={
                                        'backgroundColor': main_color_rgb,
                                        'fontWeight': 'bold',
                                        'color': 'white'
                                    },
                            style_data_conditional=[{
                                        'if': {'column_editable': False},
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                    }],
                        )

#--------------
# Set up dataframe and Data Table for Team Info
#--------------

team_info = pd.DataFrame({'Mean Points': [None], 
                          'Inflation': [1], 
                          'Remaining Salary': [league_info['salary_cap']]})

team_info_table =  dash_table.DataTable(
                            id='team-info-table',
                            columns=[{"name": i, "id": i, 'editable': False} for i in team_info.columns],
                            data=team_info.to_dict('records'),
                            style_cell={'textAlign': 'center', 'fontSize':14, 'font-family':'sans-serif'},
                            style_header={
                                        'backgroundColor': main_color_rgb,
                                        'fontWeight': 'bold',
                                        'color': 'white'
                                    },
                            style_data_conditional=[{
                                        'if': {'column_editable': False},
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                    }],
                        )


# # testing point distribution functions from above
lr_mean, lr_std, test_template = expected_pts_model(d)
# remain_init, selected = salary_proportions(my_team_df, proport, league_info['salary_cap'])
# X_init = create_X_test(remain_init, test_template)
# team_dist = create_pt_dist(X_init, selected, lr_mean, lr_std)

#==========================
# Plotting Functions
#==========================

def create_bar(x_val, y_val, orient='h', color_str=main_color_rgba, text=None):
    '''
    Function to create a horizontal bar chart
    '''
    marker_set = dict(color=color_str, line=dict(color=color_str, width=1))
    return go.Bar(x=x_val, y=y_val, marker=marker_set, orientation=orient, text=text, showlegend=False)


def create_fig_layout(fig1, fig2, fig3):
    '''
    Function to combine bar charts into a single figure
    '''
    fig = go.Figure(data=[fig1, fig2, fig3])
    
    # Change the bar mode
    fig.update_layout(barmode='group', autosize=True, height=1600, 
                      margin=dict(l=0, r=25, b=0, t=15, pad=0),
                      uniformtext_minsize=25, uniformtext_mode='hide')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    return fig

def create_hist(team_dist):
    
    hist_data = [team_dist.projection.values]
    group_labels = [''] # name of the dataset

    fig_hist = ff.create_distplot(hist_data, group_labels, bin_size=10, show_rug=False, colors=[main_color_rgba])
    fig_hist.update_layout(autosize=True, height=200, margin=dict(l=0, r=0, b=0, t=0, pad=0), showlegend=False)

    return fig_hist


#--------------
# Plot Creation
#--------------

# bar chart creation
bar_gr = dcc.Graph(id='draft-results-graph')

# histogram creation
hist_gr = dcc.Graph(id='team-points-graph')


#============================
# Build out App Layout
#============================

app.layout = html.Div([

     html.Div([
         html.Div([
            html.H5("Enter Draft Pick Information"),
            drafted_player_table,
            ], className="four columns"),

            html.Div([
                html.H5('My Team'),
                my_team_table, html.Hr(),
                submit_button, html.Hr(),
                html.H5('Team Information'),
                team_info_table,
                html.Hr(),
                hist_gr
            ], className='four columns'),

            html.Div([
                html.H5('Recommended Picks'),
                bar_gr
            ], className="four columns")
       
       ], className="row2") ,        
         
])


#============================
# Update Functions
#============================

def update_to_drop(df):
    '''
    INPUT: Dataframe containing players + salaries to be dropped from selection

    OUTPUT: Dictionary containing dropped player + salaries for passing into simulation
    '''
    to_drop = {}
    to_drop['players'] = []
    to_drop['salaries'] = []
    for _, row in df.iterrows():
        if row.Salary > 0:
            to_drop['players'].append(row.Player)
            to_drop['salaries'].append(row.Salary)

    return to_drop


def update_to_add(df):
    '''
    INPUT: Dataframe containing players + salaries to be added to my team

    OUTPUT: Dictionary containing my team player + salaries for passing into simulation
    '''
    to_add = {}
    to_add['players'] = []
    to_add['salaries'] = []
    for _, row in df.iterrows():
        if row.Player is not None and row.Player!='' and row.Salary > 0:
            to_add['players'].append(row.Player)
            to_add['salaries'].append(row.Salary)

    return to_add


def expected_pts(team_template, my_team_select, test_template, sal):
    '''
    INPUT: Blank team template dataframe, dataframe of selected players, and the X_test
           template the auto formats the one-hot-encoding for points prediction
    
    OUTPUT: A dataframe with expected points added to your team based on average points 
            scored by the player vs what would be expected based on position + salary regression
    '''

    #----------
    # Create the initial baseline projection for your team
    #----------

    # copy the blank team template
    my_team_template = team_template.copy()
    
    # determine the remaining salary for the team across positions
    remain_base, selected_base = salary_proportions(my_team_template, proport, sal)
    
    # create the X prediction for the base case and determine point distribution
    X_base = create_X_test(remain_base, test_template)
    team_dist_base = create_pt_dist(X_base, selected_base, lr_mean, lr_std)

    # loop through each player in selected players
    exp_pts = []
    for _, row in my_team_select.iterrows():

        # get the team dataframe filled with only the current player
        my_team_template = team_template.copy()
        cur_player = team_fill(my_team_template, pd.DataFrame(row).T)
        remain_pl, selected_pl = salary_proportions(cur_player, proport, sal-row.Salary)
        
        # predict points scored by the current player and all other blanks
        X_pl = create_X_test(remain_pl, test_template)
        team_dist_pl = create_pt_dist(X_pl, selected_pl, lr_mean, lr_std)
        
        # determine the number of points added by this player
        pts_added = (team_dist_pl.mean() - team_dist_base.mean()).values[0]
        exp_pts.append([row.Player, str(int(pts_added))])

    return pd.DataFrame(exp_pts, columns=['Player', 'Points Added'])



def team_fill(df, df2):
    '''
    INPUT: df: blank team template to be filled in with chosen players
           df2: chosen players dataframe

    OUTPUT: Dataframe filled in with selected player information
    '''
    # loop through chosen players dataframe
    for _, row in df2.iterrows():

        # pull out the current position and find min position index not filled (if any)
        cur_pos = row.Position
        min_idx = df.loc[(df.Position==cur_pos) & (df.Player.isnull())].index.min()

        # if position still needs to be filled, fill it
        if min_idx is not np.nan:
            df.loc[min_idx, ['Player', 'Salary']] = [row.Player, row.Salary]

        # if normal positions filled, fill in the FLEX if applicable
        elif cur_pos in ('RB', 'WR', 'TE'):
            cur_pos = 'FLEX'
            min_idx = df.loc[(df.Position==cur_pos) & (df.Player.isnull())].index.min()
            if min_idx is not np.nan:
                df.loc[min_idx, ['Player', 'Salary']] = [row.Player, row.Salary]

            # otherwise, fill in the Bench
            else:
                bench = pd.DataFrame(['Bench', row.Player, row.Salary]).T
                bench.columns = ['Position', 'Player', 'Salary']
                df = pd.concat([df, bench], axis=0)
    return df



@app.callback([Output('draft-results-graph', 'figure'),
               Output('team-points-graph', 'figure'),
               Output('my-team-table', 'data'),
               Output('team-info-table', 'data')],
              [Input('submit-button-state', 'n_clicks')],
              [State('draft-results-table', 'data'),
               State('draft-results-table', 'columns')]
)
def update_output(n_clicks, drafted_data, drafted_columns):

    # get the list of drafted players
    drafted_df = pd.DataFrame(drafted_data, columns=[c['name'] for c in drafted_columns])
    drafted_df.Salary = drafted_df.Salary.astype('int')

    # create a template of all team positions and players current selected for my team
    my_team_template = my_team_df.copy()
    my_team_select = drafted_df[drafted_df['My Team']=='Yes'].reset_index(drop=True)
    my_team_update = team_fill(my_team_template.copy(), my_team_select)

    # create a dataset of all other players that have been drafted
    drafted_df = drafted_df[drafted_df['My Team']!='Yes'].reset_index(drop=True)

    # get lists of to_drop and to_add players and remaining salary
    to_drop = update_to_drop(drafted_df)
    to_add = update_to_add(my_team_update)
    remain_sal = league_info['salary_cap'] - np.sum(to_add['salaries'])

    # run the simulation
    _, inflation = sim.run_simulation(league_info, to_drop, to_add, iterations=iterations)
    
    # get the results dataframe structured
    avg_sal = sim.show_most_selected(to_add, iterations, num_show=30)
    avg_sal = avg_sal.sort_values(by='Percent Drafted').reset_index()
    avg_sal.columns = ['Player', 'PercentDrafted', 'AverageSalary', 'ExpectedSalaryDiff']

    avail_pts = pd.merge(avg_sal, d[d.Position!='FLEX'].reset_index()[['player', 'Position']].rename(columns={'player': 'Player'}), on='Player')
    avail_pts = avail_pts[['Position', 'Player', 'AverageSalary']].rename(columns={'AverageSalary': 'Salary'})
    avail_pts = expected_pts(my_team_update, avail_pts[['Position', 'Player', 'Salary']], test_template, remain_sal)
    avg_sal = pd.merge(avg_sal, avail_pts, on='Player')

    # Creating two subplots and merging into single figure
    (pl, pc_dr, av_sl, pt) = avg_sal.Player, avg_sal.PercentDrafted, avg_sal.AverageSalary/1000, avg_sal['Points Added']
    pick_bar = create_bar(pc_dr, pl, text=pc_dr)
    sal_bar = create_bar(av_sl, pl, color_str='rgba(250, 190, 88, 1)', text=av_sl)
    pt_bar = create_bar(pt, pl, color_str='rgba(250, 128, 114, 1)', text=pt)
    gr_fig = create_fig_layout(sal_bar, pick_bar, pt_bar)

    # histogram creation
    remain_pos, selected_df = salary_proportions(my_team_update, proport, remain_sal)
    X_test = create_X_test(remain_pos, test_template)
 
    cur_team_dist = create_pt_dist(X_test, selected_df, lr_mean, lr_std)
    hist_fig = create_hist(cur_team_dist)

    if my_team_select.shape[0] > 0:
        team_pts_added = expected_pts(my_team_template, my_team_select[['Position', 'Player', 'Salary']], test_template, league_info['initial_cap'])
        my_team_update = pd.merge(my_team_update.drop('Points Added', axis=1), team_pts_added, on='Player', how='left')
        my_team_update['Points Added'] = my_team_update['Points Added'].fillna(0)

        # team_pts_added = expected_pts(my_team_update, my_team_select[['Position', 'Player', 'Salary']], test_template, remaining_sal)
        # my_team_update = pd.merge(my_team_update.drop('Points Added', axis=1), team_pts_added, on='Player', how='left')
        # my_team_update['Base Points Added'] = my_team_update['Base Points Added'].fillna(0)

    # update team information table
    team_info_update = pd.DataFrame({'Mean Points': [int(cur_team_dist.mean())], 
                                     'Inflation': [round(inflation,2)], 
                                     'Remaining Salary': [remain_sal]})
    
    # save out csv of status
    drafted_df.to_csv('c:/Users/mborysia/Desktop/Status_Save.csv', index=False)

    return gr_fig, hist_fig, my_team_update.to_dict('records'), team_info_update.to_dict('records')


#%%
if __name__ == '__main__':
    app.run_server(debug=False)

# %%
