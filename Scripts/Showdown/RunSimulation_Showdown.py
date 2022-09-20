
#%%

# import sim functions
from zSim_Helper_Showdown import *
import seaborn as sns
from IPython.core.pylabtools import figsize
import os
import sqlite3

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral

root_path = ffgeneral.get_main_path('Daily_Fantasy')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

#===============
# Settings and User Inputs
#===============

# np.random.seed(1234)

#--------
# League Settings
#--------

# connection for simulation and specific table
path = f'/Users/{os.getlogin()}/Documents/Github/Daily_Fantasy/'
conn_sim = sqlite3.connect(f'{path}/Data/Databases/Simulation.sqlite3')
set_year = 2022
league=1

# number of iteration to run
iterations = 1000

# set league information, included position requirements, number of teams, and salary cap
league_info = {}
league_info['pos_require'] = {'CPT': 1, 'FLEX': 5}
league_info['num_teams'] = 12
league_info['initial_cap'] = 50000
league_info['salary_cap'] = 50000

total_pos = np.sum(list(league_info['pos_require'].values()))

#==================
# Initialize the Simluation Class
#==================

# instantiate simulation class and add salary information to data
sim = FootballSimulation(conn_sim, set_year, league)

# return the data and set up dataframe for proportion of salary across position
d = sim.return_data()
d = d.rename(columns={'pos': 'Position', 'salary': 'Salary'})
d.Position = d.Position.apply(lambda x: x[1:])

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
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=False)


#==================
# Expected Points Functions
#==================

def csv_check(df, name):
    return df.to_csv(f'c:/Users/mborysia/Desktop/FF_App/{name}.csv', index=False)

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

def get_pick_df(d):
    player_list = []
    for pl, row in d.sort_values(by='Salary', ascending=False)[['Salary', 'Position']].iterrows():
        player_list.append([row.Position, pl, row.Salary, 0])

    pick_df = pd.DataFrame(player_list, columns=['Position', 'Player', 'List Salary', 'Salary'])
    pick_df['My Team'] = 'No'

    for p, s in keepers.items():
        pick_df.loc[pick_df.Player==p, 'Salary'] = s

    return pick_df

# set up all players drafted DataTable
def get_drafted_player_table(pick_df):
    return dash_table.DataTable(
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
                          'Remaining Salary Per': [np.round(league_info['salary_cap'] / total_pos,0)],
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

#==========================
# Plotting Functions
#==========================

def create_bar(x_val, y_val, orient='h', color_str=main_color_rgba, text=None):
    '''
    Function to create a horizontal bar chart
    '''
    marker_set = dict(color=color_str, line=dict(color=color_str, width=1))
    return go.Bar(x=x_val, y=y_val, marker=marker_set, orientation=orient, text=text, showlegend=False)


def create_fig_layout(fig1, fig2):
    '''
    Function to combine bar charts into a single figure
    '''
    fig = go.Figure(data=[fig1, fig2])
    
    # Change the bar mode
    fig.update_layout(barmode='group', autosize=True, height=1800, 
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

#-------------
# Set up the CSV download button
#-------------

blue_button_style = {'background-color': 'blue',
                    'color': 'white',
                    'height': '20',
                    'width': '100%',
                    'fontSize': 16}

download_button = html.Button("Push to Database", id="download-button", style=blue_button_style)
file_download = dcc.Download(id="download")

#============================
# Build out App Layout
#============================

def app_layout():

    d = sim.return_data()
    d = d.rename(columns={'pos': 'Position', 'salary': 'Salary'})
    d.Position = d.Position.apply(lambda x: x[1:])

    pick_df = get_pick_df(d)
    drafted_player_table = get_drafted_player_table(pick_df)

    return html.Div([
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
                            download_button, file_download,
                            html.Hr(),
                            hist_gr
                        ], className='four columns'),

                        html.Div([
                            html.H5('Recommended Picks'),
                            bar_gr
                        ], className="four columns")

                ], className="row2") ,        
                    
                ])


app.layout = app_layout


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
            to_drop['players'].append(row.Player + '_' + row.Position)
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
            to_add['players'].append(row.Player + '_' + row.Position)
            to_add['salaries'].append(row.Salary)

    return to_add


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
        elif cur_pos in ('CPT'):
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
             #  Output('team-points-graph', 'figure'),
               Output('my-team-table', 'data'),
               Output('team-info-table', 'data'),
               Output("download", "data")],
              [Input('submit-button-state', 'n_clicks'),
               Input("download-button", "n_clicks")],
              [State('draft-results-table', 'data'),
               State('draft-results-table', 'columns')],
               prevent_initial_call=True,
)
def update_output(n_clicks, n_clicks_csv, drafted_data, drafted_columns):

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
    if my_team_select.shape[0] < total_pos:
        _, _ = sim.run_simulation(league_info, to_drop, to_add, iterations=iterations)
    
    # get the results dataframe structured
    avg_sal = sim.show_most_selected(to_add, iterations, num_show=30)
    avg_sal = avg_sal.sort_values(by='Percent Drafted').reset_index()
    avg_sal.columns = ['Player', 'PercentDrafted', 'AverageSalary', 'ExpectedSalaryDiff']

    # Creating two subplots and merging into single figure
    (pl, pc_dr, av_sl) = avg_sal.Player, avg_sal.PercentDrafted,  avg_sal.AverageSalary/1000
    pick_bar = create_bar(pc_dr, pl, text=pc_dr)
    sal_bar = create_bar(av_sl, pl, color_str='rgba(237, 137, 117, 1)', text=av_sl)

    gr_fig = create_fig_layout(sal_bar, pick_bar)
 
    # hist_fig = create_hist(cur_team_dist)

    if my_team_select.shape[0] > 0:
        selected = list(my_team_select.Player)
        my_player_pts = sim.data[(sim.data.index.isin(selected)) & (sim.data.pos!='bFLEX')].drop(['pos', 'salary'], axis=1)
        
        my_player_mean = my_player_pts.copy().mean(axis=1).reset_index()
        my_player_mean.columns = ['Player', 'Points Added']
        my_player_mean['Points Added'] = my_player_mean['Points Added'].apply(lambda x: np.round(x, 1))

        my_team_update = pd.merge(my_team_update.drop('Points Added', axis=1), my_player_mean, on='Player', how='left')
        my_team_update['Points Added'] = my_team_update['Points Added'].fillna(0)

        remain_sal_per = np.round(remain_sal / (total_pos - len(selected)),0)

    else:
        remain_sal_per = np.round(remain_sal / total_pos,0)

    # update team information table
    team_info_update = pd.DataFrame({'Mean Points': np.round(my_team_update['Points Added'].sum(), 1), 
                                     'Remaining Salary Per': [remain_sal_per],
                                     'Remaining Salary': [remain_sal]})
    
    # save out csv of status
    # drafted_df.to_csv('c:/Users/mborysia/Desktop/Status_Save.csv', index=False)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'download-button' in changed_id:

        ids = dm.read(f"SELECT * FROM Player_Ids WHERE year={set_year} AND league={league}", "Simulation")
        my_team_ids = my_team_select.rename(columns={'Player': 'player'}).copy()
        dk_output = pd.merge(my_team_ids, ids, on='player')

        for pstn, num_req in zip(['WR', 'RB', 'TE'], [4, 3, 2]):
            if len(dk_output[dk_output.Position == pstn]) == num_req:
                idx_last = dk_output[dk_output.Position == pstn].index[-1]
                dk_output.loc[dk_output.index==idx_last, 'Position'] = 'FLEX'

        pos_map = {
            'QB': 'aQB', 
            'RB': 'bRB',
            'WR': 'cWR',
            'TE': 'dTE',
            'FLEX': 'eFLEX',
            'DST': 'fDST'
        }
        dk_output.Position = dk_output.Position.map(pos_map)
        dk_output = dk_output.sort_values(by='Position').reset_index(drop=True)
        pos_map_rev = {v: k for k,v in pos_map.items()}
        dk_output.Position = dk_output.Position.map(pos_map_rev)

        dk_output_ids = dk_output[['Position', 'player_id']].T.reset_index(drop=True)
        dk_output_players = dk_output[['Position', 'player']].T.reset_index(drop=True)
        dk_output = pd.concat([dk_output_players, dk_output_ids], axis=1)

        dk_output.columns = range(dk_output.shape[1])
        dk_output = pd.DataFrame(dk_output.iloc[1,:]).T

        dk_output['year'] = set_year
        dk_output['week'] = league
        

        dm.write_to_db(dk_output, 'Results', 'Best_Lineups', 'append')

        return gr_fig, my_team_update.to_dict('records'), team_info_update.to_dict('records'), None

    else:
        return gr_fig, my_team_update.to_dict('records'), team_info_update.to_dict('records'), None


#%%
if __name__ == '__main__':
    app.run_server(debug=True)

# %%
