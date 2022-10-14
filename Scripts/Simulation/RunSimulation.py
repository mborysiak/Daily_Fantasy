
#%%

# import sim functions
from pandas.core.frame import DataFrame
from zSim_Helper_Covar import *
import os

# set the root path and database management object
from ff.db_operations import DataManage
from ff import general as ffgeneral

# root_path = '/Users/sammyers/Desktop/Daily/'
if 'mborysia' in os.getcwd():
    root_path = ffgeneral.get_main_path('Daily_Fantasy')
    db_path = f'{root_path}/Data/Databases/'
else:
    root_path = os.getcwd()
    db_path = root_path

dm = DataManage(db_path)

#===============
# Settings and User Inputs
#===============

year = 2022
week = 5
num_iters = 100

total_lineups = 8

#-----------------
# Model and Sim Settings
#-----------------

# pull in the run parameters for the current week and year
op_params = dm.read(f'''SELECT * 
                        FROM Run_Params
                        WHERE week={week}
                              AND year={year}''', 'Simulation')
op_params = {k: v[0] for k,v in op_params.to_dict().items()}

# pull in projected ownership
ownership = dm.read(f'''SELECT player Player, pred_ownership Ownership
                        FROM Predicted_Ownership
                        WHERE week={week} 
                            AND year={year}''', 'Simulation')
ownership.Ownership = ownership.Ownership.apply(lambda x: np.round(100*np.exp(x),1))

#--------
# League Settings
#--------

# set league information, included position requirements, number of teams, and salary cap
salary_cap = 50000
pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}

# create a dictionary that also contains FLEX
pos_require_flex = copy.deepcopy(pos_require_start)
del pos_require_flex['DEF']
pos_require_flex['FLEX'] = 1
pos_require_flex['DEF'] = 1
total_pos = np.sum(list(pos_require_flex.values()))


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


#============================
# Build out Dash Tables
#============================

main_color = '40, 110, 132'
main_color_rgb = f'rgb({main_color})'
main_color_rgba = f'rgba({main_color}, 0.8)'

#--------------
# Set up dataframe and Data Table for My Team
#--------------

def initiate_class(op_params):

    # extract all the operating parameters
    pred_vers = op_params['pred_vers']
    ensemble_vers = op_params['ensemble_vers']
    std_dev_type = op_params['std_dev_type']
    full_model_rel_weight = eval(op_params['full_model_rel_weight'])
    covar_type = eval(op_params['covar_type'])
    use_covar = eval(op_params['use_covar'])
    use_ownership = eval(op_params['use_ownership'])

    print('Full Model Weight:', full_model_rel_weight, 'Use Covar:', use_covar, 'Use Ownership:', use_ownership)

    # instantiate simulation class and add salary information to data
    sim = FootballSimulation(dm, week, year, salary_cap, pos_require_start, num_iters,
                            pred_vers, ensemble_vers, std_dev_type, covar_type,
                            full_model_rel_weight, use_covar, use_ownership)

    return sim


def pull_prev_counts(week, year):

    df = dm.read(f'''SELECT * 
                     FROM Best_Lineups
                     WHERE week={week} 
                           AND year={year}''', 'Results')
    if len(df)>0:
        df = pd.melt(df.iloc[:, :9]).value.value_counts().reset_index()
        df.columns = ['Player', 'Prev Cnts']
    else:
        df = None
    
    return df


def rand_drop_selected(df, drop_multiplier):
    to_ignore = []
    total_selections = df.loc[df['Prev Cnts']>0, ['Player', 'Prev Cnts']].values

    for k, v in total_selections:
        prob_drop = (v * drop_multiplier) / total_lineups
        drop_val = np.random.uniform() * prob_drop
        if  drop_val >= 0.5:
            to_ignore.append(k)

    df.loc[df.Player.isin(to_ignore), 'My Team'] = 'Ignore'

    return df


def init_player_table_df(d, week, year):

    player_list = []
    for _, row in d.sort_values(by='Salary', ascending=False).iterrows():
        player_list.append([row.Position, row.player, row.Salary])

    pick_df = pd.DataFrame(player_list, columns=['Position', 'Player', 'Salary'])

    num_selected = pull_prev_counts(week, year)

    if num_selected is not None:
        pick_df = pd.merge(pick_df, num_selected, on='Player', how='left').fillna({'Prev Cnts': 0})
    else:
        pick_df['Prev Cnts'] = 0

    pick_df['My Team'] = 'No'
    pick_df = rand_drop_selected(pick_df, drop_player_multiple)

    return pick_df


# set up all players drafted DataTable
def init_player_table_dash(pick_df):
    return dash_table.DataTable(
                            id='draft-results-table',

                            columns=[{'id': c, 'name': c} for c in pick_df.columns if c != 'My Team'] +
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
                                            {'label': i, 'value': i} for i in ['Yes', 'No', 'Ignore']
                                        ]
                                        }
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

def init_possible_teams(df):
    possible_teams = ['Auto']
    possible_teams.extend(list(df.team.unique()))
    return [t for t in possible_teams if t is not None]


def init_stack_table_dash(possible_teams):
    possible_stacks = ['Auto']
    possible_stacks.extend([str(i) for i in range(1, 5)])
    return dash_table.DataTable(
                            id='stack-selection-table',

                            columns=[{'id': 'Stack Team', 'name': 'Stack Team', 'presentation': 'dropdown', 'editable': True},
                                     {'id': 'Stack Number', 'name': 'Stack Number','presentation': 'dropdown', 'editable': True}],
                            data=pd.DataFrame({'Stack Team': ['Auto'], 'Stack Number': ['Auto']}).to_dict('records'),
                            style_table={
                                            'height': '75px'
                                        },
                            style_cell={'textAlign': 'left', 'fontSize':14, 'font-family':'sans-serif'},
                            dropdown={
                                    'Stack Team': {
                                        'options': [
                                            {'label': i, 'value': i} for i in possible_teams
                                        ]},
                                    'Stack Number': {
                                        'options': [
                                            {'label': i, 'value': i} for i in possible_stacks
                                        ]
                                        }
                                    },
                            style_header={
                                        'backgroundColor': main_color_rgb,
                                        'fontWeight': 'bold',
                                        'color': 'white'
                                    }
                        )

#--------------
# Set up dataframe and Data Table for My Team and Team Info
#--------------

def basic_dash_table(id_name, df, text_align='left', font_size=14):

     return  dash_table.DataTable(
                            id=id_name,
                            columns=[{"name": i, "id": i, 'editable': False} for i in df.columns],
                            data=df.to_dict('records'),
                            style_cell={'textAlign': text_align, 'fontSize': font_size, 'font-family':'sans-serif'},
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

def init_my_team_df(pos_require):

    my_team_list = []
    for k, v in pos_require.items():
        for _ in range(v):
            my_team_list.append([k, None, 0, 0])
    my_team_df = pd.DataFrame(my_team_list, columns=['Position', 'Player', 'Salary', 'Points Added'])
    
    return my_team_df


def init_team_info_df(salary_cap, total_pos):

    return pd.DataFrame({'Mean Points': [None],
                         'Remaining Salary Per': [np.round(salary_cap / total_pos,0)],
                         'Remaining Salary': [salary_cap]})

#==========================
# Plotting Functions
#==========================

def create_bar(x_val, y_val, orient='h', color_str=main_color_rgba, text=None):
    '''
    Function to create a horizontal bar chart
    '''
    marker_set = dict(color=color_str, line=dict(color=color_str, width=1))
    return go.Bar(x=x_val, y=y_val, marker=marker_set, orientation=orient, 
                 text=text, showlegend=False)


def create_fig_layout(figs, height=450):
    '''
    Function to combine bar charts into a single figure
    '''
    fig = go.Figure(data=figs)
    
    # Change the bar mode
    fig.update_layout(barmode='group', autosize=True, height=height, 
                      margin=dict(l=0, r=25, b=0, t=15, pad=0),
                      uniformtext_minsize=25, uniformtext_mode='hide')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    return fig


def init_graph_dash_object(id_name):
    return  dcc.Graph(id=id_name)

#==========================
# Clickable buttons
#==========================

def init_submit_dash_button():
    
    # submit button
    red_button_style = {'background-color': 'red',
                        'color': 'white',
                        'height': '20',
                        'width': '100%',
                        'fontSize': 16}

    return html.Button(id='submit-button-state', n_clicks=0, 
                       children='Refresh Top Picks', style=red_button_style)


def init_download_dash_button():

    blue_button_style = {'background-color': 'blue',
                        'color': 'white',
                        'height': '20',
                        'width': '100%',
                        'fontSize': 16}

    download_button = html.Button("Push to Database", id="download-button", style=blue_button_style)
    file_download = dcc.Download(id="download")

    return download_button, file_download


#============================
# Build out App Layout
#============================

def app_layout():

    global sim, adjust_select, drop_player_multiple, min_players_opp_team

    sim = initiate_class(op_params)
    adjust_select = eval(op_params['adjust_select'])
    drop_player_multiple = eval(op_params['drop_player_multiple'])
    min_players_opp_team = eval(op_params['min_players_opp_team'])

    print('adjust_select:', adjust_select, 'drop_multiple:', drop_player_multiple, 'opp_players:', min_players_opp_team)

    # pull in current player data
    d = sim.player_data
    d = d.rename(columns={'pos': 'Position', 'salary': 'Salary'})
    
    # create the player data dash interface
    player_data_df = init_player_table_df(d, week, year)
    player_data_dash_table = init_player_table_dash(player_data_df)

    # get the interface for my team selection tracking
    my_team_df = init_my_team_df(pos_require_flex)
    my_team_dash_table = basic_dash_table('my-team-table', my_team_df)

    possible_teams = init_possible_teams(d)
    stack_selection_dash_table = init_stack_table_dash(possible_teams)

    # get the interface for tracking overal team information (points, salary, etc)
    team_info_df = init_team_info_df(salary_cap, total_pos)
    team_info_dash_table = basic_dash_table('team-info-table', team_info_df, text_align='center')

    # get the interface for the player selection bar graph
    player_selection_graph = init_graph_dash_object(id_name='draft-results-graph',)
    best_team_graph =  init_graph_dash_object(id_name='best-team-graph')

    # get the interface for submit and download buttons
    download_button, file_download = init_download_dash_button()
    submit_button = init_submit_dash_button()

    return html.Div([
                html.Div([
                            # first column of viz with player data
                            html.Div([
                                html.H6("Enter Draft Pick Information"),
                                player_data_dash_table,
                                ], className="four columns"),

                            # second column of viz with my team, team info, and refresh buttons
                            html.Div([
                                html.H6('My Team'),
                                my_team_dash_table, 

                                html.H6('Stack Selection'),
                                stack_selection_dash_table, html.Hr(),

                                submit_button, html.Hr(),

                                html.H6('Team Information'),
                                team_info_dash_table, html.Hr(),

                                download_button, file_download, html.Hr(),
                            ], className='four columns'),

                            # third column of viz with player recommendation graph
                            html.Div([
                                html.H6('Top Scoring Teams'),
                                best_team_graph, 

                                html.H6('Recommended Picks'),
                                player_selection_graph

                            ], className="four columns")

                ], className="row2") ,        
                    
                ])


# update the app based on the above layout
app.layout = app_layout


#============================
# Update Functions
#============================

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

    return df


def convert_dash_to_df(player_data, player_columns):

    # get the list of drafted players
    df = pd.DataFrame(player_data, columns=[c['name'] for c in player_columns])
    df.Salary = df.Salary.astype('int')

    return df


def create_my_team(df, pos_require):
    
    # create a template of all team positions and players current selected for my team
    my_team_template = init_my_team_df(pos_require)
    my_team = df[df['My Team']=='Yes'].reset_index(drop=True)
    my_team_dash = team_fill(my_team_template.copy(), my_team)
    num_players_selected = my_team.shape[0]

    return my_team, my_team_dash, num_players_selected


def create_players_removed(df):
    return df[df['My Team']!='Yes'].reset_index(drop=True)

def update_add_drop_lists(df):
    players = list(df.loc[df['My Team'].isin(['Yes','Ignore']), 'Player'].values)
    salaries = list(df.loc[df['My Team'].isin(['Yes','Ignore']), 'Salary'].values)

    return players, salaries

def update_stack_data(stack_data):

    # extract the max team and number of players to pull from max team
    set_max_team = stack_data[0]['Stack Team']
    min_players_same_team = stack_data[0]['Stack Number']
    
    if min_players_same_team != 'Auto':
        min_players_same_team = int(min_players_same_team)    
    
    # if auto set the max team, then convert to None for sim argument
    if set_max_team=='Auto': set_max_team=None

    return set_max_team, min_players_same_team


def format_results(results, my_team_cnt):

    len_results = len(results)
    max_idx = len_results - my_team_cnt
    min_idx = len_results - 20 - my_team_cnt

    # get the results dataframe structured
    results = results.sort_values(by='SelectionCounts').iloc[min_idx:max_idx]
    results.columns = ['Player', 'PercentDrafted', 'AverageSalary']

    results = pd.merge(results, ownership, on='Player', how='left')
    results.Ownership = results.Ownership.fillna(0)

    return results


def update_player_selection_chart(results):
    # Creating subplots and merging into single figure
    pick_bar = create_bar(results.PercentDrafted, results.Player, text=results.PercentDrafted)
    sal_bar = create_bar(results.AverageSalary/1000, results.Player, color_str='rgba(237, 137, 117, 1)', text=results.AverageSalary/1000)
    own_bar = create_bar(results.Ownership, results.Player, color_str='rgba(250, 190, 88, 1)', text=results.Ownership)
    player_pick_dash_fig = create_fig_layout([own_bar, sal_bar, pick_bar], height=1200)

    return player_pick_dash_fig


def update_top_team_chart(max_team_cnt):

    max_team_cnt = max_team_cnt.sort_values(by='high_score_team').iloc[-8:]
    team_bar = create_bar(max_team_cnt.high_score_team, max_team_cnt.team, text=max_team_cnt.high_score_team)
    team_pick_dash_fig = create_fig_layout([team_bar], height=200)
    return team_pick_dash_fig


def update_team_info_table(my_team, my_team_dash, remain_sal):

    selected = list(my_team.Player)
    my_player_pts = sim.player_data.loc[(sim.player_data.player.isin(selected)), ['player', 'pred_fp_per_game']]
    my_player_pts.columns = ['Player', 'Points Added']
    my_player_pts['Points Added'] = my_player_pts['Points Added'].apply(lambda x: np.round(x, 1))

    my_team_dash = pd.merge(my_team_dash.drop('Points Added', axis=1), my_player_pts, on='Player', how='left')
    my_team_dash['Points Added'] = my_team_dash['Points Added'].fillna(0)

    remain_sal_per = np.round(remain_sal / (total_pos - len(selected)),0)

    # update team information table
    team_info_update = pd.DataFrame({'Mean Points': np.round(my_team_dash['Points Added'].sum(), 1), 
                                     'Remaining Salary Per': [remain_sal_per],
                                     'Remaining Salary': [remain_sal]})

    return team_info_update, my_team_dash


def create_database_output(my_team):

    ids = dm.read(f"SELECT * FROM Player_Ids WHERE year={year} AND league={week}", "Simulation")
    my_team_ids = my_team.rename(columns={'Player': 'player'}).copy()
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

    dk_output['year'] = year
    dk_output['week'] = week
    
    dm.write_to_db(dk_output, 'Results', 'Best_Lineups', 'append')


@app.callback([Output('draft-results-graph', 'figure'),
               Output('best-team-graph', 'figure'),
               Output('my-team-table', 'data'),
               Output('team-info-table', 'data'),
               Output("download", "data")],
              [Input('submit-button-state', 'n_clicks'),
               Input("download-button", "n_clicks")],
              [State('draft-results-table', 'data'),
               State('draft-results-table', 'columns'),
               State('stack-selection-table', 'data')],
               prevent_initial_call=True,
)
def update_output(nc, nc2, player_dash_table_data, player_dash_table_columns, stack_data):

    # extract the player data with yes / no selections and turn into my team and non-drafted players
    player_data_df = convert_dash_to_df(player_dash_table_data, player_dash_table_columns)
    my_team, my_team_dash, my_team_player_cnt = create_my_team(player_data_df, pos_require_flex)
    players_removed = create_players_removed(player_data_df)

    # get lists of to_drop and to_add players and remaining salary
    to_drop_players, _ = update_add_drop_lists(players_removed)
    to_add_players, added_salaries = update_add_drop_lists(my_team)
    remain_sal = salary_cap - np.sum(added_salaries)

    # run the simulation
    if my_team_player_cnt <= total_pos:
        set_max_team, min_players_same_team = update_stack_data(stack_data)
        
        results, max_team_cnt = sim.run_sim(to_add_players, to_drop_players, min_players_same_team, 
                                            set_max_team, min_players_opp_team, adjust_select=adjust_select)
        results = format_results(results, my_team_player_cnt)
        player_select_graph = update_player_selection_chart(results)
        top_team_graph = update_top_team_chart(max_team_cnt)

    team_info_update, my_team_dash = update_team_info_table(my_team, my_team_dash, remain_sal)

    # convert my team and the team info tables to dictionary records for Output
    my_team_dash = my_team_dash.to_dict('records')
    team_info_update = team_info_update.to_dict('records')

    # trigger for if database button is pushed
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'download-button' in changed_id:
        create_database_output(my_team)
        return player_select_graph, top_team_graph, my_team_dash, team_info_update, None

    else:
        return player_select_graph, top_team_graph, my_team_dash, team_info_update, None


#%%
if __name__ == '__main__':
    app.run_server(debug=False, host='127.0.0.1')

# %%
