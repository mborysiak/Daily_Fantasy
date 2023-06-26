#%%
import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, ColumnsAutoSizeMode
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
import sqlite3


year = 2022
week = 2
num_iters = 150

total_lineups = 5

#-----------------
# Pull Data In
#-----------------

def get_conn(filename):
    from pathlib import Path
    filepath = Path(__file__).parents[0] / filename
    conn = sqlite3.connect(filepath)
    return conn

def pull_op_params(conn, week, year):

    # pull in the run parameters for the current week and year
    op_params = pd.read_sql_query(f'''SELECT * 
                                      FROM Run_Params
                                      WHERE week={week}
                                            AND year={year}''', conn)
    op_params = {k: v[0] for k,v in op_params.to_dict().items()}
    return op_params

def pull_ownership(conn, week, year):

    # pull in projected ownership
    ownership = pd.read_sql_query(f'''SELECT player Player, pred_ownership Ownership
                                      FROM Predicted_Ownership
                                      WHERE week={week} 
                                            AND year={year}''', conn)
    ownership.Ownership = ownership.Ownership.apply(lambda x: np.round(100*np.exp(x),1))
    return ownership


def pull_requirements():
    # set league information, included position requirements, number of teams, and salary cap
    salary_cap = 50000
    pos_require_start = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DEF': 1}

    # create a dictionary that also contains FLEX
    pos_require_flex = copy.deepcopy(pos_require_start)
    del pos_require_flex['DEF']
    pos_require_flex['FLEX'] = 1
    pos_require_flex['DEF'] = 1
    total_pos = np.sum(list(pos_require_flex.values()))
    return salary_cap, pos_require_start, pos_require_flex, total_pos

#------------------
# App Components
#------------------

def create_interactive_grid(data):
    gb = GridOptionsBuilder.from_dataframe(data)
    # add pagination
    # gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        columns_auto_size=ColumnsAutoSizeMode.FIT_CONTENTS,
        # fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        height=500, 
        width='100%',
        reload_data=False
    )

    data = grid_response['data']
    selected = grid_response['selected_rows'] 
    df = pd.DataFrame(selected) 

    return df

def create_plot(df):
    # Create a plot
    ax = df[['player', 'dk_salary']].set_index('player').plot(kind='barh')

    # Display the plot using Streamlit
    return st.write(ax.get_figure())

@st.cache_data
def convert_df_for_dl(df):
    return df.to_csv(index=False).encode('utf-8')

    


def main():
    # Set page configuration
    st.set_page_config(layout="wide")
    
    col1, col2 = st.columns(2)
    conn = get_conn('Simulation_App.sqlite3')
    op_params = pull_op_params(conn, week, year)
    ownership = pull_ownership(conn, week, year)

    with col1:
        st.write(ownership.head())
        df = create_interactive_grid(ownership)
    with col2:
        st.write(op_params)
        # create_plot(df)

        # st.download_button(
        #     "Press to Download",
        #     convert_df_for_dl(df),
        #     "file.csv",
        #     "text/csv",
        #     key='download-csv'
        # )

 


if __name__ == '__main__':
    main()
# %%

