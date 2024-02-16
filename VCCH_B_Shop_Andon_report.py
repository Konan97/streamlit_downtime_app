### IMPORT ###
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import datetime

from snowflake.snowpark.context import get_active_session
from datetime import date, timedelta

### GLOBAL VARIABLES ###
session = get_active_session()
#AA safety AB power supply AC equipment ADEF warnings
# conveyer(PC) process(PP) application(PA) material(PM)
# PC1432, NVH, PA1343 no connections, PM1261 waiting for upgrade,
# PP1361
dic_plcs_to_sections = {
    '1593003OP003_1':'WD20 South',
    '1593002OP002_1':'WD20 North'
}
# 'WD20 South' : 1593003OP003_1
# 'WD20 North' : 1593002OP002_1

dic_sections_to_plcs = {
    'WD20 South': '1593003OP003_1',
    'WD20 North': '1593002OP002_1'
}
# 5195 station 1
# 5225 station 6
# 5230 station SIP
RB_to_station = {
    '5145': '1', '5150': '2', '5155': '3', '5165': '4', '5170': '5', '5175': '6', '5180': 'SIP',
    '5195': '1', '5200': '2', '5205': '3', '5210': '4', '5215': '5', '5225': '6', '5230': 'SIP'
}


### HELPER FUNCTIONS ###

# Change state after clicking on form
# https://discuss.streamlit.io/t/form-submit-buttons-on-click-being-called-before-clicking/27767
def click_form(key, boolean):
    st.session_state[key] = boolean

# Remove regular break times return a new df
# 0800 to 0810
# 1000 to 1040
# 1300 to 1310
# Anything after 1540 until 0700 the next day
# example: 7:49 to 8:09 should break into 7:49 to 8:00 (Andon) and 8:00 to 8:09 (regular break)
def remove_breaks(data):
    # 6:00      2       4
    #       1  ---     ---
    # 8:00 ---  *   3   *
    #       *  --- ---  *
    # 8:10 ---      *   *
    #              --- ---
    # 10:00
    # timeframes to be modified
    break_sch = {datetime.time(8, 0, 0) : datetime.time(8, 10, 0), datetime.time(10,0,0):datetime.time(10,40,0), datetime.time(13, 0, 0): datetime.time(13, 10, 0)}
    # ==test==stituation=1===============================================
    # test = data.loc[0].copy()
    # test['Alarm Start'] = test['Alarm Start'].replace(hour=8, minute=1)
    # test['Alarm End'] = test['Alarm End'].replace(hour=8, minute=5)
    # data.loc[0] = test
    # ===================================================================
    addition_row = [] # additional rows to append to the current df
    for index, row in data.iterrows():
        for start, end in break_sch.items():
            # stituation 1
            if row['Alarm Start'].time() >= start and row['Alarm End'].time() <= end:
                st.write('hello')
                data = data.drop([index])
                break
            # stituation 2
            elif row['Alarm End'].time() >= start and row['Alarm End'].time() <= end:
                # replace row['Alarm End'] to start
                data.at[index, 'Alarm End'] = data.iloc[index]['Alarm End'].replace(hour=start.hour, minute= start.minute, second= start.second)
                break
            # stituation 3
            elif row['Alarm Start'].time() >= start and row['Alarm Start'].time() <= end: #and row['Alarm End'].time() <= datetime.time(10, 0, 0):
                data.at[index, 'Alarm Start'] = data.iloc[index]['Alarm Start'].replace(hour=end.hour, minute=end.minute, second=end.second)
                break
            # stituation 4
            elif row['Alarm Start'].time() < start and row['Alarm End'].time() > end: #and row['Alarm End'].time() < datetime.time(10, 0, 0):
                new_row = row.copy()
                # replace row['Alarm End'] to start
                data.at[index, 'Alarm End'] = row['Alarm End'].replace(hour=start.hour, minute=start.minute, second=start.second)
                # replace ['Alarm Start'] to end
                
                new_row['Alarm Start'] = new_row['Alarm Start'].replace(hour=end.hour, minute=end.minute, second= end.second)
                addition_row.append(new_row)
                break
    
    # st.write(addition_row)
    for j in addition_row:
        data.loc[len(data)] = j
    
    return data.reset_index(drop=True)


    
# Change SQL query to get names of different "sections" of Paint Shop (modify [a_line])
def get_plc_options(date_input, session=session):
    
    query = '''
    select distinct "PLC_Origin"
    from "VCCH"."CHB_HRES"."PPAS_ALARM"
    where "PLC_Origin" is not null
    and "StartEquipmentTime" between \'{}\' and \'{}\'    
    '''.format(
        date_input[0].strftime('%Y-%m-%d'),
        date_input[1].strftime('%Y-%m-%d')
    )

    plc_options = session.sql(query).to_pandas()
    st.dataframe(plc_options)
    output = []
    for plc in plc_options['PLC_Origin']:
        if plc in dic_plcs_to_sections:
            output.append(dic_plcs_to_sections[plc])
    st.write(output)
    return set(output)


# TO-DO: Rename [a_line] to appropriate column name defining the section of interest in Paint Shop
# TO-DO: If we don't need all columns, change * to names of columns of interest
@st.cache_data
def load_alarms_data(date_input, wd, session=session):

    st.write(dic_sections_to_plcs[wd])
    '''
    Load dataset of alarms.
    '''

    query = """
    select *
    from "VCCH"."CHB_HRES"."PPAS_ALARM"
    where "StartEquipmentTime" between \'{}\' and \'{}\'
    and "PLC_Origin" = \'{}\'
    order by "StartEquipmentTime" asc;
    """.format(
        date_input[0],
        date_input[1],
        dic_sections_to_plcs[wd]
    )
    

    df = session.sql(query).to_pandas()
    
    return df
    

# TO-DO: Rename 'a_line' and 'Line' to appropriate column name defining the section of interest in Paint Shop
def preprocess_data(df):

    df_preprocessed = df.copy()

    # Rename columns
    df_preprocessed.rename(
        columns={
            'StartEquipmentTime': 'Alarm Start', 
            'EndEquipmentTime': 'Alarm End',
        }, 
        inplace=True
    )

    cols_of_interest = ['Alarm Start', 'Alarm End']
    cols_secondary = [col for col in df_preprocessed.columns if col not in cols_of_interest]

    # Keep only necessary columns to calculate downtime
    df_preprocessed = df_preprocessed[cols_of_interest + cols_secondary]

    # Sort by start time and end time (if two alarms start at the same time, the shorter one will be displayed first)
    df_preprocessed = df_preprocessed.sort_values(['Alarm Start', 'Alarm End'])

    # If 2+ rows have same end times, keep the one with first start.
    # e.g. Alarm X (12:00 - 12:10), Alarm Y (12:05 - 12:10) => Y is fully contained within X
    df_preprocessed = df_preprocessed.drop_duplicates(['Alarm End'], keep='first')

    # If 2+ rows have same severity and start time, keep the one with the latest end time
    df_preprocessed = df_preprocessed.drop_duplicates(['Alarm Start'], keep='last')
    
    return df_preprocessed

def join_overlaps(df_original, key='Interval Start', start='Alarm Start', end='Alarm End'):

    # Overlapping time period problem in the event table:
    # https://towardsdatascience.com/overlapping-time-period-problem-b7f1719347db

    # |-------------|
    #   |---|                           
    #          |---------|
    #                       |---|

    # = 

    # |------------------|
    #                       |---| 
    
    df_no_overlaps = df_original.copy()
    # Start_or_End is an indicator for start or end
    startdf = pd.DataFrame({key: df_no_overlaps[key], 'Time': df_no_overlaps[start], 'Start_or_End': 1})
    enddf = pd.DataFrame({key: df_no_overlaps[key], 'Time': df_no_overlaps[end], 'Start_or_End': -1})
    # Concat and sort the whole thing by key and time
    mergedf = pd.concat([startdf, enddf]).sort_values([key, 'Time'])
    # Use cumulative_sum to find 
    mergedf['cumulative_sum'] = mergedf.groupby(key)['Start_or_End'].cumsum()
    # Assign new start date
    mergedf['New_Start'] = mergedf['cumulative_sum'].eq(1) & mergedf['Start_or_End'].eq(1)
    # Use cumulative_sum to assign group id
    mergedf['Group'] = mergedf.groupby(key)['New_Start'].cumsum()
    # group_id by choosing the start_date row
    df_no_overlaps['Group_ID'] = mergedf['Group'].loc[mergedf['Start_or_End'].eq(1)]
    # Keep min and max of downtime intervals
    df_no_overlaps = df_no_overlaps.groupby([key, 'Group_ID']).aggregate({start: min, end: max}).reset_index()
    df_no_overlaps.drop('Group_ID', axis=1, inplace=True)

    return df_no_overlaps

def remove_contained_intervals(df_interval):

    # When looking to fill in the information of a big interval where overlapping existed, two different steps will take place:

    # - Remove contained intervals: If an alarm Y occurs completely within the span of a previous alarms X, Y can be discarded 
    # as it does not contribute to the downtime total.

    # - Sort cascading alarms: When an alarm Y partially overlaps with a previous alarm X, but Y also extends forward in time, 
    # the partial overlap is not taken into consideration ("chopped") and only the extension of Y contributes to the downtime total (see explanation below).

    # Keep track of indices of rows whose interval is contained within another one
    idx_to_remove = []
    # Pick an alarm from latest to oldest
    for i in range(1, len(df_interval)+1):
        row = df_interval.iloc[-i]  # Alarm X
        end = row['Alarm End']

        # Look through alarms that triggered before (ordered chronologically already)
        for j in range(i+1, len(df_interval)+1):
            row_to_inspect = df_interval.iloc[-j]   # Alarm Y
            # If alarm X ends before alarm Y, then A fully 
            # occurs within the interval of Y and can be removed.
            
            if end <= row_to_inspect['Alarm End']:
                idx_to_remove.append(df_interval.index[-i])
                break
    
    # Drop alarms that are fully contained within others
    df_interval = df_interval.drop(idx_to_remove)

    return df_interval

def sort_cascading_alarms(df_interval):

    # Deal with cascading case ---> In DataFrame it looks like this
    #                                   Start       End
    # |-----------| A                 07:04:21    07:11:06
    #    |--------------------| B     07:05:00    07:33:57
    #                 |----------| C  07:13:11    07:34:03

    # = 
                                        
    # |-----------| A                 07:04:21    07:11:06
    #             |-----------| B     07:11:06    07:33:57
    #                         |--| C  07:33:57    07:34:03

    df_interval.reset_index(drop=True, inplace=True)
    # Save start of first alarm to input later (shift messes things up)
    first_start = df_interval['Alarm Start'].iloc[0]
    # Use preceding end time as new start time
    # https://stackoverflow.com/questions/54706007/pandas-fill-in-a-dataframe-column-with-a-serie-starting-at-a-specifc-index
    df_interval.loc[1:, 'Alarm Start'] = df_interval['Alarm End'].shift(1)[1:]
    df_interval.iloc[0, df_interval.columns.get_indexer(['Alarm Start'])] = first_start

    return df_interval

def reconstruct_intervals(df, df_reconstructed):

    # Deal with nans, add missing info ('Year', 'Month', 'Day', 'Week', 'Week Day', 'Origin', 'Comment', 'Severity')
    df_nans = df_reconstructed[df_reconstructed['Comment'].isna()]

    # Loop through intervals which failed to merge
    for _, (start, end) in df_nans[['Alarm Start', 'Alarm End']].iterrows():
        
        df_interval = df.loc[(df['Alarm Start'] >= start) & (df['Alarm End'] <= end)]
        df_interval = remove_contained_intervals(df_interval)
        df_interval = sort_cascading_alarms(df_interval)
        df_reconstructed = pd.concat([df_reconstructed, df_interval])

    # Once the interval has been reconstructed, remove the row with initial and
    # final interval times which was generated by the join_overlaps function.
    # These will contain the same NaNs as before since they do not match with alarms.
    df_reconstructed.dropna(subset='Comment', inplace=True)

    return df_reconstructed

def filter_interval(df_reconstructed, date_input, time_input_start, time_input_end):

    df_reconstructed['Interval Start'] = pd.Timestamp(date_input[0]).replace(hour=time_input_start.hour, minute=time_input_start.minute)
    df_reconstructed['Interval End'] = pd.Timestamp(date_input[1]).replace(hour=time_input_end.hour, minute=time_input_end.minute)
    within_interval_date = (df_reconstructed['Alarm Start'] >= df_reconstructed['Interval Start']) & (df_reconstructed['Alarm End'] <= df_reconstructed['Interval End'])
    df_reconstructed = df_reconstructed.loc[within_interval_date]

    # If an alarm starts before the interval end and ends after the interval end, treat the interval end as the upper limit
    df_reconstructed.loc[df_reconstructed['Alarm End'] >= df_reconstructed['Interval End'], 'Alarm End'] = df_reconstructed['Interval End']

    return df_reconstructed

def reconstruct_date_columns(df_reconstructed):

    df_reconstructed['Year'] = df_reconstructed['Alarm Start'].apply(lambda x: x.year)
    df_reconstructed['Month'] = df_reconstructed['Alarm Start'].apply(lambda x: x.month)
    df_reconstructed['Day'] = df_reconstructed['Alarm Start'].apply(lambda x: x.day)
    df_reconstructed['Week'] = df_reconstructed['Alarm Start'].apply(lambda x: x.week)
    df_reconstructed['Week Day'] = df_reconstructed['Alarm Start'].apply(lambda x: x.day_of_week + 1)

    return df_reconstructed

def clean_df_final(df_reconstructed, cols_of_interest, cols_secondary):

    # Sort by alarm end
    df_reconstructed.sort_values('Alarm End', inplace=True)    
    df_reconstructed.reset_index(drop=True, inplace=True)

    # Rename columns and order them
    cols = cols_of_interest[:2] + ['Downtime'] + cols_of_interest[2:] + cols_secondary + ['Year', 'Month', 'Day', 'Week', 'Week Day']
    df_reconstructed = df_reconstructed[cols]

    return df_reconstructed

def calculate_station_downtime(df_line, date_input, time_input_start, time_input_end, cols_of_interest, cols_secondary):

    # Turn overlapping alarms into a single "long" alarm
    df_no_overlaps = join_overlaps(df_line)
    
    min_num_faults = len(df_no_overlaps)    # Assume that any partial overlap is connected to a "root cause"=fault

    # Merge "not-overlapping" intervals with original intervals to match info.
    df_reconstructed = pd.merge(df_line, df_no_overlaps, how='left')

    # Fix when an interval is not matched, this means that 
    # 2+ alarms defined the interval with an overlap.
    df_reconstructed = reconstruct_intervals(df_line, df_reconstructed)

    # Filter by time interval
    # Define time range of interest for downtime calculation (e.g. working hours)
    df_reconstructed = filter_interval(df_reconstructed, date_input, time_input_start, time_input_end)

    if len(df_reconstructed) != 0:
        # Reconstruct date columns
        df_reconstructed = reconstruct_date_columns(df_reconstructed)
    
        # Calculate downtime (in s)
        df_reconstructed['Downtime'] = (df_reconstructed['Alarm End'] - df_reconstructed['Alarm Start']).apply(lambda x: x.total_seconds())
    
        # Apply final touches (e.g. sorting rows, columns, ...)
        df_reconstructed = clean_df_final(df_reconstructed, cols_of_interest, cols_secondary)

    max_num_faults = len(df_reconstructed)

    return df_reconstructed, min_num_faults, max_num_faults

def display_df(df_reconstructed):

    df_display = df_reconstructed.copy()
    df_display['Year'] = df_display['Year'].astype(str)
    df_display.index = np.arange(1, 1+len(df_display))
    st.dataframe(df_display, use_container_width=True)

def summarise_downtime(df_reconstructed):

    downtime_seconds = df_reconstructed['Downtime'].sum()
    downtime_minutes = downtime_seconds // 60
    remainder_seconds = downtime_seconds % 60
    downtime_hours = downtime_minutes // 60
    remainder_minutes = downtime_minutes % 60


    st.text(
        '{} second{}'.format(
            round(downtime_seconds),
            's' if round(downtime_seconds) != 1 else ''
        )
    )

    st.text(
        '= {} minute{}, {} second{}'.format(
            int(downtime_minutes),
            's' if int(downtime_minutes) != 1 else '',
            round(remainder_seconds),
            's' if round(remainder_seconds) != 1 else ''
        )
    )

    st.text(
        '= {} hour{}, {} minute{}, {} second{}'.format(
            int(downtime_hours),
            's' if int(downtime_hours) != 1 else '',
            round(remainder_minutes),
            's' if round(remainder_minutes) != 1 else '',
            round(remainder_seconds),
            's' if round(remainder_seconds) != 1 else ''
        )
    )

# def display_download_button(df_reconstructed, interval):

#     # Create a Pandas Excel writer using XlsxWriter as the engine.
#     filename = interval.replace(':', '')    # Replace for valid name file
#     buffer = BytesIO()   
    
#     with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
#         df_reconstructed.to_excel(writer)

#     st.download_button(
#         label='DOWNLOAD EXCEL FILE',
#         data=buffer,
#         file_name= filename + '.xlsx',
#         mime='application/vnd.ms-excel',
#         use_container_width=True
#     )

def plot_timeline(df, interval, grouping=True):

    # Plot timeline of errors
    # https://plotly.com/python/gantt/

    # ALTERNATIVE
    # fig = px.timeline(df, x_start='Alarm Start', x_end='Alarm End', y='Comment')
    # Currently runs into errors: https://community.plotly.com/t/is-plotly-express-timeline-down/76081

    df_gant = df.copy()
    df_gant.rename(columns={'Alarm Start': 'Start', 'Alarm End': 'Finish', 'Comment': 'Task'}, inplace=True)
    
    # Use color scales for different lengths of downtime:
    conditions = [df_gant['Downtime'] < 10, df_gant['Downtime'] < 60]
    replacements = ['< 10s', '10-60s']

    df_gant['Complete'] = np.select(conditions, replacements, default='> 60s')
    colors = {
        '< 10s': 'rgb (0, 255, 100)',
        '10-60s': (1, 0.9, 0.16),
        '> 60s': 'rgb (220, 0, 0)'
    }

    fig = ff.create_gantt(df_gant, colors=colors, show_colorbar=True, index_col='Complete', group_tasks=grouping, title='Alarm Errors --- {}'.format(interval))

    if not grouping:
        fig.update_yaxes(autorange='reversed') # otherwise tasks are listed from the bottom up

    st.plotly_chart(fig, use_container_width=True)

def plot_bar_alarms(df, column_name):

    df_bar = df.copy()

    value_counts = df_bar[column_name].value_counts().reset_index()
    value_counts.rename(columns={'index': column_name, column_name: 'count'}, inplace=True)
    fig = px.bar(value_counts, x=column_name, y='count', title='Alarms by ' + column_name)
    fig.update_layout(title_x=0.4)
    st.plotly_chart(fig, use_container_width=True)

def plot_pie_alarms(df, column_name):

    df_grouped = df.copy()

    df_grouped = df_grouped.groupby(column_name).agg({'Downtime': sum}).reset_index()
    fig = px.pie(df_grouped, names=column_name, values='Downtime', title='Downtime by ' + column_name)
    fig.update_layout(title_x=0.4)
    st.plotly_chart(fig, use_container_width=True)


### APP ###

st.set_page_config(layout='wide')
st.title('Andon Analysis (HRES)')

today = date.today()

for form in ['initial_filters_applied', 'latest_search', 'latest_df', 'lines_selected']:
    if form not in st.session_state:
        st.session_state[form] = False

#######################################################################################################################
#                                              SELECT FILTER PARAMETERS
#######################################################################################################################


st.header('1. TIME INTERVAL FOR ALARM SEARCH')
st.write('Your interval selection will be loaded once you click on the button below the options.')

with st.form('Initial Filters'):

    date_col, time_col_start, time_col_end = st.columns(3)

    # Date (Default: (yesterday, today))
    with date_col:
        date_input = st.date_input(
            'Select date',
            value=(today - timedelta(days=1), today)
        )

    # Time
    with time_col_start:
        time_input_start = st.time_input(
            'Select time interval start',
            value=pd.Timestamp(today).replace(hour=7)
        )

    with time_col_end:
        time_input_end = st.time_input(
            'Select time interval end',
            value=pd.Timestamp(today).replace(hour=15, minute=40)
        )

    # https://docs.streamlit.io/library/advanced-features/button-behavior-and-examples
    initial_filters_applied = st.form_submit_button(
        'USE THESE DATE AND TIME FILTERS', 
        use_container_width=True,
        on_click=click_form,
        args=('initial_filters_applied', True)
    )

# Check date is valid
if st.session_state['initial_filters_applied'] and len(date_input) != 2:
    st.write('Please select a start and end date. If you are only searching for a single day, enter day as end date too.')

#######################################################################################################################
#                                              SELECT LINE
#######################################################################################################################

if st.session_state['initial_filters_applied'] and len(date_input) == 2:

    st.header('2. LINE(S) TO ANALYSE')

    with st.form('Line Selection'):
        # Since in SQL first date is inclusive but the second is exclusive, add an extra day to end date to make both inclusive
        date_input_search = (date_input[0], date_input[1] + timedelta(days=1))
    
        line_options = get_plc_options(date_input_search)
        
        line_input = st.multiselect(
            label='Your selection will be loaded once you click on the confirmation button. Select one or multiple lines.',
            options=sorted(line_options)
        )
    
        lines_selected = st.form_submit_button(
            label='CONFIRM LINE SELECTION',
            on_click=click_form,
            args=('lines_selected', True)
        )

# Check line input
if st.session_state['lines_selected'] and len(line_input) == 0:
    st.write('Please select at least 1 line number.')

#######################################################################################################################
#                                              LOAD DATASET
#######################################################################################################################

# If filters are correct
if st.session_state['initial_filters_applied'] and len(date_input) == 2 and st.session_state['lines_selected'] and len(line_input) != 0:  

    # Do not filter by time straight away, date only
    # e.g.
    # |-----------------| (12:00-12:10)
    #         |-----------------| (12:05-12:15)
    # This would result in two intervals: 12:00-12:10 and 12:10-12:15
    # If filter had been applied initially for >= 12:10, the second interval would have been missed
    
    st.header('3. LOAD ALARMS DATA')
    
    
    search = '({} {} ... {} {}) - Lines: {}'.format(date_input[0], time_input_start, date_input[1], time_input_end, ' / '.join(line_input))
   
    # If interval has not just been searched, load data again
    if st.session_state['latest_search'] != search:
        with st.spinner('Loading alarm data...'):
            dfs = [load_alarms_data(date_input_search, line) for line in line_input]
            st.session_state['latest_search'] = search
            st.session_state['latest_dfs'] = dfs

    # Split results across tabs for each line
    if len(line_input) > 1:
        st.write('Click on the tabs below for analysing the different lines.')

    line_input_tabs = ['Line {}'.format(l) for l in line_input]

    
    for line_idx, tab in enumerate(st.tabs(line_input_tabs)):
        with tab: 
            df = st.session_state['latest_dfs'][line_idx]
            df_preprocessed = preprocess_data(df)

            if len(df_preprocessed) == 0:
                st.write('There are no alarms contributing to downtime within the selected time interval. Make sure the interval is correct and that the line exists.')
            else:
                cols_of_interest = ['Alarm Start', 'Alarm End', 'PLC']
                cols_secondary = [col for col in df_preprocessed.columns if col not in cols_of_interest]
    
                # Since the time interval can span over a few days, the initial interval is only within a day in order
                # to be able to remove overlaps within a single day. Later on it is defined as the interval of interest.
                df_preprocessed['Interval Start'] = df_preprocessed['Alarm Start'].apply(lambda x: pd.Timestamp(x).replace(hour=0, minute=0, second=0, microsecond=0))
                df_preprocessed['Interval End'] = df_preprocessed['Alarm Start'].apply(lambda x: pd.Timestamp(x).replace(hour=23, minute=59, second=59, microsecond=0))
                
                #######################################################################################################################
                #                                          CALCULATE DOWNTIME PER Station
                #######################################################################################################################
                
                plc_input = dic_sections_to_plcs[line_input[line_idx]]
                
                df_line = df_preprocessed[df_preprocessed['PLC_Origin'] == plc_input].copy()
                
                interval_dates = search[search.find('(') : search.find(')') + 1]
                interval_txt = '{} {}'.format(line_input_tabs[line_idx], interval_dates)
                
                #######################################################################################################################
                #                                            DOWNTIME CALCULATION
                #######################################################################################################################
                df_line = df_line[(df_line['EventLength'] >= 1) & (df_line['t_type'] != 'LEGAL_BREAK')]

                # filter out break time
                st.header(interval_txt)
                st.divider()
                # filter only Andon data
                df_andon = df_line[df_line["PLC_FB"] == 'FB_PRODIAG_CS_QUALITY_STOP'].reset_index(drop=True)
                # remove break time
                df_andon = remove_breaks(df_andon)
                st.write("===============breaks==removed====================")
                st.dataframe(df_andon)
                # groupby PLC instance
                stations = [i for i in df_andon['PLC_Instance'].unique()]
                # st.write(stations)
                
                downtime_station = {}
                pull_station = {}
                # st.write(stations)
                daily = set([df_andon['Alarm Start'][i].date() for i in range(len(df_andon))])
                # st.write(daily)
                
                for s in stations:
                    df_station = df_andon[df_andon['PLC_Instance'] == s]
                    
                    with st.spinner('Calculating downtime...'):
                        df_reconstructed, min_faults, max_faults = calculate_station_downtime(df_station, date_input, time_input_start, time_input_end, cols_of_interest, cols_secondary)
                        
                        downtime_by_day = {}
                        pull_by_day = {}
                        direction = "" # L or R
                        station_name = "" # 1-6
                        for d in daily: 
                            downtime = 0
                            pulls = 0
                            for i in range(len(df_reconstructed)):
                                direction = "" # L or R
                                if df_reconstructed['Alarm Start'][i].date() == d:
                                    downtime += round(df_reconstructed['Downtime'][i]/60, 2)
                                    pulls += 1
                            
                                # check left or right
                                if df_reconstructed['Comment'][i] == "Quality stop active right":
                                    direction = 'R'
                                else:
                                    direction = 'L'
                                    
                                downtime_by_day[str(d)] = downtime
                                pull_by_day[str(d)] = pulls
                                # roller bed number to station name
                                station_name = RB_to_station[s[len(s)-4:len(s)]]
                        downtime_station[station_name+direction] = downtime_by_day.copy()
                        pull_station[station_name+direction] = pull_by_day.copy()
                        
                            
                # most downtime station
                downtime_station = pd.DataFrame.from_dict(downtime_station)
                # total time by day line graph        
                
                # line and bar plots
                # bar chart when select multiple days
                if len(daily) > 1:
                    st.subheader('Downtime by stations (minutes)')
                    st.line_chart(downtime_station)
                    st.subheader('Downtime by number of pulls')
                    st.bar_chart(downtime_station)

                st.divider()
                # bar chart of just one day
                if len(daily) == 1:
                    daily_downtime_station = {}
                    for station, val in downtime_station.items():
                        daily_downtime_station[station] = val[str(list(daily)[0])]
                    st.subheader('Downtime by stations (minutes)')
                    st.bar_chart(daily_downtime_station)
                    daily_pull_station = {}
                    for station, val in pull_station.items():
                        daily_pull_station[station] = val[str(list(daily)[0])]
                    st.subheader('Downtime by number of pulls')
                    st.bar_chart(daily_downtime_station)

                col1, col2, col3 = st.columns(3)
                col1.metric(label="Most Downtime Station", value = downtime_station.max().idxmax())
                
                
                downtime_station['Total'] = downtime_station.sum(axis='columns')
                downtime_station.loc['Total'] = downtime_station.select_dtypes(np.number).sum()
                st.dataframe(downtime_station)
                
                
                
                
                    