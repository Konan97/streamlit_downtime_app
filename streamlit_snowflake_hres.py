### IMPORT ###
import plotly.figure_factory as ff
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

from snowflake.snowpark.context import get_active_session
from datetime import date, timedelta

### GLOBAL VARIABLES ###
session = get_active_session()
#AA safety AB power supply AC equipment ADEF warnings
# conveyer(PC) process(PP) application(PA) material(PM)
# PC1432, NVH, PA1343 no connections, PM1261 waiting for upgrade,
# PP1361
dic_plcs_to_sections = {
    'PC1392': 'PRIMER',
    'PC1191': 'PTED',
    'PA1210': 'TOPCOAT',
    'PC1431': 'PRIMER',
    'PC1245': 'SEALING',
    'PA1604': 'SEALING',
    'PA1603': 'SEALING',
    'PA1214': 'TOPCOAT',
    'PC1593': 'TOPCOAT',
    'PP1459': 'TOPCOAT',
    'PM1361': 'PRIMER',
    'PA1222': 'TOPCOAT',
    'PP1359': 'PRIMER',
    'PA1601': 'SEALING',
    'PP1451': 'TOPCOAT',
    'PP1249': 'SEALING',
    'PA1220': 'TOPCOAT',
    'PC1194': 'REPAIR',
    'PA1200': 'PRIMER',
    'PP1341': 'PRIMER',
    'PP1441': 'TOPCOAT',
    'PA1443': 'TOPCOAT',
    'PC1297': 'SEALING',
    'PC1112': 'PTED',
    'PA1602': 'SEALING',
    'PC1192': 'PTED',
    'PP1128': 'PTED',
    'PC1492': 'TOPCOAT',
    'PP1111': 'PTED',
    'PP1911': 'REPAIR',
    'PM1461': 'TOPCOAT',
    'PM1361': 'PRIMER',
    'PC1122': 'PTED',
    'PP1121': 'PTED',
    'PP1642': 'TOPCOAT',
    'PC1190': 'HBS',
    'PP1125': 'PTED',
    'PC1595': 'HBS',
    'PC1597': 'HBS'
}

dic_sections_to_plcs = {}


### HELPER FUNCTIONS ###

# Change state after clicking on form
# https://discuss.streamlit.io/t/form-submit-buttons-on-click-being-called-before-clicking/27767
def click_form(key, boolean):
    st.session_state[key] = boolean

# TO-DO: Change SQL query to get names of different "sections" of Paint Shop (modify [a_line])

def get_plc_options(date_input, session=session):
    
    query = '''
    select distinct "PLC"
    from "VCCH"."CHB_HRES"."PPAS_ALARM"
    where "PLC" is not null
    and "StartEquipmentTime" between \'{}\' and \'{}\'    
    '''.format(
        date_input[0].strftime('%Y-%m-%d'),
        date_input[1].strftime('%Y-%m-%d')
    )

    plc_options = session.sql(query).to_pandas()
    
    output = []
    for plc in plc_options['PLC']:
        if plc in dic_plcs_to_sections:
            output.append(dic_plcs_to_sections[plc])
    return set(output)

def get_plcs_from_section():

    for section in dic_plcs_to_sections.values():

        dic_sections_to_plcs[section] = [p for p,s in dic_plcs_to_sections.items() if s == section]

    return dic_sections_to_plcs 

# TO-DO: Rename [a_line] to appropriate column name defining the section of interest in Paint Shop
# TO-DO: If we don't need all columns, change * to names of columns of interest
@st.cache_data
def load_alarms_data(date_input, line, session=session):

    dic_sections_to_plcs = get_plcs_from_section()
    
    '''
    Load dataset of alarms.
    '''

    query = """
    select *
    from "VCCH"."CHB_HRES"."PPAS_ALARM"
    where "StartEquipmentTime" between \'{}\' and \'{}\'
    and "PLC" in {}
    order by "StartEquipmentTime" asc;
    """.format(
        date_input[0],
        date_input[1],
        tuple(dic_sections_to_plcs[line])
    )
    

    df = session.sql(query).to_pandas()

    return df

# split df by type of PLCs conveyer(PC) process(PP) application(PA) material(PM)
# return df
def df_by_type(df_line, typeName):
    if typeName == 'Total':
        return df_line
    return df_line[df_line['PLC'].str.contains(typeName)]
    

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
st.title('Downtime Analysis (HRES)')

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

    line_input = [str(l) for l in line_input]
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
                #                                          CALCULATE DOWNTIME PER LINE
                #######################################################################################################################
                dic_sections_to_plcs = get_plcs_from_section()
                plc_input = []
                for plc in dic_sections_to_plcs[line_input[line_idx]]:
                    plc_input.append(plc)
                
                df_line = df_preprocessed[df_preprocessed['PLC'].isin(plc_input)].copy()
                
                interval_dates = search[search.find('(') : search.find(')') + 1]
                interval_txt = '{} {}'.format(line_input_tabs[line_idx], interval_dates)
                
                #######################################################################################################################
                #                                            DOWNTIME CALCULATION
                #######################################################################################################################
                df_line = df_line[df_line['EventLength'] >= 1]
                st.header(interval_txt)
                # conveyer(PC) process(PP) application(PA) material(PM)
                typeDict = {'Total': '', 'PC':'Conveyor system', 'PP':'Process related', 'PA':'Application related', 'PM':'Materials related'}
                for type, des in typeDict.items():
                    st.header(des + ' (' +type+')')
                
                    # df_andon = df_line[df_line["PLC_FB"] == 'FB_PRODIAG_CS_QUALITY_STOP']
                    st.write("=====================================")
    
                    with st.spinner('Calculating downtime...'):
                        df_reconstructed, min_faults, max_faults = calculate_station_downtime(df_by_type(df_line, type), date_input, time_input_start, time_input_end, cols_of_interest, cols_secondary)
    
                    if len(df_reconstructed) == 0:
                        st.write('There are no alarms in the specified interval.')
    
                    else:
    
                        # Display DataFrame results
                        st.write(
                            '''
                            - The alarms occurring during the specified interval and contributing to the downtime 
                            are shown in the table below. 
                            - You can make the table full-screen by hovering over it 
                            and clicking the button that shows up in the upper right corner. You can sort columns
                            by clicking on the column name.
                            - Downtime is calculated in seconds.
                            '''
                        )
        
                        display_df(df_reconstructed)
    
                        # Create list of comments and number of occurences to display in selectbox
        
                        comment_downtimes = df_reconstructed[['Comment', 'Downtime']].groupby('Comment').agg(Downtime=('Downtime', np.sum), Count=('Downtime', np.size))
                        comment_downtimes = comment_downtimes.sort_values('Downtime', ascending=False)
                        comment_downtimes['Downtime'] = comment_downtimes['Downtime']//60
                        downtimes = [str(round(v[0])) for v in comment_downtimes.values]
                        num_occurrences = [str(int(v[1])) + ' occurrences' if v[1] > 1 else str(int(v[1])) + ' occurrence' for v in comment_downtimes.values]

                        chart_data = pd.DataFrame(
                            comment_downtimes,
                            columns=comment_downtimes.columns)
                        
                        st.bar_chart(chart_data)
                        
                        # e.g., Not In Automatic (155963 seconds --- 6 occurrences)
                        comment_options = comment_downtimes.index + ' (' + downtimes + ' seconds --- ' + num_occurrences + ')'
                        
                        alarms_to_pick = st.multiselect(
                            label='If there are any alarms you want to take into consideration for calculating downtime, select them below. They are sorted by the amount of downtime they generate.',
                            options=comment_options
                        )
                    
                        # Get actual comments: e.g., Process Overtime (363979 seconds --- 31 occurrences) -> Process Overtime
                        if len(alarms_to_pick) != 0:
                            alarms_to_pick = [a[:a.rfind('(')-1] for a in alarms_to_pick]  
                            df_reconstructed = df_reconstructed[df_reconstructed['Comment'].isin(alarms_to_pick)]
        
                        # Text summary of table
                        summary_columns = st.columns(2)
    
                        # TECHNICAL AVAILABILITY
                        # = (available time - downtime) / available time
                        with summary_columns[0]:   
                            st.header(type + ' Downtime in Interval')
                            summarise_downtime(df_reconstructed)
    
                        # LIMITATION (st.download_button):- https://docs.snowflake.com/en/developer-guide/streamlit/limitations
                        # with summary_columns[1]:
                        #     st.header('Download Table of Alarms')
                        #     display_download_button(df_reconstructed, interval_txt)
        
                        #######################################################
                        #                         PLOTS
                        #######################################################
                        
                        # Timeline
                        st.header('Timeline of Alarm Errors')
                        st.write(
                            '''
                            - You can make the chart full-screen by hovering over it 
                            and clicking the button that shows up in the upper right corner.
                            - You can also click and drag on the chart to select a time interval of interest.
                            '''
                        )
                        
                        plot_timeline(df_reconstructed, interval_txt)
        
                        # Charts
                        # chart_cols = st.columns(2)
        
                        # for col_idx, ratio_type in enumerate(['Quantity', 'Downtime']):
                        #     with chart_cols[col_idx]:
                        #         st.header('{} of Alarms'.format(ratio_type))
        
                        #         if ratio_type == 'Quantity':
                        #             quantity_box = st.selectbox(
                        #                 label='Select column to filter number of alarms by.',
                        #                 options=cols_secondary
                        #             )
        
                        #             plot_bar_alarms(df_reconstructed, quantity_box)
                                
                        #         else:
        
                        #             downtime_pie = st.selectbox(
                        #                 label='Select column to filter downtime of alarms by.',
                        #                 options=cols_secondary
                        #             )
        
                        #             plot_pie_alarms(df_reconstructed, downtime_pie)