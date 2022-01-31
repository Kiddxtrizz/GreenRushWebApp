from helper_functions import get_data, choose_dataset, convert_to_csv, load_data,get_age_bins
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import streamlit as st
import time
import requests
import re
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


st.set_page_config(
    page_title = "The Green Rush",
    page_icon = "random",
    layout = "wide",
    initial_sidebar_state = "auto",
    menu_items = {
        
        'Get Help': "https://support.socrata.com/hc/en-us",
        'Report a bug': "https://support.socrata.com/hc/en-us/requests/new",
        'About': "Just a simple web app"
    }

)

# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()

# for i in range(0,101):
#     status_text.text("%i%% Complete" % i)
#     progress_bar.progress(i)
#     time.sleep(0.05)

# progress_bar.empty()
# status_text.empty()

st.empty()
st.sidebar.title('The Green Rush')
st.sidebar.subheader('Democratizing Data one repo at a time.')

#get data
api_table = get_data()

#create drop-down menu
menu_1 = st.sidebar.selectbox('Select the Dataset you\'re interested in', list(api_table.Name))

#store data 
st.write("#### You selected", menu_1)
results = choose_dataset(api_table, menu_1)

# with st.sidebar.expander("Filter View"):
#     for i in results.columns:
#         st.selectbox(i, results[i])

data = results

# with st.expander('Dataset filters'):
#     for i in range(len(data.columns)):
#         st.selectbox(data.columns[i], data.iloc[:,i], key='selector')


page_names = ['Table View', 'Visual Creator', 'Dashboard']
page = st.sidebar.radio('Navigation', page_names)

if page == 'Table View': 
    with st.expander("Click to view filters"):
        col1, col2, col3, col4 = st.columns(4)

    with col1: 
        col_selector = st.selectbox("Select a column header", list(results.columns))
        if not col_selector:
            st.error("Please select at least one element")

    with col2: 
        ads = st.text_input("Advanced Search: Please select a column header first", value="")

    if ads != "":
        try:
            c = st.dataframe(data[data[col_selector].str.find(ads) != -1].reset_index(drop=True), 1450,600)
        except AttributeError as e:
            st.error("Please choose a word and not a number in the filter shelf")
    else:
        st.dataframe(data,1450,600)

    try:
        data_csv = results[results[col_selector].str.find(ads) != -1].reset_index(drop=True)
        csv = convert_to_csv(data)
    except AttributeError as ae:
        csv = convert_to_csv(data)

    st.download_button(
        label = "Download File",
        data = csv,
        file_name = 'results.csv',
        mime = 'text/csv')

elif page == 'Visual Creator':
    st.sidebar.empty()
    vsc = st.sidebar.radio('Select Chart Type', options=['Bar', 'Line'])

    if vsc == 'Bar':
        with st.expander("Click to view filters"):
            col1, col2, col3, col4 = st.columns(4)

            with col1: 
                x = st.selectbox("Select a column for x-axis", list(results.columns))

            with col2:
                y = st.selectbox("Select a column for y-axis", list(results.columns))

        with st.container():
            c= results.groupby([x]).count()
            value_array = []

            fig = px.bar(data, x=x, y=list(range(len(data[y]))), title=f"{x} by {y}", 
                           height = 800,
                           width = 1450,)
            fig.update_yaxes(type='category')
            st.plotly_chart(fig, sharing='streamlit')
    else:
        with st.expander("Click to view filters"):
            col1, col2, col3 = st.columns(3)

            with col1: 
                x = st.selectbox("Select a column for x-axis", list(results.columns))

            with col2:
                y = st.selectbox("Select a column for y-axis", list(results.columns))

        with st.container():
            fig = px.line(data, x=x, y=y, title=f"{x} by {y}", 
                           height = 800,
                           width = 1450,)
            st.plotly_chart(fig, sharing='streamlit')
else:
    pass

        
#         if len(data[col_selector].unique()) <= 20:
#             st.bar_chart(data[col_selector].value_counts() , 1450,600)
#         else:
#             st.write('Way too many elements to display')
            
