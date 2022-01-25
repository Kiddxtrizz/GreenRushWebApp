import streamlit as st
import plotly.express as px
from datetime import datetime
import pandas as pd
import requests
from streamlit import caching
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')

BASE_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey={}'.format(API_KEY)



st.set_page_config(
    page_title = "My First Web App XD",
    page_icon = "random",
    layout = "wide",
    initial_sidebar_state = "expanded"
)


@st.cache()
def get_data(url):
    r = requests.get(url)
    data = r.json()
    
    return data

st.empty()

st.title("Milestone Project: My Stock Ticker")
st.markdown("**Created By: Trey W.**")
st.sidebar.title("Navigation")
search = st.sidebar.text_input("Search")

if search == '':
    data = get_data(BASE_URL)

    last_rfsh = data['Meta Data']['3. Last Refreshed']
    symbl = data['Meta Data']['2. Symbol']
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.write('Last Refresh: {}'.format(last_rfsh))
    
    with c2:
        st.write('Current Symbol: {}'.format(symbl))
        
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'],orient='index')
    df = df.sort_index()
    
    df['date'] = pd.to_datetime(df.index)
    df[['Year', 'Week', 'Day']] = df['date'].dt.isocalendar()
    df['Month'] = df['date'].dt.month
    df['Month Name'] = df['date'].dt.month_name()
    
    st.sidebar.markdown('### Filter Shelf')
    
    Year = st.sidebar.selectbox("Year filter", df['Year'].unique())
    Month = st.sidebar.selectbox("Month filter", df['Month Name'].unique())
    
    temp_df = df[(df['Year'] == Year) & (df['Month Name'] == Month)]
    temp_df['2. high'] = temp_df['2. high'].astype(float)
    temp_df['5. volume'] = temp_df['5. volume'].astype(float)
    
    fig = px.line(temp_df, x=temp_df.index, y='2. high')
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig)

    fig1 = px.bar(temp_df, x=temp_df.index, y='5. volume')
    st.plotly_chart(fig1)
    
else:
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}'.format(search,API_KEY)
    r = requests.get(url)
    
    data = r.json()
    
    try:
        last_rfshed = data['Meta Data']['3. Last Refreshed']
        symbl = data['Meta Data']['2. Symbol']
        
        c11, c22, c33 = st.columns(3)
    
        with c11:
            st.write('Last Refresh: {}'.format(last_rfshed))

        with c22:
            st.write('Current Symbol: {}'.format(symbl))
        
        
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'],orient='index')
        df = df.sort_index()

        df['date'] = pd.to_datetime(df.index)
        df[['Year', 'Week', 'Day']] = df['date'].dt.isocalendar()
        df['Month'] = df['date'].dt.month
        
        st.write

        st.sidebar.markdown('### Filter Shelf')

        Year = st.sidebar.selectbox("Year filter", df['Year'].unique()) 
        Month = st.sidebar.selectbox("Month filter", df['Month'].unique())

        temp_df = df[(df['Year'] == Year)]

        st.write(temp_df)
        
        
        fig = px.line(temp_df, x=temp_df.index, y='2. high')
        fig.update_xaxes(
            dtick="M1",
            tickformat="%b\n%Y",
            ticklabelmode="period")
        st.plotly_chart(fig)
        
        fig1 = px.bar(temp_df, x=temp_df.index, y='5. volume')
        st.plotly_chart(fig1)
        
    except (KeyError,NameError):
        st.error("Symbol does not exisit")
        

