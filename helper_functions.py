# Data wrangling and manipulation
import pandas as pd
import numpy as np
import glob
import os

# import requests module 
import requests
import re
from bs4 import BeautifulSoup
import json
import time

#web app utility 
import streamlit as st


URL = "https://masscannabiscontrol.com/open-data/data-catalog/"
HEADERS = ({'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'})


@st.cache
def get_data():
    
    try:
        response = requests.get(URL, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)
    
    text = BeautifulSoup(response.text, 'html.parser')
    
    names = text.find_all('td', class_="column-2")
    links = text.find_all('a')
    
    api_endpoints = [ x.get('href') for x in links if x.get('href').endswith('json')]
    name = [x.text for x in names]
    
    data = list(zip(name, api_endpoints))
    
    return pd.DataFrame(data, columns=['Name', 'api_endpoints'])

def choose_dataset(table, name):
    """
    Create a function to take
    in an api endpoint and 
    output the results
    """

    # store api keys in a list
    idx = table[table.Name.str.find(name) != -1].index
    api = table['api_endpoints'][idx].values[0]
    
    #request json data
    results = requests.get(api, headers=HEADERS)

    # Convert to pandas DataFrame
    results_df = pd.json_normalize(results.json())
    # results_df = pre_processing(results_df)
        
    # return final output 
    return results_df


def convert_to_csv(x):
    
    # convert dataframe to csv
    return x.to_csv().encode('utf-8')
    
def load_data():
    
    cwd = os.getcwd()
    loc = '\data'
    
    Path = cwd + loc
    
    file = glob.glob(Path +'\*.csv')
    df = pd.read_csv(file[0])
    
    return df

def get_age_bins(target):
    age_bin = []
    for i in range(len(target)):
        try:
            if target['age'].iloc[i] > 71:
                age_bin.append('71 and over')
            elif target['age'].iloc[i] >= 59 and target['age'].iloc[i] <= 71 :
                age_bin.append('59 to 71')
            elif target['age'].iloc[i] >= 46 and target['age'].iloc[i] <= 58 :
                age_bin.append('46 to 58')
            elif target['age'].iloc[i] >= 33 and target['age'].iloc[i] <= 45 :
                age_bin.append('33 to 45')
            elif target['age'].iloc[i] >= 19 and target['age'].iloc[i] <= 32 :
                age_bin.append('19 to 32')
            else:
                age_bin.append("")
        except IndexError as e:
            print(e)  
    return age_bin

def frequnecy_table(i):
    freq_tab_district = pd.DataFrame()
    freq_tab_district = pd.crosstab(index=i, columns='count').sort_values(by='count',ascending=False)
    freq_tab_district['Relative Frequency'] = (freq_tab_district/freq_tab_district.sum())*100
    freq_tab_district['Cumulative Frequency'] = freq_tab_district['Relative Frequency'].cumsum()
    return freq_tab_district