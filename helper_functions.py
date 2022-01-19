# Data wrangling and manipulation
import pandas as pd
import numpy as np
import glob

# API 
from sodapy import Socrata

# import requests module 
import requests
import re

#web app utility 
import streamlit as st

@st.cache
def get_data():
    files = glob.glob('*.csv')
    data = pd.read_csv(files[0])
    
    return data

def pre_processing(df):
    
    index_  = 0
    
    while index_ < len(df):
        try:
            if re.match(r"(geo[^i]|[\w]+geom)", df.iloc[:, index_].name) != None:
                col_drop_nm = df.iloc[:, index_].name
                true_res = pd.json_normalize(df.iloc[:, index_])
                df = df.drop(col_drop_nm, axis=1)
                df[['type', 'coordinates']] = true_res
        except IndexError:
            break
        index_ +=1
    
    return df


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
    headers = ({'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'})
    results = requests.get(f'https://masscannabiscontrol.com/resource/{api}.json', headers=headers)

    # Convert to pandas DataFrame
    results_df = pd.read_json(results.content)
    results_df = pre_processing(results_df)
        
    # return final output 
    return results_df


def convert_to_csv(x):
    
    # convert dataframe to csv
    return x.to_csv().encode('utf-8')
    
@st.cache
def load_data():
    file = glob.glob('*.csv')
    df = pd.read_csv(file[1])
    
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