# Data wrangling and manipulation
import pandas as pd
import numpy as np

# API 
from sodapy import Socrata

# import requests module 
import requests
import re
from bs4 import BeautifulSoup
import time

#web app utility 
import streamlit as st

@st.cache
def get_endpoints(url):
   
    # We need a URl to start the process
    # Make a request to the URL 
    # Check for status code 200... 

    prefix = url #store url 
    
    try:
        response = requests.get(prefix)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)
        
    time.sleep(5)
    
    """
    Use bs4 to build a grapg to loop 
    through the available webpages and 
    capture relevant links with API endpoints.
    """
    # store the content from the url
    # into a variable (e.g. content)
    context = response.content
    
    #parse native HTML tags
    soup = BeautifulSoup(context, "html.parser")
    
        
    # Create Variables to store results from loop     
    pages = list()
     

    """
    Get the next page prefix 
    
    """
    for links in soup.find_all('a'):
        #if type is equal to str 
        if type(links.get('href')) == str:
             #if string contains dev or opendata 
            if re.findall(r'[/browse]\W+(page)', links.get('href') ):
                pages.append(links.get('href'))
    
    
    """
    Create a while loop
    to capture all of the 
    endpoints
    
    """
    
    
    count= 0
    api_endpoints = list()
    # opendata = list()
    page_sort = sorted(list(pages))
    
    while count < len(set(pages)):
    
        try:
            webpage = requests.get(prefix[:-7]+page_sort[count])
            webpage.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print ("Http Error:",errh)
        except requests.exceptions.ConnectionError as errc:
            print ("Error Connecting:",errc)
        except requests.exceptions.Timeout as errt:
            print ("Timeout Error:",errt)
        except requests.exceptions.RequestException as err:
            print ("OOps: Something Else",err)
            
        content = webpage.content
        apis = BeautifulSoup(content, "html.parser")


        for api in apis.find_all('a'):
        #if type is equal to str 
            if type(api.get('href')) == str:
             #if string contains dev or opendata 
                if re.findall(r'\w+\W+(foundry)', api.get('href') ):
                    api_endpoints.append(api.get('href'))
                # elif re.findall(r'\w+\W+(opendata)', api.get('href') ):
                #     api_endpoints.append(api.get('href'))

        count += 1
        time.sleep(8)
    
    """
    Parse the links 
    to get the coveted 
    enpoints
    """
    
    ep = [endpoints[-9:].strip() for endpoints in api_endpoints]
    
    return ep

@st.cache
def get_dataset_name(x):
    
    """
    Create a function to pull 
    in the name of the dataset
    for the corresponding api endpoint 
    """
    
    #formatting for the final table
    #want to display the full name 
    #of the dataset
    pd.options.display.max_colwidth = 200
    
    #setup a basic client
    # authenticated client (needed for non-public datasets):
    client = Socrata("opendata.mass-cannabis-control.com", None)
    
    # list comprehension to capture relevant metadata (i.e. Name)
    dataset_name = [client.get_metadata(y)['name'] for y in x]
    
    # combine the api-endpoints
    # with the name of the assoc.
    # dataset 
    data = list(zip(x, dataset_name))
    
    #store final result into dataframe
    api_table = pd.DataFrame(data, columns=['api_endpoints', 'Name']).drop_duplicates().reset_index(drop=True)

    return api_table

# def pre_processing(df):
    
#     index_  = 0
    
#     while index_ < len(df):
#         try:
#             if re.match(r"(geo[^i]|[\w]+geom)", df.iloc[:, index_].name) != None:
#                 col_drop_nm = df.iloc[:, index_].name
#                 true_res = pd.json_normalize(df.iloc[:, index_])
#                 df = df.drop(col_drop_nm, axis=1)
#                 df[['type', 'coordinates']] = true_res
#         except IndexError:
#             break
#         index_ +=1
    
#     return df


def choose_dataset(table, name, limit):
    """
    Create a function to take
    in an api endpoint and 
    output the results
    """

    #setup a basic client
    client = Socrata("opendata.mass-cannabis-control.com", None)
    
    # get columns
    name = name
    
    # store api keys in a list
    idx = table[table.Name.str.find(name) != -1].index
    api = table['api_endpoints'][idx].values[0]
    
    #set limit 
    limit = 2000

    # Pull data via api enpoint 
    results = client.get(f"{api}", limit=limit)

    # Convert to pandas DataFrame
    results_df = pd.DataFrame.from_records(results)
    # results_df = pre_processing(results_df)
        
    # return final output 
    return results_df


def convert_to_csv(x):
    
    # convert dataframe to csv
    return x.to_csv().encode('utf-8')
    
    
