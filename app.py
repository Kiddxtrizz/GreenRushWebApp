from helper_functions import get_endpoints, get_dataset_name, choose_dataset, convert_to_csv
import streamlit as st
import time
import requests

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




st.sidebar.title('The Green Rush')
st.sidebar.subheader('Democratizing Data one repo at a time.')

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

for i in range(0,101):
    status_text.text("%i%% Complete" % i)
    progress_bar.progress(i)
    time.sleep(0.05)
    
progress_bar.empty()
status_text.empty()

try:
    #collect api endpoints
    api_links = get_endpoints('https://opendata.mass-cannabis-control.com/browse')

    #extract
    api_table = get_dataset_name(api_links)

    #create drop-down menu
    menu_1 = st.sidebar.selectbox('Select and item from the drop-down below', list(api_table.Name))

    #store data 
    st.write("#### You selected", menu_1)
    results = choose_dataset(api_table, menu_1, limit=2000)

    col_selector = st.sidebar.multiselect(
            "Filter the view", list(results.columns), list(results.columns)

    )
    if not col_selector:
        st.error("Please select at least one country")
    else:
        data = results
        st.dataframe(data[col_selector], 1450,600)
        
    csv = convert_to_csv(results)

    st.download_button(
        label = "Download File",
        data = csv,
        file_name = 'results.csv',
        mime = 'text/csv'
    )

except requests.exceptions.HTTPError as errh:
    st.error("Http Error:",errh)
except requests.exceptions.ConnectionError as errc:
    st.error("Error Connecting:",errc)
except requests.exceptions.Timeout as errt:
    st.error("Timeout Error:",errt)
except requests.exceptions.RequestException as err:
    st.error("OOps: Something Else",err)

    