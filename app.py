from helper_functions import get_data, choose_dataset, convert_to_csv
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
    #get data
    api_table = get_data()

    #create drop-down menu
    menu_1 = st.sidebar.selectbox('Select an item from the drop-down below', list(api_table.Name))

    #store data 
    st.write("#### You selected", menu_1)
    results = choose_dataset(api_table, menu_1)

    col_selector = st.sidebar.multiselect(
            "Filter the view", list(results.columns), list(results.columns)

    )
    if not col_selector:
        st.error("Please select at least one element")
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

except Exception as e:
    st.error('What happened', e)
# except requests.exceptions.HTTPError as errh:
#     st.error("Http Error:",errh)
# except requests.exceptions.ConnectionError as errc:
#     st.error("Error Connecting:",errc)
# except requests.exceptions.Timeout as errt:
#     st.error("Timeout Error:",errt)
# except requests.exceptions.RequestException as err:
#     st.error("OOps: Something Else",err)

    