from helper_functions import get_data, choose_dataset, convert_to_csv
import streamlit as st
import time
import requests
import re

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


menu = ["Home", "Search", "Market Research", "About"]
choice = st.sidebar.selectbox("Webpage Menu", menu)

if choice == "Home":
    st.empty()
    st.markdown("# Welcome to My Homepage" +" "+ "👋🏾 👊🏾 🧑🏽‍🚀  👨🏾‍💻 👨🏾‍🎨 ")
    st.sidebar.header("My Name is what?!!?")
    
    
elif choice == "Search":
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

            
    page_names = ['Table View', 'Visual Creator']
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
            st.dataframe(data[data[col_selector].str.find(ads) != -1], 1450,600)
        else:
            st.dataframe(data,1450,600)
        
        
        csv = convert_to_csv(results)

        st.download_button(
            label = "Download File",
            data = csv,
            file_name = 'results.csv',
            mime = 'text/csv'
        )
        
    else:
        with st.expander("Click to view filters"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1: 
                col_selector = st.selectbox("Select a column to view", list(results.columns))
                if not col_selector:
                    st.error("Please select at least one element")
        
        if len(data[col_selector].unique()) <= 20:
            st.bar_chart(data[col_selector].value_counts() , 1450,600)
        else:
            st.write('Way too many elements to display')
            
elif choice == 'Market Research':
    st.empty()
    st.header("Coming Soon")

else:
    st.empty()
    st.header("About")

# except Exception as e:
#     st.error('What happened')
# except requests.exceptions.HTTPError as errh:
#     st.error("Http Error:",errh)
# except requests.exceptions.ConnectionError as errc:
#     st.error("Error Connecting:",errc)
# except requests.exceptions.Timeout as errt:
#     st.error("Timeout Error:",errt)
# except requests.exceptions.RequestException as err:
#     st.error("OOps: Something Else",err)

    