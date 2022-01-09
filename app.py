from helper_functions import get_data, choose_dataset, convert_to_csv, load_data
import streamlit as st
import time
import requests
import re
import pickle
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

api_token = 'pk.eyJ1Ijoia2lkZHRyaXp6IiwiYSI6ImNrbGk4dHp2cjRsZXAycG5yZTZ5cGFqNHIifQ.x5o6rqASgACRb4fuTSYkYg'

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
    st.markdown("# Welcome to My Homepage" +" "+ "👋🏾 👊🏾")
    st.balloons()
    
    with st.container():
        st.header("The Proposal")
        with st.expander("Click to learn More"):
            st.markdown("The purpose of this web application is to highlight the need for a tool to help streamline the location selection, permitting, and licensing process for (Social Equity, Economic Empowerment, Boston Equity) program participants.")
            st.markdown("As a social equity participant alum, I have first-hand experience with the rigors of securing a location for a business. The process is cumbersome and expensive,\
    from utilizing outdated PDFs of zoning maps (if they exist) to relying solely on the municipality's bylaws/ordinances or other obscure sources. Although these resources exist to help facilitate economic development, a predominant portion are dispersed amongst web pages, paper files, and overburdened town officials. Making the process time-consuming, inefficient, and confounding for SE/EE/BE participants and other entrepreneurs looking to enter the market." )
            st.markdown("To resolve this problem, I have consolidated much of this information into a one-stop web and mobile platform that centralizes and captures relevant market and municipal information to help provide support to and underserved majority trying to entrer a burgeoning industry.")
    
    st.header("Problems to Address")
    with st.expander("Click to learn More"):
        string = open('Needs.md').read()
        st.markdown(string)
    
    st.header("Project Overview")
    with st.expander("Click to learn More"):
        st.image('Picture3.png')
    
    st.header("Evaluation")
    with st.expander("Click to learn More"):
        st.markdown("The team will be implementing SCRUM methodology to track and evaluate success. The team will produce a minimum viable product to showcase to all stakeholders after each sprint. To ensure the progress is tracked – a burndown chart will be used to mark every successful sprint and maintain the velocity of each sprint.")
    
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
        
#         if len(data[col_selector].unique()) <= 20:
#             st.bar_chart(data[col_selector].value_counts() , 1450,600)
#         else:
#             st.write('Way too many elements to display')
            
elif choice == 'Market Research':
    #load in data
    df = load_data()
    
    # #create pipeline
    # pipe = Pipeline([('Standardize', StandardScaler()),
    #                 ('PCA', PCA()),
    #                 ('Kmeans', KMeans())])
    
    #load models in
    scaler = pickle.load(open('scaler.pickle', 'rb'))
    pca = pickle.load(open('pca.pickle', 'rb'))
    kmeans_pca = pickle.load(open('kmeans_pca.pickle', 'rb'))
    
    # get features to transform 
    features = df[['dest_long', 'dest_lat', 'zip_code','population_denisty', 'median_household_income', 'sex','distance','DOB_year','age','Settlement_size']]
    
    #standardize
    df_seg_scaled = scaler.transform(features)
    
    #Reduce dimesionality 
    df_seg_pca = pca.transform(df_seg_scaled)
    
    #Cluster data points
    df_kmeans_pca = kmeans_pca.predict(df_seg_pca)
    
    #combine it altogether 
    df_segm = df.copy()
    df_segm['segment'] = df_kmeans_pca 
    df_segm['labels'] = df_segm['segment'].map({0:'Career Focused', 1:'Standard', 2:'Well-Off', 3:'Working Professional'})
    
    #
    dff = df_segm[['dest_long', 'dest_lat', 'zip_code','population_denisty', 'median_household_income', 'sex','distance','DOB_year','age','Settlement_size', 'segment', 'labels']]
    dff = dff.rename(columns={'dest_long': 'lon','dest_lat': 'lat'})
    df_segm_analysis = dff.groupby(['segment']).mean()
    df_segm_analysis['N obs'] = df_segm[['segment', 'age']].groupby(['segment']).count()
    df_segm_analysis['Prop Obs'] = df_segm_analysis['N obs']/df_segm_analysis['N obs'].sum()
    df_segm_analysis = df_segm_analysis.rename({0:'Working Professional', 1:'Standard', 2:'Career Focused', 3:'Well-Off'})
    
    with st.container():
        col1,col2 = st.columns(2)
        
        chocies = st.sidebar.selectbox("Choose a Segment to Examine", list(dff['labels'].unique()))
        checkbox = st.sidebar.radio("STP Selection", ('Segment', 'Target', 'Position'))
        
        # st.map(dff[['lon', 'lat']])
        if checkbox == "Segment":
            with col1:
                st.markdown("## Consumer Segment Breakdown")

                # st.dataframe(df_segm_analysis[['population_denisty', 'median_household_income', 'sex','distance','DOB_year','age','Settlement_size']], height=1450, width=800)
                
                fig = px.pie(df_segm_analysis, values='Prop Obs', names=df_segm_analysis.index[:4])
                st.plotly_chart(fig, sharing="streamlit")
                
                # st.dataframe(dff[['zip_code', 'population_denisty', 'median_household_income', 'sex','distance','DOB_year','age','Settlement_size', 'labels']][dff['labels'] == chocies], height=500, width=800)
                with st.expander("Visual Filter Shelf"):
                        menuu = st.selectbox("Choose One", ['Age Distribution', 'Zipcode Distribution (Pareto Chart)'])
                    
                

            with col2:

                fig = px.scatter_mapbox(dff[dff['labels'] == chocies], lat='lat', lon='lon', color='labels', 
                                       height = 1100,
                                       width = 800,
                                        )

                fig.update_layout(
                    mapbox_style="mapbox://styles/kiddtrizz/cky3fdrla2fun15nzclck4mdu", 
                    mapbox_accesstoken="pk.eyJ1Ijoia2lkZHRyaXp6IiwiYSI6ImNreTNoeHNiZTAyN3czMm8wYmthZXh1Z3oifQ.5_xkXfXF1EXDnORZIn40Xw"
                )

                st.plotly_chart(fig, sharing='streamlit')
        elif checkbox == "Target":
            st.write("yes")

        else:
            st.write("yes")

else:
    st.empty()
    st.header("About Me")
    st.markdown("My name is Trey W. I'm a Boston (Roxbury) Native who has enjoyed over 20 years in the beautiful city of Boston, and I couldn't be luckier. To top that, I have had the pleasure of attending Northeastern University and obtaining the title of Double Husky after completing both my undergraduate and graduate degrees from the prestigious university. During my graduate studies, I saw a chance to apply my unique perspective, critical thinking skills, and entrepreneurial mindset to various projects and tasks. As a result, I have decided to create a place to showcase my thoughts, ideas, and projects. I hope you enjoy it! Feel free to contact me if you have any questions.")
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

    