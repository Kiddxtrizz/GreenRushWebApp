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
    
    with st.container():
        st.header("The Proposal")
        with st.expander("Click to learn More"):
            st.markdown("The purpose of this web application is to highlight the need for a tool to help streamline the location selection, permitting, and licensing process for (Social Equity, Economic Empowerment, Boston Equity) program participants in Massachusetts.")
    
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
            
            c = st.dataframe(data[data[col_selector].str.find(ads) != -1], 1450,600)
        else:
            st.dataframe(data,1450,600)
        

        csv = convert_to_csv(results[results[col_selector].str.find(ads) != -1])
        
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
    df = df.reset_index(drop=True).dropna()
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
    dff = df_segm[['dest_long', 'dest_lat', 'zip_code','population_denisty', 'median_household_income','total_quantity','total price', 'sex','distance','DOB_year','age','Settlement_size', 'segment', 'labels']]
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
            c1,c2,c3 = st.columns(3)
            pass
            # c1.metric("Population", df_segm_analysis[df_segm_analysis['N obs'] == chocies])
            
            with col1:
                st.markdown("## Consumer Segment Breakdown")

                st.dataframe(dff[dff['labels'] == chocies], height=600, width=800)
                
#                 fig = px.pie(df_segm_analysis, values='Prop Obs', names=df_segm_analysis.index[:4])
#                 st.plotly_chart(fig, sharing="streamlit")
                
                # st.dataframe(dff[['zip_code', 'population_denisty', 'median_household_income', 'sex','distance','DOB_year','age','Settlement_size', 'labels']][dff['labels'] == chocies], height=500, width=800)
                with st.expander("Visual Filter Shelf"):
                        menuu = st.selectbox("Choose One", ['Age Distribution', 'Zipcode Distribution (Pareto Chart)']) 
                ages = dff[dff['age'] > 0]
                ages = get_age_bins(dff)
                dff['age_bin'] = ages
                if menuu == 'Age Distribution':
                    fig = px.histogram(dff, x="age_bin", color="sex", marginal='box')
                    st.plotly_chart(fig, sharing='streamlit')
                else:
                    def frequnecy_table(i):
                        freq_tab_district = pd.DataFrame()
                        freq_tab_district = pd.crosstab(index=i, columns='count').sort_values(by='count',ascending=False)
                        freq_tab_district['Relative Frequency'] = (freq_tab_district/freq_tab_district.sum())*100
                        freq_tab_district['Cumulative Frequency'] = freq_tab_district['Relative Frequency'].cumsum()
                        return freq_tab_district
                    
                    
                    zip_freq = frequnecy_table(dff['zip_code'])
                    
                    st.dataframe(zip_freq)
                    
                    
                    # fig, ax = plt.subplots()
                    # ax.bar(zip_freq.index, zip_freq["count"], color="C0")
                    # ax2 = ax.twinx()
                    # ax2.plot(zip_freq.index, zip_freq["Cumulative Frequency"], color="C1", marker="D", ms=7)
#                     ax2.yaxis.set_major_formatter(PercentFormatter())

#                     ax.tick_params(axis="y", colors="C0")
#                     ax2.tick_params(axis="y", colors="C1")

#                     st.pyplot(fig)


            with col2:

                fig = px.scatter_mapbox(dff[dff['labels'] == chocies], lat='lat', lon='lon', color=dff[dff['labels'] == chocies]['total price'], 
                                       height = 1100,
                                       width = 800,
                                        size = dff[dff['labels'] == chocies]['total_quantity'],
                                       opacity = 0.5,
                                       color_continuous_scale=px.colors.diverging.Portland , size_max=15)

                fig.update_layout(
                    mapbox_style="mapbox://styles/kiddtrizz/cky3fdrla2fun15nzclck4mdu", 
                    mapbox_accesstoken="pk.eyJ1Ijoia2lkZHRyaXp6IiwiYSI6ImNreTNoeHNiZTAyN3czMm8wYmthZXh1Z3oifQ.5_xkXfXF1EXDnORZIn40Xw"
                )

                st.plotly_chart(fig, sharing='streamlit')
                
        elif checkbox == "Target":
            

            #Purchase & Order Occassion
            labeler = LabelEncoder()
            purchase_df = df_segm.copy()
            purchase_df['brand_1'] = labeler.fit_transform(purchase_df['brand_1'])
            purchase_df = purchase_df.drop(['product_cat'], axis=1)
            
            temp1 = purchase_df[['ID', 'incidence']].groupby(['ID'], as_index= False).count()
            temp1 = temp1.set_index('ID')
            temp1 = temp1.rename(columns = {'incidence': "N_visits"})
            
            temp2 = purchase_df[['ID', 'total_quantity']].groupby(['ID'], as_index = False).sum()
            temp2 = temp2.set_index('ID')
            temp2 = temp2.rename(columns = {'incidence': 'Lifetime_order_totals'})
            temp3 = temp1.join(temp2)
            
            temp3['Avg_Order_quant'] = temp3['total_quantity']/ temp3['N_visits']
            
            temp4 = purchase_df[['ID', 'total price']].groupby(['ID'], as_index = False).sum()
            temp4 = temp4.set_index('ID')
            temp4 = temp4.rename(columns = {'incidence': 'Total spend'})
            temp5 = temp3.join(temp4)
            
            temp6 = purchase_df[['ID', 'segment']].groupby(['ID'], as_index=False).min()
            temp6 = temp6.set_index('ID')
            df_purchase_desc = temp5.join(temp6)
            
            
            
            seg_pop = df_purchase_desc[['N_visits', 'segment']].groupby(['segment']).count()/df_purchase_desc.shape[0]
            seg_pop = seg_pop.rename(columns = {'N_Purchases': 'Segment Proportion'}, index ={0:'Working Professional', 1:'Well-Off', 2:'Standard', 3:'Working Professional'})
            
            #Purchase incidence
            segs_mean = df_purchase_desc.groupby(['segment']).mean()
            segs_std = df_purchase_desc.groupby(['segment']).std()

            
            #brand choice
            df_purchase_incidence = df_segm[df_segm['incidence'] == 1]
            brand_dummies = pd.get_dummies(df_purchase_incidence['brand_1'], prefix='Brand', prefix_sep='_')
            brand_dummies['segment'], brand_dummies['ID'] = df_purchase_incidence['segment'], df_purchase_incidence['ID']
            
            temp5 = brand_dummies.groupby(['ID'], as_index = True).mean()
            mean_brand_choice = temp5.groupby(['segment'], as_index=True).mean()
            
            brand_seg_revenue = pd.DataFrame()
            
            for i in df_segm['brand_1']:
                tempss = purchase_df[purchase_df['brand_1'] == i]
                tempss.loc[:, f'Revenue Brand {i}'] = tempss['item_price_1'] * tempss['quant_1']
                brand_seg_revenue[['segment', f'Revenue Brand {i}']] = tempss[['segment', f'Revenue Brand {i}']].groupby(['segment'], as_index=False).sum()
              
            
            st.dataframe(df_purchase_desc)
            
            fig = px.pie(seg_pop, values='N_visits', names=seg_pop.index[:4])
            st.plotly_chart(fig, sharing="streamlit")
            
            with st.container():
                
                col1,col2,col3 = st.columns(3)

                with col1:
                    pass
                    # fig, ax = plt.subplots
                    # plt.figure(figsize = (10,10))
                    # plt.bar(x = (0,1,2),
                    # tick_label = ('Working Professional', 'Standard', 'Career Focused'),
                    # height = segs_mean['total_quantity'],
                    # yerr = segs_std['total_quantity'],
                    # color = ('b', 'orange', 'g'))
                    # st.pyplot(fig)
           


        else:
            purchase_df = df_segm.copy()
            segment_dummies = pd.get_dummies(df_segm['segment'], prefix= 'Segment', prefix_sep='_')
            df_pa = pd.concat([purchase_df, segment_dummies], axis=1)
            df_pa = df_pa.dropna()
            df_pa['total price'] = pd.to_numeric(df_pa['total price'])

            y = df_pa['incidence']
            X = pd.DataFrame()
            X['Mean_Price'] = (df_pa['total price'])/(sum(df_pa['total price']))
            
            price_range = np.arange(np.min(df_pa['total price']), np.max(df_pa['total price']), 0.01)
            df_price_rng = pd.DataFrame(price_range)
            Y_pr = model_purchase.predict_proba(df_price_rng)
            purchase_proba = Y_pr[:][:,1]
            pe = model_purchase.coef_[:, 0] * price_range * (1 - purchase_proba)
            df_price_elas = pd.DataFrame(price_range)
            df_price_elas =  df_price_elas.rename(columns = {0: 'Price Point'})
            df_price_elas['Mean_PE'] = pe
            
else:
    st.empty()
    st.header("About Me")
    st.markdown("My name is Trey W. I'm a Boston (Roxbury) Native who has enjoyed over 20 years in the beautiful city of Boston, and I couldn't be luckier. To top that, I have had the pleasure of attending Northeastern University and obtaining the title of Double Husky after completing both my undergraduate and graduate degrees from the prestigious university. During my graduate studies, I saw a chance to apply my unique perspective, critical thinking skills, and entrepreneurial mindset to various projects and tasks. As a result, I have decided to create a place to showcase my thoughts, ideas, and projects. I hope you enjoy it! Feel free to contact me if you have any questions.")
# # except Exception as e:
# #     st.error('What happened')
# # except requests.exceptions.HTTPError as errh:
# #     st.error("Http Error:",errh)
# # except requests.exceptions.ConnectionError as errc:
# #     st.error("Error Connecting:",errc)
# # except requests.exceptions.Timeout as errt:
# #     st.error("Timeout Error:",errt)
# # except requests.exceptions.RequestException as err:
# #     st.error("OOps: Something Else",err)

    