import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as pltpwd
import matplotlib.pyplot as plt

 

# loading Dataset 
df = pd.read_csv('usedcars_dataset.csv')
st.title("Used Car Analysis")

list = ['Introduction','Feature and price relationship', 'Normalized-Losses and Make', 'Risk Level', 'Speed and the weight of the car', 'Intercorrelation Heatmap']

page = st.sidebar.radio('Select the page to view', list)

from PIL import Image
img = Image.open("usedcar.jpeg")



if page == 'Introduction':
    
    st.write("△ Market for used cars is increasing day by day and a consumer always wants the best resale price of his car.")
    
    st.write("△ Using our dataset of used car, we aimed to clarify some of these unknown variables and provide an easier experience for those searching for a car.")
    st.image(img)

    

if page == 'Feature and price relationship':
    st.subheader("Co-relationship between price and other features") 
    
    flist1 = ['horsepower','highway-mpg', 'symboling', 'engine-size']
    feature1 = st.selectbox("Select feature 1",(flist1))
    sc_plot = alt.Chart(df).mark_circle(size=60).encode(
    x=alt.X(feature1),
    y=alt.Y('price'),
    tooltip=[feature1, 'price'],
        #color='blue'
    ).interactive()
    
    st.write(sc_plot)
 

    
if page == 'Normalized-Losses and Make':
    
    df2 =df[df['normalized-losses'].notna()]
    df3 = df2.groupby(["make"]).mean()
    df3=df3.sort_values(by=['normalized-losses'],ascending=False)
    df3 = df3.reset_index()
    
    st.subheader("Normalized-Losses and Make") 
    radioButton = st.radio(
     "What do you want to show?",
     ('Introduction', 'Visualization'))
    if radioButton == 'Introduction':
        st.subheader("Normalized-losses is the relative average loss payment per insured vehicle year. This value is normalized for all autos within a particular size classification, and represents the average loss per car per year. ")
    if radioButton == 'Visualization':
        fig = px.bar(df3, x='make', y='normalized-losses', color='make', title="Bar chart of normalized losses and make")
        st.plotly_chart(fig)
        

        fig2 = px.scatter(df, 
                 x="normalized-losses", 
                 y="make",
                 size="price", 
                 color="make",
                 hover_name="normalized-losses", 
                 size_max=60)
        st.plotly_chart(fig2)

        
        box_fig = px.box(df2, x='make', y='normalized-losses', title="Box plot of normalized losses and make")
        st.write(box_fig)

        
        
if page == 'Risk Level':
    st.subheader("Insurance Risk Level")
    st.markdown("**Using these charts, we could find the _low risk engine-type/ make/ body-style_**")
    f1 = st.radio("Select one among these:", ['engine-type', 'make','body-style'])
    f2 = st.selectbox("Select a feature on which you want to evaluate the risk level:", ['symboling', 'price'])
    
    if f1 == 'engine-type':
        dfe = df.groupby([f1]).mean()
        dfe=dfe.sort_values(by=[f2],ascending=False)
        dfe = dfe.reset_index()
        fige = px.bar(dfe, x=f1, y=f2, color=f1, title="Bar chart of "+f1+" and "+f2)
        st.plotly_chart(fige)
    
    if f1 == 'make':
        dfe = df.groupby([f1]).mean()
        dfe=dfe.sort_values(by=[f2],ascending=False)
        dfe = dfe.reset_index()
        fige = px.bar(dfe, x=f1, y=f2, color=f1, title="Bar chart of "+f1+" and "+f2)
        st.plotly_chart(fige)
     
    if f1 == 'body-style':
        dfe = df.groupby([f1]).mean()
        dfe=dfe.sort_values(by=[f2],ascending=False)
        dfe = dfe.reset_index()
        fige = px.bar(dfe, x=f1, y=f2, color=f1,title="Bar chart of "+f1+" and "+f2)
        st.plotly_chart(fige)
        
        

# Heatmap Visualization 
if page == 'Prediction of city gas mileage':
    st.subheader("Prediction of city gas mileage")
    st.markdown("**_Under_ Progress**")
    corr = df.corr()
    sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
    plt.figure(figsize=(13,7))
    a = sns.heatmap(corr, annot=True, fmt='.2f')
    rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
    roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
    st.write(a)

    
    
if page == 'Speed and the weight of the car':
    
    radioButton = st.radio("What do you want to show?",('Higest mpg', 'Lowest mpg'))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if radioButton == 'Higest mpg':
         g = sns.lmplot('highway-mpg',"curb-weight", df, hue="make",fit_reg=False);
         
         st.pyplot()
    if radioButton == 'Lowest mpg':
         g = sns.lmplot('city-mpg',"curb-weight", df, hue="make",fit_reg=False);
         st.pyplot()
   
    
    

if page=='Intercorrelation Heatmap':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header('Intercorrelation Matrix Heatmap')
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f,ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()
