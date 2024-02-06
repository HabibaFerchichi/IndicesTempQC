# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:23:51 2024

@author: habibaferchichi
"""

import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st 
import numpy as np # pip install streamlit
from PIL import Image

import hmac
import streamlit as st
####### Add global password #################################
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Main Streamlit app starts here

################################################### application ######################################################

#st.set_page_config(page_title='Indices thermiques')
#st.header('Indices thermiques au Québec')
#st.subheader('Example riviere Ouelle')

### --- LOAD DATAFRAME
# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
logo= Image.open('data/logo.png')
st.set_page_config(page_title="Thermal metrics Dashboard", page_icon=logo, layout="wide")

################################################### load all indices of Quebec rivers#####################
# Load thermal indices of all databases
# with open('results_dic/thermalindices_finalDF_CorrectionClimat.pkl', 'rb') as handle:
#     allindices_res = pickle.load(handle)
allindices_res = pd.read_pickle('results_dic/thermalindices_finalDF_CorrectionClimat.pkl')

#nb rivers
#len(allindices_res['Riv'].unique())
# rename columns in frensh and without accent otherwise affect the sxported csv after
#change columns names
#allindices_res.columns
allindices_res1 = allindices_res.rename(columns={
                              'StationName':'Station','longitudeT':'longitude', 
                              'latitudeT':'latitude', 'year':'an',
                              'indices':'Indice','value':'Valeur_Indice','nb_years':'Nb_ans',
                              'Quality_ThermalIndices':'Qualite_Indice','Climate_Class':'Classe_Climatique',
                              'MissingData(%)':'Donnees manquantes(%)','Obsperyear':'Nb_obs',
                               'Data_season':'Saison_donnees','obs_summer':'NB_obsEstivale',
                               'definition_fr':'Definition','Qualité_série':'Qualite_serieT',
                                "Maximum annuelle de température d'air maximale": "Maximum annuelle de temperature d'air maximale (°C)",
                               "Moyenne annuelle de température d'air maximale":"Moyenne annuelle de temperature d'air maximale (°C)",
                               "Moyenne annuelle de Précipitation totale":"Moyenne annuelle des Precipitations totales (mm)"
                               })
df= allindices_res1[['Riv', 'Station', 'an','Classe_Climatique','Indice',
       'Valeur_Indice', 'Definition', 'Qualite_Indice', 'Nb_ans', 'Qualite_serieT',
       "Maximum annuelle de temperature d'air maximale (°C)",
       "Moyenne annuelle de temperature d'air maximale (°C)",
       "Moyenne annuelle des Precipitations totales (mm)",
       'Donnees manquantes(%)', 'Nb_obs', 'Saison_donnees', 'NB_obsEstivale',
       'longitude', 'latitude']]
# change the name of quality index
df['Definition']= df['Definition'].replace("Indice de qualité d'eau adapté","Indice composite de tolérance thermique")
cols=['an','Valeur_Indice', 'Nb_ans', 
       "Maximum annuelle de temperature d'air maximale (°C)",
       "Moyenne annuelle de temperature d'air maximale (°C)",
       "Moyenne annuelle des Precipitations totales (mm)",
       'longitude', 'latitude']
df[cols] = df[cols].astype('float')
# sort values by river names and years
df = df.sort_values(["Riv","an"])
# load stations statistics of all databases
# with open('results_dic/allriversStat_resFinal.pkl', 'rb') as handle:
#     allstat_res = pickle.load(handle)
allstat_res = pd.read_pickle('results_dic/allriversStat_resFinal.pkl')

#df = pd.read_excel("ouelle_themalindices_CorrectionClimat.xlsx")
#del df[df.columns[0]]

df=df.dropna()
#group by station name and order the ans: must done in the previous code
# Reset the index

#df["an"] = df['an'].astype(str)
df["an"] = df['an'].astype(int)

#st.dataframe(df)

# set log for side bar
#st.sidebar.image("logo.png",caption="INDICES THERMIQUES AUX RIVIÈRES DU QUEBEC",use_column_width ='always')    

# ---- SIDEBAR ----
# Increase text size using CSS styling
st.sidebar.header("Filtrer ici:")

Riv = st.sidebar.selectbox(
    "Sélectionner une rivière:",
    options=df["Riv"].unique())

m= df ["Station"].unique()

station = st.sidebar.multiselect(
    "Sélectionner une station:",
    options = np.append('',m) ,
    default= m[0]
)

all_options = st.sidebar.checkbox("Sélectionner tout")

if all_options:
    station = st.sidebar.multiselect(
    "Sélectionner une station:",
    options = m ,
    default= m)

# # Filter the DataFrame based on selected stations
df_selection1 = df.query("Riv == @Riv & Station == @station " )

# ---- MAINPAGE ----

col1, mid, col2 = st.columns([1,2,40])
with col1:
    st.image('data/Logo_bleu.png', width=60)
    #st.image(logo, width=50)
with col2:
    st.title("INDICES THERMIQUES DES RIVIÈRES DU QUÉBEC " )
st.markdown('<style>div.block-container{padding-top:3rem;}</style>',unsafe_allow_html=True)

    

# Plot 1: bar chart of WQI
# Filter the DataFrame to keep only rows with WQI
#wqi_df = df_selection1[df_selection1['indice'] == 'wqi']
#wqi_df = df_selection1[df_selection1['Definition'] == "Indice composite de tolérance thermique"]

#if df_selection1.empty:
#    st.warning("No data available for the selected Riv.")
#else:
    # Create bar chart of WQI (fixed and independent of other filters)
 #   fig = px.bar(wqi_df, x='an', y='Valeur_Indice', color='Station', barmode='group',
 #                width=600,height=650, template = "seaborn")
 #   #st.plotly_chart(fig)

#### Ad other filters

indice = st.sidebar.selectbox("Sélectionner un indice:", options=df_selection1 ["Definition"].unique())

df_selection = df_selection1.query(
    "Definition == @indice")

#add the defintion of indices
def get_indice_definition(indice_name):
    definition = df_selection[df_selection["Definition"] == indice_name]["Definition"].iloc[0]
    return definition
# Add the definition of the selected indice to the selection bar
indice_definition = get_indice_definition(indice)
st.sidebar.subheader("Indice selectionné")
st.sidebar.info(f"{indice_definition}")

# add bution of choosing variable
var1 = ['Tmax','Tmean','Tmin']
varT = st.sidebar.selectbox(
     "Sélectionner une série temporelle:",
     options = var1)

# Plot 1: bar chart of selected index
# Filter the DataFrame to keep only rows with WQI
if df_selection.empty:
    st.warning("No data available for the selected Riv.")
else:
    # Create bar chart of WQI (fixed and independent of other filters)
    fig = px.bar(df_selection, x='an', y='Valeur_Indice', color='Station', barmode='group',
                 width=600,height=650, template = "seaborn")

################################ Add new plot of Theorical Gaussian fitting using a,b,c parameters ######################
#add filter of year to plot gaussian fitting for this year
# x= df_selection1 ["an"].unique()
# an_selec  = st.sidebar.multiselect(
#     "Sélectionner une année:",
#     options = np.append('',x) ,
#     default= x[0])
an_selec = st.sidebar.selectbox("Proprietés du régime thermique: Sélectionner une année:", options=df_selection1 ["an"].unique())
df_selection2 = df_selection1.query(
    "an == @an_selec")
#extract data of gaussian parameters
gauss_df = df_selection2[df_selection2['Indice'].isin(['a','b','c'])]
# round values of b and c (duree de saison chaude et jour de l'occurence de valeur maximale)
gauss_df.loc[gauss_df['Indice'].isin(['b','c']),'Valeur_Indice']=gauss_df.loc[gauss_df['Indice'].isin(['b','c']),'Valeur_Indice'].round(0)
# add filter year
#an1 = 2003
#gauss1 = gauss_df[gauss_df['an']==an1]
a= gauss_df.loc[gauss_df["Indice"]=='a','Valeur_Indice'].values[0]
b= gauss_df.loc[gauss_df["Indice"]=='b','Valeur_Indice'].values[0]
c= gauss_df.loc[gauss_df["Indice"]=='c','Valeur_Indice'].values[0]
d= np.arange(1,366)
# Plot the fitted curve using Plotly Express
y= a*np.exp(-0.5*((d-c)/b)**2)
var_df= pd.DataFrame({'jour':d,'y':y})
jour_selec=[]
for i in range (0,len(var_df)):
    j= var_df['jour'][i]
    selec= j + b
    if selec<366:
      val= var_df.loc[var_df['jour']==selec,'y'].values[0]
      jour_selec.append({'jour': j, 'selec': selec, 'y': var_df['y'][i], 'val': val})

df_jour= pd.DataFrame(jour_selec)
df_jour['y']= round(df_jour['y'],0)
df_jour['val']= round(df_jour['val'],0)
selected_rows = df_jour[df_jour['y'] == df_jour['val']]
b_info= selected_rows.tail(1)

fig_final = px.line(x=d, y=y, labels={"x": "Jour de l'année", "y": "Temperature (°C)"},
                    title='Ajustement Gaussien sur les températures moyennes journalieres')
# Change the color of legend label to white
fig_final.add_annotation(x=c, y=a, text='La température maximale interannuelle', arrowhead=2, showarrow=True, arrowcolor="white")
fig_final.add_annotation(x=c, y=0, text="Le jour de l'occurrence de Tmax interannuelle", arrowhead=2, showarrow=True, arrowcolor="white",ax=40, ay=-20)
import plotly.graph_objects as go
fig_final.add_trace(go.Scatter(x=[c,c], y=[0, a], mode="lines", line=dict(color="green", width=2, dash="dash"),showlegend=False))
fig_final.add_trace(go.Scatter(x=[b_info['jour'].values[0],b_info['selec'].values[0]],
                               y=[b_info['val'].values[0], b_info['val'].values[0]],
                               mode="lines+markers", line=dict(color="red", width=2),showlegend=False))
fig_final.add_annotation(x=b_info['selec'].values[0], y=b_info['val'].values[0], 
                         text='La duree de la saison chaude', arrowhead=2, 
                         showarrow=True, arrowcolor="white",ax=90, ay=20)


############################################### Plot 2: Map of index values for all stations ###################################################
# # Add Scatter Plot
from plotly.subplots import make_subplots
# Scatter Plot
scatter_fig = px.line(df_selection, x='an', y="Maximum annuelle de temperature d'air maximale (°C)", color='Station', markers=True, template = "seaborn")
scatter_fig1 = px.line(df_selection, x='an', y="Moyenne annuelle de temperature d'air maximale (°C)", color='Station', markers=True, template = "seaborn")

# Bar Plot
bar_fig = px.bar(df_selection, x='an', y="Moyenne annuelle des Precipitations totales (mm)", color='Station', template = "seaborn")

# Box Plot
scatter_fig2 = px.line(df_selection, x='an', y='Valeur_Indice', color='Station', markers=True, template = "seaborn")

# Combine the plots into one figure with shared x-axis
fig1 = make_subplots( vertical_spacing=0.1,
     horizontal_spacing = 0,shared_xaxes=True,
    rows=4, cols=1, row_width=[0.5, 0.3, 0.3,0.3],
    subplot_titles=("Moyenne de température de l'air maximale (°C)",
                    "Température de l'air maximale annuelle (°C)",
                    "Moyenne annuelle des Précipitations totales (mm)",indice))

# Add Scatter Plot
for trace in scatter_fig1.data:
    fig1.add_trace(trace, row=1, col=1)
for trace in scatter_fig.data:
    fig1.add_trace(trace, row=2, col=1)
# Add Bar Plot
for trace in bar_fig.data:
    fig1.add_trace(trace, row=3, col=1)
# Add scatter Plot
for trace in scatter_fig2.data:
    fig1.add_trace(trace, row=4, col=1)

# update axis 
# Update yaxis properties
# fig1.update_yaxes(title_text="Température de l'air(°C)", row=1, col=1)
fig1.update_yaxes(title_text="Température de l'air(°C)", row=1, col=1)
fig1.update_yaxes(title_text="Précipitation(mm)",  row=3, col=1,autorange='reversed')
fig1.update_yaxes(title_text="Valeur",row=4, col=1)
fig1.update_layout(width=800, height=650)#showlegend=False,,width=700, height=600
fig1.update_xaxes(title_text="Année",row=4, col=1)


######################################################### Plot 3: Map of index Valeur_Indices for all stations #########################################
##  add administrative regions:
import geopandas as gpd
import plotly.express as px
#import plotly.graph_objects as go

# Load your shapefile (replace 'regions_shapefile.shp' with your shapefile path)
regions = gpd.read_file('data/ZGIEBV.shp')
fixed_strings = [s.encode('latin1').decode('utf-8') for s in regions['ZGIE']]
regions['ZGIE']=fixed_strings
regions2= regions[['OBJECTID', 'NO_ZGIEBV', 'ZGIE',
       'ZGIE_KM2', 'Shape_Leng', 'Shape_Area', 'geometry']]
# simplify geometry to 1000m accuracy
regions2["geometry"] = (
    regions2.to_crs(regions2.estimate_utm_crs()).simplify(1000).to_crs(regions2.crs)
)
regions2.dropna(axis=0, subset="geometry", how="any", inplace=True)
regions2.set_index("ZGIE")
# Reproject to WGS84 (EPSG:4326)
regions2 = regions2.to_crs(epsg=4326)
# Convert regions to GeoJSON
geojson = regions2.__geo_interface__

map_fig1 = px.choropleth_mapbox(regions2,
                              geojson = regions2.geometry,
                              locations = regions2.index,
                              color='ZGIE',
                              #center={"lat": -33.865143, "lon": 151.209900},
                                mapbox_style="carto-positron", 
                                zoom=2.5,
                              width = 800,
                              height = 500,
                              opacity=0.05,  # Adjust opacity as needed # Use the color map
                              hover_data={'ZGIE': False},
                              custom_data=['ZGIE'])
map_fig1.update_traces(hovertemplate='%{customdata[0]}')

# Update the layout of the combined figure to set the map center on quebec province
map_fig1.update_layout(mapbox_center={'lat': 53, 'lon': -70})
#map_fig1.update_traces(hovertemplate='Region: %{customdata[0]})
map_fig1.update_layout(showlegend=False,mapbox_style="carto-positron",coloraxis_colorbar=dict(title=''))
map_fig1.update_layout(coloraxis_colorbar=dict(title="Valeur de l'indice"))

# Put a  condition on th color palette < for IPCC red to bue, for for the rest blue to red
colors=[[0.0, "rgb(103,0,31)"],
                [0.1111111111111111, "rgb(178,24,43)"],
                [0.2222222222222222, "rgb(214,96,77)"],
                [0.3333333333333333, "rgb(244,165,130)"],
                [0.4444444444444444, "rgb(253,219,199)"],
                [0.5555555555555556, "rgb(209,229,240)"],
                [0.6666666666666666, "rgb(146,197,222)"],
                [0.7777777777777778, "rgb(67,147,195)"],
                [0.8888888888888888, "rgb(33,102,172)"],
                [1.0, "rgb(5,48,97)"]]
if indice=='Indice composite de tolérance thermique':
    selected_colors = colors
else:
    selected_colors = [[0.0,"rgb(5,48,97)"],
                [0.1111111111111111, "rgb(33,102,172)"],
                [0.2222222222222222,  "rgb(67,147,195)"],
                [0.3333333333333333, "rgb(146,197,222)"],
                [0.4444444444444444,  "rgb(209,229,240)"],
                [0.5555555555555556,"rgb(253,219,199)"],
                [0.6666666666666666, "rgb(244,165,130)"],
                [0.7777777777777778, "rgb(214,96,77)"],
                [0.8888888888888888,"rgb(178,24,43)"],
                [1.0,"rgb(103,0,31)"]]

df_selection['Valeur_Indice'] = pd.to_numeric(df_selection['Valeur_Indice'], errors='coerce')
map_fig = px.scatter_mapbox(
    df_selection.sort_values("an"),
    lat="latitude",
    lon="longitude",
    hover_name="Station",
    hover_data=["Definition", "Valeur_Indice"],
    color="Valeur_Indice",
    size= "Valeur_Indice",
    zoom=8,  # Adjust the zoom level to focus on Quebec
    #center={"lat": 54, "lon": -74},  # Center the map on Quebec
    height=600,  
    #width=1000,
    color_continuous_scale= selected_colors,
    #color_continuous_scale=["blue", "orange", "red"],
    animation_frame="an", animation_group="Valeur_Indice",mapbox_style="carto-positron")
    #size_max = 30 (default =20))
hovertemplate_backup = map_fig.data[0].hovertemplate
map_fig.update_traces(hovertemplate=hovertemplate_backup)

map_fig.update_layout(showlegend=False,mapbox_style="carto-positron",coloraxis_colorbar=dict(title=''))
map_fig.update_layout(coloraxis_colorbar=dict(title="Valeur de l'indice"))

# map_fig.data[0].hovertemplate = hovertemplate_backup
for i in range(len(map_fig1.data)):
    map_fig.add_trace(map_fig1.data[i])

# re-order traces so scatter is at top
map_fig.data = map_fig.data[::-1]



######### Statistics table:  Calculate minimum and maximum Valeur_Indices for selected indices
# Group the data by Riv and station names
stations= df_selection['Station'].unique()
res=[]
for s in stations:
  dat= df_selection[df_selection.Station==s]
  dat1 = dat[(dat.Valeur_Indice==dat.Valeur_Indice.min())|(dat.Valeur_Indice==dat.Valeur_Indice.max())]
  #Find the index of the row with the minimum value
  min_index = dat1['Valeur_Indice'].idxmin()
  # Find the index of the row with the maximum value
  max_index = dat1['Valeur_Indice'].idxmax()
  # Create a DataFrame with rows containing minimum and maximum values
  # Add row names for minimum and maximum values
  dat1.index = ['Minimum_indice' if idx == min_index else 'Maximum_indice' for idx in dat1.index]
  dat1['max_min']= ['Minimum_indice' if idx == min_index else 'Maximum_indice' for idx in dat1.index] 
  dat2 = dat1.sort_values('Valeur_Indice')
  res.append(dat2)
  
grouped_data= pd.concat(res)

# Reset the index to flatten the grouped data
#grouped_data = grouped_data.reset_index()

############################## Display the table and figures ############################

col1,col2= st.columns(2)
with col1:
   st.subheader("Variation de l'indice thermique sélectionné")
   #st.markdown("### Variation of selected thermal index")
   col1.plotly_chart(fig1,use_container_width=True)
with col2:
   st.subheader("Variation de l'indice thermique sélectionné")
   col2.plotly_chart(fig,use_container_width=True)#use_container_width=True

#ADD the download buttons for data
df_selection = df_selection.reset_index()
df_selection = df_selection.drop(columns=["index"])
with col2:
    with st.expander("Afficher Données"):
        st.write(df_selection.style.background_gradient(cmap="Blues").format({"Maximum annuelle de température d'air maximale (°C)": '{:.2f}',
         "Moyenne annuelle de température d'air maximale (°C)": '{:.2f}',"Moyenne annuelle des Précipitations totales (mm)": '{:.2f}',
         "Valeur_Indice": '{:.2f}'}))
        csv = df_selection.to_csv(index = False).encode('utf-8')
        st.download_button("Télécharger Tableau", data = csv, file_name = "ICTT_Index_info.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')

df_selection = df_selection.reset_index()
df_selection = df_selection.drop(columns=["index"])
with col1:
  with st.expander("Afficher Données"):
        # st.write(df_selection.style.background_gradient(cmap="Blues"))
        st.write(df_selection.style.background_gradient(cmap="Blues").format({"Maximum annuelle de température d'air maximale (°C)": '{:.2f}',
           "Moyenne annuelle de température d'air maximale (°C)": '{:.2f}',"Moyenne annuelle des Précipitations totales (mm)": '{:.2f}',
           "Valeur_Indice": '{:.2f}'}))
        csv = df_selection.to_csv(index = False).encode('utf-8')
        st.download_button("Télécharger Tableau", data = csv, file_name = "selectedIndex_info.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')

# Add other figures
st.subheader("Carte de la variation de l'indice thermique sélectionné")
st.plotly_chart(map_fig,use_container_width=True)

########################################  add table of indices statistics #################################

import plotly.figure_factory as ff
st.subheader(":bar_chart: Statistiques descriptives d'indice selectionné")
# with st.expander():
    #df_sample = df[0:5][["Region","State","City","Category","Sales","Profit","Quantity"]]
#del(grouped_data[grouped_data.columns[0]])
grouped_data=grouped_data.reset_index()
grouped_data=grouped_data.rename(columns={'index':'Minimum/Maximum'})
st.write(grouped_data.style.background_gradient(cmap="Blues").format({"Maximum annuelle de température d'air maximale (°C)": '{:.2f}',
         "Moyenne annuelle de température d'air maximale (°C)": '{:.2f}',"Moyenne annuelle des Précipitations totales (mm)": '{:.2f}',
         "Valeur_Indice": '{:.2f}'}))


#########################################  add table of time series statistics ######################################
import plotly.figure_factory as ff
st.subheader(":bar_chart: Statistiques descriptives de la série temporelle")
tab_stat= allstat_res
tab1 =pd.DataFrame()
tab1= tab_stat.query("Riv == @Riv & Station == @station & Variable== @varT")
st.write(tab1.style.background_gradient(cmap="Blues").format({'Maximum': '{:.2f}','Minimum': '{:.2f}','Moyenne': '{:.2f}',
         'ecartype': '{:.2f}', '25%': '{:.2f}', '50%': '{:.2f}', '75%': '{:.2f}'}))


######################################### Add final gaussian figure with table ####################################
# Display the plot
col5 = st.columns(1)
with col5[0]:
    st.subheader("Proprietés du régime thermique pour l'année sélectionnée")
    st.plotly_chart(fig_final,use_container_width=True)
with col5[0]:
    with st.expander("Afficher Données"):
        st.write(gauss_df.style.background_gradient(cmap="Blues").format({"Maximum annuelle de température d'air maximale (°C)": '{:.2f}',
         "Moyenne annuelle de température d'air maximale (°C)": '{:.2f}',"Moyenne annuelle des Précipitations totales (mm)": '{:.2f}',
         "Valeur_Indice": '{:.2f}'}))
        csv = gauss_df.to_csv(index = False).encode('utf-8')
        st.download_button("Télécharger Tableau", data = csv, file_name = "RegimeThermique_info.csv", mime = "text/csv",
                            help = 'Click here to download the data as a CSV file')




# del(tab1[tab1.columns[0]])
# with st.expander("Tableau descriptive"):
#     #df_sample = df[0:5][["Region","State","City","Category","Sales","Profit","Quantity"]]
#     st.write(tab1.style.background_gradient(cmap="Blues").format({'max': '{:.2f}','min': '{:.2f}','moyenne': '{:.2f}',
#                                 'ecartype': '{:.2f}', '25%': '{:.2f}', '50%': '{:.2f}', '75%': '{:.2f}'}))
#     csv = tab1.to_csv(index = False).encode('utf-8')
#     st.download_button("Télécharger Tableau", data = csv, file_name = "stats_descriptives.csv", mime = "text/csv",
#                             help = 'Click here to download the data as a CSV file')

#st.write("Here goes your normal Streamlit app...")
#st.button("Click me")
