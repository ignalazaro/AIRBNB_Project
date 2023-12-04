# Refactored code

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import altair as alt
import seaborn as sns
sns.set_style("whitegrid")
import base64
import datetime
from matplotlib import rcParams
from  matplotlib.ticker import PercentFormatter
import branca.colormap as cm
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap
from streamlit_folium import folium_static as st_folium
from folium.plugins import Fullscreen
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
nltk.download('stopwords')
#text mining
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import numpy as np


# Load data
df = pd.read_csv('datos/listings.csv')
listings_1 = pd.read_csv("datos/listings_1.csv")

# Sidebar
# st.image("", caption="Descripción de la imagen")
st.sidebar.header("Welcome!")
st.sidebar.markdown(" ")
st.sidebar.markdown("*A newbie Data Analyst From Barcelona.  \nI specialize in data analysis, data visualization, machine learning, and possess in-depth knowledge of cloud technologies and SQL. \nMy passion lies in transforming data into actionable insights to drive strategic decision-making and organizational growth.*")
st.sidebar.markdown("**Author**: Ignacio Lázaro")
st.sidebar.markdown("**Mail**: ignaciolazaro80@gmail.com")
st.sidebar.markdown("- [Linkedin](https://www.linkedin.com/in/ignaciolázaro/)")
st.sidebar.markdown("**Version:** 1.0.0")

# Title
st.title("Airbnb Roma listings Data Analysis")
st.markdown('-----------------------------------------------------')
st.markdown("*Through Airbnb Roma data we will conduct an exploratory analysis and offer insights into that data. For this we will use the data behind the website **Inside Airbnb** come from publicly available information on the Airbnb website available [here](http://insideairbnb.com/), containing advertisements for accommodation in Roma until 2020*")

# Summary
st.header("Summary")
st.markdown("Airbnb is a platform that provides and guides the opportunity to link two groups - the hosts and the guests. Anyone with an open room or free space can provide services on Airbnb to the global community. It is a good way to provide extra income with minimal effort. It is an easy way to advertise space, because the platform has traffic and a global user base to support it. Airbnb offers hosts an easy way to monetize space that would be wasted.")
st.markdown("On the other hand, we have guests with very specific needs - some may be looking for affordable accommodation close to the city's attractions, while others are a luxury apartment by the sea. They can be groups, families or local and foreign individuals. After each visit, guests have the opportunity to rate and stay with their comments. We will try to find out what contributes to the listing's popularity and predict whether the listing has the potential to become one of the 100 most reviewed accommodations based on its attributes.")
st.markdown('-----------------------------------------------------')

# Dataframe
st.header("Airbnb Roma Listings: Data Analysis")
st.markdown("Following is presented the first 10 records of Airbnb data. These records are grouped along 16 columns with a variety of informations as host name, price, room type, minimum of nights,reviews and reviews per month.")
st.markdown("We will start with familiarizing ourselves with the columns in the dataset, to understand what each feature represents. This is important, because a poor understanding of the features could cause us to make mistakes in the data analysis and the modeling process. We will also try to reduce number of columns that either contained elsewhere or do not carry information that can be used to answer our questions.")
st.dataframe(df.head(10))
st.markdown("Another point about our data is that it allows sorting the dataframe upon clicking any column header, it a more flexible way to order data to visualize it.")

# Geographical distribution
st.header("Listing Locations")
st.markdown("We could also filter by listing **price**, **minimum nights** on a listing or minimum of **reviews** received. ")
values = st.slider("Price Range (€)", float(df.price.min()), float(4000), (100., 1500.))
number_reviews = st.slider('Minimum Reviews', 0, 700, (0))
reviews = st.slider('Review score rating', 3.5, 5.0, (0.2))
neighbourhood = st.selectbox('Neighbourhood', options=[
    'VIII Appia Antica', 'I Centro Storico', 'II Parioli/Nomentano',
       'V Prenestino/Centocelle', 'XIII Aurelia',
       'VII San Giovanni/Cinecittà', 'XII Monte Verde', 'IX Eur',
       'IV Tiburtina', 'XV Cassia/Flaminia', 'XIV Monte Mario',
       'VI Roma delle Torri', 'III Monte Sacro', 'X Ostia/Acilia',
       'XI Arvalia/Portuense'])
room_type = st.selectbox('Room Type', options=['Private room', 'Entire home/apt', 'Shared room'])
st.map(df.query(f"price.between{values} and number_of_reviews_x>={number_reviews} and review_scores_rating>={reviews} and neighbourhood=='{neighbourhood}' and room_type=='{room_type}' ")[["latitude", "longitude"]].dropna(how="any"), zoom=10)
st.markdown("In a general way the map shows that locations in the city centre are more expensive, while the outskirts are cheaper (a pattern that probably does not only exists in Roma). In addition, the city centre seems to have its own pattern.")
st.markdown('Unsurprisingly, Centro Storico has the highest concentration of expensive Airbnbs.')


# District
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Neighbourhoods")
st.markdown("Roma is classified in XV Districts,")
st.markdown("Again unsurprisingly it is possible see that the average price in the Manhattan district can be much higher than other districts. Manhattan has an average price of twice the Bronx ")
df_neigh = df.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).reset_index()
fig = sns.barplot(x='neighbourhood', y='price', data=df_neigh.head(),palette="Blues_d")
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
fig.set_xlabel("",fontsize=10)
fig.set_ylabel("Price (€)",fontsize=10)
st.pyplot()

# Scatter Mapbox
fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', size='price', zoom=11, mapbox_style='carto-positron',
                   title='AirBnb Superhost Apartment Distribution in Roma', template= "plotly_dark", size_max=20, 
                   color='review_scores_rating',range_color=[4.7, 5])
st.plotly_chart(fig)

# Maps
adam = gpd.read_file("datos/neighbourhoods.geojson")
feq2 = df[df['accommodates']==2]
feq2 = feq2.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)
feq2 = pd.DataFrame(feq2)
feq2.reset_index()
feq3 = df[df['accommodates']==3]
feq3 = feq3.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)
feq3 = pd.DataFrame(feq3)
feq3.reset_index()
feq4 = df[df['accommodates']==4]
feq4 = feq4.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)
feq4 = pd.DataFrame(feq4)
feq4.reset_index()
adam = pd.merge(adam, feq2, on='neighbourhood', how='left')
adam.rename(columns={'price': 'average_price'}, inplace=True)
adam['average_price'] = adam['average_price'].round(decimals=2)
map_dict = adam.set_index('neighbourhood')['average_price'].to_dict()
color_scale = cm.linear.YlOrRd_07.scale(vmin = min(map_dict.values()), vmax = max(map_dict.values()))
def get_color(feature):
    value = map_dict.get(feature['properties']['neighbourhood'])
    if value is not None:
        return color_scale(value)
    else:
        return 'grey'  # Un color predeterminado si el barrio no está en el diccionario
df_monumento = pd.DataFrame({'Monumento':['Coliseo','Vaticano','Fontana di Trevi','Panteon', 'Foro Romano'], 'latitude':[41.8902692,41.9038162,41.9010332,41.8986108,41.8924623],
              'longitude':[12.4918651,12.4469113,12.478402,12.474298,12.4827501]
              })
########## FOLIUM MAP CONFIG ######
mapsuperhost = folium.plugins.DualMap(location=[41.900735, 12.483280], zoom_start=9,layout="vertical")
latssuperhost = df['latitude'].tolist()
lonssuperhost = df['longitude'].tolist()
locations = list(zip(latssuperhost, lonssuperhost),)
FastMarkerCluster(data=locations).add_to(mapsuperhost.m1)
latsmon = df_monumento['latitude'].tolist()
longmon = df_monumento['longitude'].tolist()
locsmon = list(zip(latsmon, longmon),)
nombres_monumentos = df_monumento['Monumento'].tolist()
for loc,nombre in zip(locsmon, nombres_monumentos):
    folium.Marker(
        location=loc,
        icon=folium.Icon(icon='info-sign'),  # Icono personalizado, ajusta según tus preferencias
        tooltip= nombre # Texto que aparecerá al pasar el ratón
    ).add_to(mapsuperhost.m2)
map3 = folium.Map(location=[41.902652, 12.484885], zoom_start=11)
folium.GeoJson(data=adam,
               name='Roma',
               tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_price'],
                                                      labels=True,
                                                      sticky=True),
               style_function= lambda feature: {
                   'fillColor': get_color(feature),
                   'color': 'black',
                   'weight': 1,
                   'dashArray': '5, 5',
                   'fillOpacity':0.55
                   },
               highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.9}).add_to(mapsuperhost.m2)
st_map = st_folium(mapsuperhost)



# Refactored code for QUANTITY OF ROOM TYPES BY neighbourhood
st.header("QUANTITY OF ROOM TYPES BY neighbourhood")
st.markdown("In this section we will analyze the quantity of room types by neighbourhood. We will use a bar plot to visualize the percentage of each room type in each neighborhood. This will allow us to identify which neighborhoods have the highest percentage of each room type.")
st.markdown('As we can see in the plot we can conclude that **Entire home/apt** is the most common room type in all neighborhoods, followed by **Private room**. **Shared room** is the least common room type in all neighborhoods. This is not surprising, as most people prefer to rent an entire home or apartment, rather than a private or shared room.')
# Define the list of neighborhoods to include
barrios = ['I Centro Storico', 'II Parioli/Nomentano', 'VII San Giovanni/Cinecittà', 'XIII Aurelia', 'XII Monte Verde', 'V Prenestino/Centocelle', 'VII San Giovanni/Cinecittà']# Group the dataframe by neighborhood and room type, and count the number of listings in each group
room_types_df = df.groupby(['neighbourhood', 'room_type']).size().reset_index(name='Quantity')
# Filter the dataframe to only include the specified neighborhoods
room_types_df_select = room_types_df[room_types_df['neighbourhood'].isin(barrios)]
# Calculate the percentage of each room type in each neighborhood
room_types_df_select['Percentage'] = room_types_df_select.groupby(['neighbourhood'])['Quantity'].apply(lambda x:100 * x / float(x.sum()))
# Define a custom color palette for the plot
custom_colormap = plt.cm.get_cmap("flare")
n_colors = 6
custom_palette = [custom_colormap(i / n_colors) for i in range(n_colors)]
# Create the bar plot using seaborn
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = sns.catplot(y='Percentage', x='neighbourhood', hue="room_type", data=room_types_df_select, height=6, kind="bar", palette=custom_palette, ci=95)
fig.set(ylim=(0, 100))
plt.xticks(rotation=45)
# Format the y-axis labels as percentages
for ax in fig.axes.flat:
    ax.yaxis.set_major_formatter(PercentFormatter(100))
# Move the legend to the center of the plot
sns.move_legend(fig, loc="center", frameon=False, title=None, bbox_to_anchor=(.5, 1), ncol=4, fontsize=8)
# Format the y-axis labels as percentages again (in case they were overwritten by the legend)
for ax in fig.axes.flat:
    ax.yaxis.set_major_formatter(PercentFormatter(100))
# Adjust the layout of the plot
plt.tight_layout()
# Display the plot
st.pyplot(fig)


# Refactored code for PRICE AVERAGE BY ACOMMODATION

# Group the dataframe by room type and calculate the average price for each group
avg_price_room = df.groupby("room_type").price.mean().reset_index()\
    .round(2).sort_values("price", ascending=False)\
    .assign(avg_price=lambda x: x.pop("price").apply(lambda y: "%.2f" % y))

# Rename the columns of the dataframe
avg_price_room = avg_price_room.rename(columns={'room_type':'Room Type', 'avg_price': 'Average Price ($)', })

# Display the dataframe as a table
st.table(avg_price_room)

# Display some additional information about the data
st.markdown("Despite together **Hotel Room** listings represent just over 10%, they are responsible for the highest price average, followed by **Entire home/apt**. Thus there are a small number of **Hotel Room** listings due its expensive prices.")
st.markdown("Well, actually does not happens a correlation between reviews and price apparently, the cheaper the more opinions he has. Another point is,  Queens has more reviews than others, which reinforces our theory about being the most cost-effective district.")


# Refactored code for DEMAND AND PRICE ANALYSIS

import plotly.express as px

# Allow the user to select a room type to analyze
accommodation = st.radio("Room Type", df.room_type.unique())

# If the user selects "All Accommodations", use the entire dataframe
if accommodation == "All Accommodations":
    demand_df = df[df.reviews_per_month_x.notnull()]
    demand_df.loc[:,'reviews_per_month_x'] = pd.to_datetime(demand_df.loc[:,'reviews_per_month_x'])
# Otherwise, filter the dataframe to only include the selected room type
else:
    demand_df = df.query(f"""room_type==@accommodation""")

# Create a scatter plot of price vs. number of reviews, colored by neighborhood
fig2 = px.scatter(demand_df, y="price", x="reviews_per_month_x", color="neighbourhood")
fig2.update_xaxes(title="Nª Reviews")
fig2.update_yaxes(title="Price ($)", range=[0, 3000])

# Display the plot
st.plotly_chart(fig2)

# Refactored code for LISTINGS BY HOST

# Load the listings data from a CSV file
listings_1 = pd.read_csv("datos/listings_1.csv")

# Group the listings by host ID and host name, and count the number of listings for each host
freq = listings_1.groupby(['host_id', 'host_name']).size().reset_index(name='num_host_listings')

# Sort the hosts by the number of listings they have in descending order
freq = freq.sort_values(by=['num_host_listings'], ascending=False)

# Filter the hosts to only include those with 20 or more listings
freq = freq[freq['num_host_listings'] >= 20]

# Select the columns to display in the table and limit the number of rows to 30
freq =freq[['host_name', 'num_host_listings']].head(30).reset_index(drop=True)

# Display the table
st.dataframe(freq)


# Refactored code for WORDCLOUD

# Load the image file for the wordcloud
image='wordcloud.png'

# Display the image
st.image(image, use_column_width=True)
