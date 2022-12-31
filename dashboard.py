import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
import geopandas
from datetime import datetime
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

st.set_page_config(layout='wide')  # para aumentar o tamanho da tabela

@st.cache(allow_output_mutation=True)  # permitir que o arquivo seja lido mesmo após as alterações
def get_data(path):
    data = pd.read_csv(path)

    return data

@st.cache(allow_output_mutation=True)  # permitir que o arquivo seja lido mesmo após as alterações
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

def set_feature(data):
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def overview_data(data):
    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

    st.title('Data Overview')

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]
    else:
        data = data.copy()

    st.dataframe(data)

    # organizando os gráficos no dashboard - 'st.beta_columns' vai ser descontinuado - usar somente 'st.columns'
    c1, c2 = st.columns((1, 1))

    # Average metrics
    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge - o merge faz a correlação exata de acordo com o parametro escolido, o concat faz na ordem das linhas, logo pode ser que embaralhe os dados
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')  # -------> testar sem o inner, é default
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQFT LIVING', 'PRICE/m²']

    c1.header('Average values')
    c1.dataframe(df, height=600)  # st.write(df.head())

    # statistic descriptive
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    media = pd.DataFrame(num_attributes.apply(np.mean))
    mediana = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df1 = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    df1.columns = ['attributes', 'max', 'min', 'media', 'mediana', 'std']

    c2.header('Descriptive analysis')
    c2.dataframe(df1, height=800)

    return None

def portfolio_density(data, geofile):
    st.title('Region Overview')

    c1, c2 = st.columns((1, 1))
    c1.header('Portfolio Density')

    df = data.sample(10)

    # Base Map - Folium
    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup=f'Sold R${row["price"]} '
                            f'on: {row["date"]}. '
                            f'Features: {row["sqft_living"]} sqft, '
                            f'{row["bedrooms"]} bedrooms, '
                            f'{row["bathrooms"]} bathrooms, '
                            f'year built: {row["yr_built"]} ').add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # region price map
    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

    region_price_map.choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)

    return None

def commercial_distribution(data):
    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')  # strftime é de string format time

    # ====================================================================
    # ---> Average Price per Year
    st.header('Average price by year built')

    # ---> Average Price per Year -> Filter
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_yr_built = st.sidebar.slider('Year built', min_year_built, max_year_built, min_year_built)

    # ---> Average Price per Year -> Data_select
    df = data.loc[data['yr_built'] < f_yr_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    # ---> Average Price per Year -> Plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ====================================================================
    # ---> Average Price per Day
    st.header('Average Price per Day')

    # ---> Average Price per Day -> Filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    st.sidebar.subheader('Select Max Date')
    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    # ---> Average Price per Day -> Data select
    data['date'] = pd.to_datetime(data['date'])
    df = data.loc[data['date'] < f_date]
    df = df[['date', 'price']].groupby('date').mean().reset_index()

    # ---> Average Price per Day -> Plot
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # ====================================================================
    # ---> Histograma
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # ---> Histograma -> Filter
    price_min = int(data['price'].min())
    price_max = int(data['price'].max())
    price_avg = int(data['price'].mean())
    df_price = st.sidebar.slider('price', price_min, price_max, price_avg)

    # ---> Histograma -> Data select
    df = data.loc[data['price'] < df_price]

    # ---> Histograma -> Plot
    fig = px.histogram(df, x='price', nbins=50)  # nbins é pra selecionar quantas barras deve aparecer no gráfico
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution(data):
    st.sidebar.header('Attributes Options')
    st.header('House Attributes')
    c1, c2 = st.columns(2)

    # -----> Bedrooms
    c1.subheader('Houses per bedrooms')
    # Bedrooms - Filter
    df_bedrooms = st.sidebar.selectbox('max number of bedrooms', sorted(set(data['bedrooms'].unique())))

    # Bedrooms - Data selection
    df = data.loc[data['bedrooms'] < df_bedrooms]

    # Bedrooms - Plot
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # -----> Bathrooms
    c2.subheader('Houses per bathrooms')
    # Bahtrooms - Filter
    df_bathrooms = st.sidebar.selectbox('max number of bathrooms', sorted(set(data['bathrooms'].unique())))

    # Bahtrooms - Data select
    df = data.loc[data['bathrooms'] < df_bathrooms]

    # Bahtrooms - Plot
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # ===============================
    c1, c2 = st.columns(2)

    # -----> Floors
    c1.subheader('Houses per floors')
    # -----> Floors - Filter
    df_floors = st.sidebar.selectbox('max number of floors', sorted(set(data['floors'].unique())))

    # -----> Floors - Data selection
    df = data.loc[data['floors'] < df_floors]

    # -----> Floors - Plot
    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # -----> Waterview
    c2.subheader('Houses per Waterview')
    # -----> Waterview - Filter
    df_waterview = st.sidebar.checkbox('Only houses with waterview')

    # -----> Waterview - Data selection
    if df_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    # -----> Waterview - Plot
    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None

if __name__ == '__main__':
    #ETL
    #Extraction
    path = 'data/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(path)
    geofile = get_geofile(url)

    #Transformation
    data = set_feature(data)

    overview_data(data)
    portfolio_density(data, geofile)
    commercial_distribution(data)
    attributes_distribution(data)