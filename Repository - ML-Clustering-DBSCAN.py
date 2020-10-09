import pandas as pd
import numpy as np
import folium
import sklearn.utils
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

################################################################################################################################
################################## DBSCAN = CLUSTERING BASED ON LAT, LONG AND PRECIPITATION ####################################
df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/weather-stations20140101-20141231.csv')
df = df[pd.notnull(df['Tm'])]
df.reset_index(drop=True)
sklearn.utils.check_random_state(1000)
x = df[['Lat','Long','P']]
x = np.nan_to_num(x)
x = StandardScaler().fit_transform(x)
#########LETS FIT DBSCAN
kms_per_radian = 6371
epsilon = 900 / kms_per_radian
dbs = DBSCAN(eps=epsilon,min_samples=10)
dbs.fit(x)
labels_serie = dbs.labels_
#########CREATE ARRAY OF BOOLEANS WHERE TRUE IF DATAPOINTS ARE IN BORDER POINTS, FALSE IF OUTLIERS
border_points = np.zeros_like(labels_serie, dtype=bool) #create array from list specifying boolean dtype
border_points[dbs.core_sample_indices_] = True #Array of TRUES, turning FALSE if Outlier points thanks to [dbs.core_sample_indices_]
#########FIND NO OF CLUSTERS MINUS OUTLIERS IF ANY
no_clusters = set(labels_serie) #turn list into set
no_clusters = len(no_clusters)-(1 if -1 in no_clusters else 0) #no of clusters muinus outlier if any outlier found in list_clusters
#########UNIQUE CLUSTERS
unique_clusters = set(labels_serie)
df['Cluster'] = labels_serie
####################PLOT
weather_stations_map = folium.Map(location=[df['Lat'][0],df['Long'][0]],zoom_start=5)
df = df[['Lat','Long','Cluster','P']].reset_index(drop=False)
labels_list = list(dict.fromkeys(labels_serie))
colors = ['green','orange','pink', 'darkred', 'cadetblue','lightblue', 'blue','darkpurple',
          'darkblue', 'darkgreen', 'darkred', 'lightgray', 'lightgreen','lightred', 'gray',
          'purple', 'red', 'white']

for cluster,color in zip(labels_list, colors):
    new_df = df[df['Cluster'] == cluster].reset_index(drop=True)
    if cluster == -1:
        color = 'black'
        radius = 2
    else:
        for lat,long,label,radius in zip(new_df['Lat'],new_df['Long'],new_df['Cluster'],new_df['P']):
            #radius = 3
            folium.CircleMarker([lat,long],
                                fill = True,
                                color = color,
                                fill_opacity=0.6,
                                popup=label,
                                radius = radius/3).add_to(weather_stations_map)
weather_stations_map.save('Weather Stations.html')