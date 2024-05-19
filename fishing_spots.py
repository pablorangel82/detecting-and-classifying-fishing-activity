import folium
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Point
from geopy.distance import geodesic
import numpy as np
import pandas as pd
from ast import literal_eval
import os

def find_centroid(centroids, label):
    results = []
    for tuple in centroids:
        if tuple[1] == label:
            results.append(tuple[0])
    return results

def create(technique):
    print("Creating " + technique)
    dataset_csv_path = f'data/{technique}.csv'
    csv_path = 'data/fishing_spots.csv'
    map_save_path = f'data/{technique}_map.html'

    df = pd.read_csv(dataset_csv_path, usecols=['lat', 'lon', 'is_fishing'])
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    df_coords = df[df.is_fishing == 1]

    coords = df_coords[['latitude', 'longitude']].values.tolist()

    X = np.array(coords)

    dbscan = DBSCAN(eps=2.0, min_samples=5)
    labels = dbscan.fit_predict(coords)

    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    clusters_list = [[] for _ in range(num_clusters)]

    centroids = []
    for label in unique_labels:
        if label != -1:
            cluster_geography_points = X[labels == label]
            centroid = np.mean(cluster_geography_points, axis=0)
            centroids.append((centroid, label))

    for i, (point, label) in enumerate(zip(coords, labels)):
        if label != -1:
            clusters_list[label].append(point)

    data = []

    for i, lista in enumerate(clusters_list):
        cluster_points = MultiPoint(lista)
        convex_hull = cluster_points.convex_hull
        center_point = find_centroid(centroids, i)
        vertices = list(convex_hull.exterior.coords)

        max_distance_point = max(list(convex_hull.exterior.coords), key=lambda point: geodesic((center_point[0][0], center_point[0][1]), point).meters)
        max_distance = geodesic((center_point[0][0], center_point[0][1]), max_distance_point).miles

        data.append({
            'polygon': vertices,
            'density': len(lista),
            'type': technique,
            'cluster_label': i,
            'centroid': [(center_point[0][0], center_point[0][1])],
            'max_distance': max_distance
        })

    if os.path.isfile(csv_path):
        data_existent = pd.read_csv(csv_path)
        data_concatenated = pd.concat([data_existent, pd.DataFrame(data)], ignore_index=True)
    else:
        data_concatenated = pd.DataFrame(data)

    data_concatenated.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)
    df = df[df.type == technique]

    map = folium.Map(location=[0, 0], zoom_start=4)

    for index, row in df.iterrows():
        polygon_coords = literal_eval(row.polygon)
        centroid_coords = literal_eval(row.centroid)

        popup_content = f"""
        <p>Type: {row.type}</p>
        <p>Density: {pd.to_numeric(row.density)}</p>
        <p>Cluster label: {pd.to_numeric(row.density)}</p>
        <p>Distance(MN): {pd.to_numeric(row.max_distance)}</p>
        """

        folium.PolyLine(
            locations=polygon_coords,
            color='blue',
            popup=folium.Popup(popup_content, max_width=300)
        ).add_to(map)

        folium.Circle(
            location=centroid_coords[0],
            radius= 5000,
            fill='True',
            fill_color='red',
            color='red',
            popup='Centroid'
        ).add_to(map)

    map.save(map_save_path)

def extraction():
    print ('Extracting fishing spots...')
    create("drifting_longlines")
    create("purse_seines")
    create("fixed_gear")
    create("trawlers")