import numpy as np
from sklearn.cluster import DBSCAN, KMeans

AVG_EARTH_RADIUS = 6371


def _haversine_array(lat1, lng1, lat2, lng2):
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h


def dummy_manhattan_distance(row):
    lat1, lng1, lat2, lng2 = map(np.radians, row)
    a = _haversine_array(lat1, lng1, lat1, lng2)
    b = _haversine_array(lat1, lng1, lat2, lng2)
    return a + b


def bearing_array(row):
    lat1, lng1, lat2, lng2 = row
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def center_lat_feat(row):
    lat1, lng1, lat2, lng2 = row
    center_lat = (lat1 + lat2) / 2

    return center_lat


def center_lng_feat(row):
    lat1, lng1, lat2, lng2 = row
    center_lng = (lng1 + lng2) / 2

    return center_lng


def coords_clusters(coords, n_clusters):

    coords_flat = np.vstack((coords[['sourceLatitude', 'sourceLongitude']].values,
                             coords[['destinationLatitude', 'destinationLongitude']].values))

    model = KMeans(n_clusters=n_clusters, random_state=42).fit(coords_flat)
    src_cluster = model.predict(coords[['sourceLatitude', 'sourceLongitude']])
    dest_cluster = model.predict(coords[['destinationLatitude', 'destinationLongitude']])

    import time
    start = time.time()
    clusters = DBSCAN(eps=0.01, min_samples=100, leaf_size=30).fit_predict(coords_flat)
    print(time.time() - start)
    src_cluster = clusters[:len(coords)]
    dest_cluster = clusters[len(coords):]

    return src_cluster, dest_cluster
