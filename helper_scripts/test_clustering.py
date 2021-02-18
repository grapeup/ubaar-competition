
import pandas as pd
import os

import plotly.express as px

from feature_extraction.path_utils import project_root
from feature_extraction.coords_features import coords_clusters_dbscan, coords_clusters_kmeans

if __name__ == '__main__':

    data = pd.read_csv(os.path.join(project_root(), 'data', 'raw', 'ubaar-competition', 'train.csv'),
                       encoding="utf-8", index_col="ID")

    coords = data[["sourceLatitude", "sourceLongitude", "destinationLatitude", "destinationLongitude"]]

    # coords['cluster_src'], _ = coords_clusters_kmeans(coords, n_clusters=50)
    coords['cluster_src'], _ = coords_clusters_dbscan(coords)

    fig = px.scatter_mapbox(coords, lat="sourceLatitude", lon="sourceLongitude", zoom=3, height=900,
                            color='cluster_src', title="Clusters")
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=2, mapbox_center_lat=41,
                      margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.write_html(os.path.join(project_root(), "data", "processed", "clusters.html"))

    fig.show()


