import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

from feature_extraction.path_utils import project_root

if __name__ == '__main__':

    data = pd.read_csv(os.path.join(project_root(), 'data', 'raw', 'ubaar-competition', 'train.csv'),
                       encoding="utf-8", index_col="ID")

    fig = go.Figure()
    for row in data[["sourceLatitude", "sourceLongitude", "destinationLatitude", "destinationLongitude"]].values[:200]:
        fig.add_trace(go.Scattermapbox(
            mode="markers+lines",
            lon=[row[1], row[3]],
            lat=[row[0], row[2]],
            marker={'size': 10}))

    fig.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        mapbox={
            'style': "stamen-terrain",
            'center': {'lon': -20, 'lat': -20},
            'zoom': 1},
        title="Trips")
    fig.write_html(f"data/trips.html")
    fig.show()

    fig = px.scatter_mapbox(data, lat="sourceLatitude", lon="sourceLongitude", zoom=3, height=900, title="Sources of trips")
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=2, mapbox_center_lat=41,
                      margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.write_html(os.path.join(project_root(), "data", "processed", "sources.html"))

    fig.show()



