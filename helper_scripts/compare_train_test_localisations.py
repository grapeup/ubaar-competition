from feature_extraction.path_utils import project_root
import plotly.express as px

import pandas as pd
import os


if __name__ == '__main__':

    train = pd.read_csv(os.path.join(project_root(), 'data', 'raw', 'ubaar-competition', 'train.csv'),
                        encoding="utf-8", index_col="ID")

    test = pd.read_csv(os.path.join(project_root(), 'data', 'raw', 'ubaar-competition', 'test.csv'),
                       encoding="utf-8", index_col="ID")
    train['test'] = False
    test['test'] = True

    train = train.append(test)
    coords = train[["sourceLatitude", "sourceLongitude", "destinationLatitude", "destinationLongitude", "test"]]

    fig = px.scatter_mapbox(coords, lat="sourceLatitude", lon="sourceLongitude",
                            color='test', title="Train/test data coordinates")
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=2, mapbox_center_lat=41,
                      margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.write_html(os.path.join(project_root(), "data", "processed", "train_test.html"))
    fig.show()



