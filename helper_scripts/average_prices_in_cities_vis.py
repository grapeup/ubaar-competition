
import pandas as pd
import os

import plotly.express as px
import reverse_geocoder as rg

from feature_extraction.path_utils import project_root

if __name__ == '__main__':
    src_dest = 'source'

    data = pd.read_csv(os.path.join(project_root(), 'data', 'raw', 'ubaar-competition', 'train.csv'),
                       encoding="utf-8", index_col="ID")

    coords = data[[f'{src_dest}Latitude', f'{src_dest}Longitude']]
    localisations = rg.search([tuple(row) for row in coords.values])

    data[f'{src_dest}_city'] = [l['name'] for l in localisations]
    data[f'{src_dest}_province'] = [l['admin1'] for l in localisations]

    # data['price_per_km'] = data['price'] / data['distanceKM']
    city_ave_prices = dict(data.groupby(f'{src_dest}_city')['price'].mean())

    data['ave_price'] = data.apply(lambda x: city_ave_prices[x[f'{src_dest}_city']], axis=1)

    fig = px.scatter_mapbox(data, lat=f"{src_dest}Latitude", lon=f"{src_dest}Longitude", zoom=3, height=900,
                            color='ave_price', title="Average prices")
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_zoom=2, mapbox_center_lat=41,
                      margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.write_html(os.path.join(project_root(), "data", "processed", "ave_prices.html"))

    fig.show()
