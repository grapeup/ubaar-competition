import os
from date_utils import convert_data_to_gregorian, convert_date_to_day
from coords_features import dummy_manhattan_distance, bearing_array, center_lat_feat, center_lng_feat, coords_clusters

from metaflow import FlowSpec, step
import pandas as pd


class UbaarFeaturesExtraction(FlowSpec):
    @step
    def start(self):

        self.data = pd.read_csv(os.path.join('data', 'raw', 'ubaar-competition', 'train.csv'),
                                encoding="utf-8", index_col="ID")
        self.features = pd.DataFrame()

        self.next(self.get_day_feature)

    @step
    def get_day_feature(self):

        gregorian_data = self.data['date'].apply(convert_data_to_gregorian)
        self.features['day'] = gregorian_data.apply(convert_date_to_day)

        self.next(self.get_lat_long_features)

    @step
    def get_lat_long_features(self):
        coords = self.data[['sourceLatitude', 'sourceLongitude', 'destinationLatitude', 'destinationLongitude']]
        self.features['dmd'] = coords.apply(dummy_manhattan_distance, axis=1)
        self.features['bearing_array'] = coords.apply(bearing_array, axis=1)
        self.features['center_lat'] = coords.apply(center_lat_feat, axis=1)
        self.features['center_lng'] = coords.apply(center_lng_feat, axis=1)

        self.features['cluster_src'], self.features['cluster_dest'] = coords_clusters(coords, n_clusters=50)

        self.next(self.add_categorical_features)

    @step
    def add_categorical_features(self):

        self.features['vehicleType'] = self.data['vehicleType']
        self.features['vehicleOption'] = self.data['vehicleOption']

        cat_columns = ['vehicleType', 'vehicleOption', 'cluster_src', 'cluster_dest']
        self.features = pd.get_dummies(self.features, columns=cat_columns, drop_first=True)
        self.features = self.features.drop(columns=cat_columns)

        self.next(self.add_raw_features)

    @step
    def add_raw_features(self):

        self.features['weight'] = self.data['weight']
        self.features['distanceKM'] = self.data['distanceKM']
        self.features['taxiDurationMin'] = self.data['taxiDurationMin']

        self.features['vehicleOption'] = self.data['vehicleOption']
        self.features['price'] = self.data['price']

        self.next(self.report)

    @step
    def report(self):
        print(f"Shape: {self.features.shape}")
        self.next(self.save)

    @step
    def save(self):
        self.features.to_csv(os.path.join('data', 'processed', 'ubaar_features.csv'))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    UbaarFeaturesExtraction()
