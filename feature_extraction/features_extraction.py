import os
from feature_extraction.date_utils import convert_data_to_gregorian, convert_date_to_day
from feature_extraction.coords_features import dummy_manhattan_distance, bearing_array, center_lat_feat, center_lng_feat, coords_clusters

from metaflow import FlowSpec, step
import pandas as pd

from feature_extraction.path_utils import project_root


class UbaarFeaturesExtraction(FlowSpec):
    @step
    def start(self):

        self.data = pd.read_csv(os.path.join(project_root(), 'data', 'raw', 'ubaar-competition', 'train.csv'),
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

        self.features['cluster_src'], self.features['cluster_dest'] = coords_clusters(coords, n_clusters=20)

        self.next(self.add_categorical_features)

    @step
    def add_categorical_features(self):

        self.features['vehicleType'] = self.data['vehicleType']
        self.features['vehicleOption'] = self.data['vehicleOption']

        cat_columns = ['vehicleType', 'vehicleOption', 'cluster_src', 'cluster_dest', 'day']
        self.features = pd.get_dummies(self.features, columns=cat_columns, drop_first=True)

        self.next(self.add_raw_features)

    @step
    def add_raw_features(self):

        self.features['weight'] = self.data['weight']
        self.features['distanceKM'] = self.data['distanceKM']
        self.features['taxiDurationMin'] = self.data['taxiDurationMin']

        self.features['sourceLatitude'] = self.data['sourceLatitude']
        self.features['sourceLongitude'] = self.data['sourceLongitude']
        self.features['destinationLatitude'] = self.data['destinationLatitude']
        self.features['destinationLongitude'] = self.data['destinationLongitude']

        self.features['price'] = self.data['price']

        self.next(self.remove_missing_data)

    @step
    def remove_missing_data(self):

        self.features.dropna(inplace=True)
        self.next(self.report)

    @step
    def report(self):
        print(f"Shape: {self.features.shape}")
        self.next(self.save)

    @step
    def save(self):
        self.features.to_csv(os.path.join(project_root(), 'data', 'processed', 'ubaar_features.csv'))
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    UbaarFeaturesExtraction()