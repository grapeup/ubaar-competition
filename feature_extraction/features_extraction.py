import os
from feature_extraction.date_utils import convert_data_to_gregorian, convert_date_to_day, convert_date_to_month
from feature_extraction.coords_features import dummy_manhattan_distance, bearing_array, center_lat_feat, \
    center_lng_feat, coords_clusters_dbscan, coords_clusters_kmeans

from sklearn.preprocessing import LabelEncoder

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
        self.features['month'] = gregorian_data.apply(convert_date_to_month)

        self.next(self.get_lat_long_features)

    @step
    def get_lat_long_features(self):
        coords = self.data[['sourceLatitude', 'sourceLongitude', 'destinationLatitude', 'destinationLongitude']]
        self.features['dmd'] = coords.apply(dummy_manhattan_distance, axis=1)
        self.features['bearing_array'] = coords.apply(bearing_array, axis=1)
        self.features['center_lat'] = coords.apply(center_lat_feat, axis=1)
        self.features['center_lng'] = coords.apply(center_lng_feat, axis=1)

        self.features['cluster_src_db'], self.features['cluster_dest_db'] = coords_clusters_dbscan(coords)
        self.features['cluster_src_km'], self.features['cluster_dest_km'] = coords_clusters_kmeans(coords,
                                                                                                   n_clusters=120)

        self.next(self.add_categorical_features)

    @step
    def add_categorical_features(self):

        self.features['vehicleType'] = self.data['vehicleType']
        self.features['vehicleOption'] = self.data['vehicleOption']

        self.features['vehicleTypeOption'] = [a + '_' + b for a, b in zip(self.data['vehicleType'].values,
                                                                          self.data['vehicleOption'].values)]

        cat_columns_clusters = ['cluster_dest_db', 'cluster_src_db', 'cluster_src_km', 'cluster_dest_km']
        cat_columns_date = ['day', 'month']
        cat_columns = ['vehicleType', 'vehicleOption', 'vehicleTypeOption']
        # cat_columns += cat_columns_clusters
        # cat_columns += cat_columns_date

        self.features = pd.get_dummies(self.features, columns=cat_columns, drop_first=True)

        self.features['day'] = LabelEncoder().fit_transform(self.features['day'])
        self.features['month'] = LabelEncoder().fit_transform(self.features['month'])

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

        self.features['src_dest'] = (self.data['SourceState'] == self.data['destinationState'])
        self.features['ave_speed'] = self.data['distanceKM'] / self.data['taxiDurationMin']

        import numpy as np
        self.features['weight_dur'] = np.log((self.data['taxiDurationMin']+30*self.data['weight']))
        self.features['weight_dist_dur'] = np.log(1. + (10. + self.data['weight']) * (100. + self.data['distanceKM']) *
                                                  (1000. + self.data['taxiDurationMin']))

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
