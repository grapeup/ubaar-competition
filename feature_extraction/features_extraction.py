import os
from feature_extraction.date_utils import date_features
from feature_extraction.coords_features import coord_features
from feature_extraction.other_features import raw_features, categorical_features
from feature_extraction.path_utils import project_root

from metaflow import FlowSpec, step
import pandas as pd


class UbaarFeaturesExtraction(FlowSpec):
    @step
    def start(self):

        self.data = pd.read_csv(os.path.join(project_root(), 'data', 'raw', 'ubaar-competition', 'train.csv'),
                                encoding="utf-8", index_col="ID")
        self.features = pd.DataFrame()

        self.next(self.get_day_feature)

    @step
    def get_day_feature(self):
        self.features = date_features(self.data, self.features)

        self.next(self.get_lat_long_features)

    @step
    def get_lat_long_features(self):
        self.features = coord_features(self.data, self.features)

        self.next(self.add_categorical_features)

    @step
    def add_categorical_features(self):
        self.features = categorical_features(self.data, self.features)

        self.next(self.add_raw_features)

    @step
    def add_raw_features(self):

        self.features = raw_features(self.data, self.features)

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
