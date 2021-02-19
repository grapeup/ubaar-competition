import os
import pandas as pd
from feature_extraction.date_utils import date_features
from feature_extraction.coords_features import coord_features
from feature_extraction.other_features import raw_features, categorical_features
from feature_extraction.path_utils import project_root
import xgboost as xgb
import joblib

raw_data = pd.read_csv(os.path.join(project_root(), 'data', 'raw', 'ubaar-competition', 'train.csv'),
                       encoding="utf-8", index_col="ID")
all_features_cols = pd.read_csv(os.path.join(project_root(), 'data', 'processed', 'ubaar_features.csv'),
                                encoding="utf-8", index_col="ID").columns

model = joblib.load(os.path.join(project_root(), 'data', 'processed', 'model.bin'))
num_cols = ['sourceLatitude', 'sourceLongitude', 'destinationLatitude', 'destinationLongitude',
            'distanceKM', 'taxiDurationMin', 'weight', 'price']
num_cols_dict = {col: float for col in num_cols}


def _add_missing_cat_columns(features, all_features_cols):
    missing_columns = [c for c in all_features_cols if c not in features.columns]
    features[missing_columns] = 0
    return features


def infer(row: str):
    data = pd.DataFrame([row], columns=raw_data.columns)

    data = data.astype(num_cols_dict)
    features = pd.DataFrame()

    features = date_features(data, features)
    features = coord_features(data, features, do_clusters=False)
    features = categorical_features(data, features)

    features = _add_missing_cat_columns(features, all_features_cols)  # Todo
    features = raw_features(data, features)

    features_columns = [c for c in features.columns if not c.startswith('cluster_')]
    features_columns.remove('price')
    features = features[features_columns]

    dtest = xgb.DMatrix(features)
    price = model.predict(dtest)

    return float(price[0])
