import pandas as pd
from sklearn.preprocessing import LabelEncoder


def categorical_features(data, features):
    features['vehicleType'] = data['vehicleType']
    features['vehicleOption'] = data['vehicleOption']

    features['vehicleTypeOption'] = [a + '_' + b for a, b in zip(data['vehicleType'].values,
                                                                 data['vehicleOption'].values)]

    cat_columns_clusters = ['cluster_dest_db', 'cluster_src_db', 'cluster_src_km', 'cluster_dest_km']
    cat_columns_date = ['day', 'month']
    cat_columns = ['vehicleType', 'vehicleOption', 'vehicleTypeOption']
    # cat_columns += cat_columns_clusters
    # cat_columns += cat_columns_date

    features = pd.get_dummies(features, columns=cat_columns, drop_first=True)

    features['day'] = LabelEncoder().fit_transform(features['day'])
    features['month'] = LabelEncoder().fit_transform(features['month'])

    return features


def raw_features(data, features):
    features['weight'] = data['weight']
    features['distanceKM'] = data['distanceKM']
    features['taxiDurationMin'] = data['taxiDurationMin']

    features['sourceLatitude'] = data['sourceLatitude']
    features['sourceLongitude'] = data['sourceLongitude']
    features['destinationLatitude'] = data['destinationLatitude']
    features['destinationLongitude'] = data['destinationLongitude']

    features['src_dest'] = (data['SourceState'] == data['destinationState'])
    features['ave_speed'] = data['distanceKM'] / data['taxiDurationMin']

    import numpy as np
    features['weight_dur'] = np.log((data['taxiDurationMin'] + 30 * data['weight']))
    features['weight_dist_dur'] = np.log(1. + (10. + data['weight']) * (100. + data['distanceKM']) *
                                         (1000. + data['taxiDurationMin']))

    features['price'] = data['price']

    return features
