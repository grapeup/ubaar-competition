import jdatetime
import datetime


def convert_data_to_gregorian(date):
    date = str(date)

    return jdatetime.date(1300 + int(date[:2]), int(date[2:4]), int(date[4:6])).togregorian().strftime('%Y-%m-%d')


def convert_date_to_day(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%A')


def convert_date_to_month(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%B')


def date_features(data, features):
    gregorian_data = data['date'].apply(convert_data_to_gregorian)
    features['day'] = gregorian_data.apply(convert_date_to_day)
    features['month'] = gregorian_data.apply(convert_date_to_month)

    return features
