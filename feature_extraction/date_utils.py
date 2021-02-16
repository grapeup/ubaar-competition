import jdatetime
import datetime


def convert_data_to_gregorian(date):
    date = str(date)

    return jdatetime.date(1300 + int(date[:2]), int(date[2:4]), int(date[4:6])).togregorian().strftime('%Y-%m-%d')


def convert_date_to_day(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%A')
