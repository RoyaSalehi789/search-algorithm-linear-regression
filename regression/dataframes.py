import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


def dataframes():
    x = pd.read_csv('Flight_Price_Dataset_Q2.csv')

    label_encoder = LabelEncoder()

    x['departure_time'] = label_encoder.fit_transform(x['departure_time'])
    x['stops'] = label_encoder.fit_transform(x['stops'])
    x['arrival_time'] = label_encoder.fit_transform(x['arrival_time'])
    x['class'] = label_encoder.fit_transform(x['class'])

    y = list(x['price'])
    y = {'price': y}
    y = pd.DataFrame(y)

    x['departure_time'] = x['departure_time'].astype(float)
    x['stops'] = x['stops'].astype(float)
    x['arrival_time'] = x['arrival_time'].astype(float)
    x['class'] = x['class'].astype(float)
    x['days_left'] = x['days_left'].astype(float)
    y['price'] = y['price'].astype(float)

    x = x.drop('price', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    min_x_list = list(x.min())
    max_x_list = list(x.max())

    y_max = y.max()
    y_min = y.min()

    return x, x_train, x_test, y, y_train, y_test, min_x_list, max_x_list, y_min, y_max


def create_array():
    x, x_train, x_test, y, y_train, y_test, min_x_list, max_x_list, y_min, y_max = dataframes()

    departure_time_train = np.array([])
    stops_train = np.array([])
    arrival_time_train = np.array([])
    class_train = np.array([])
    duration_train = np.array([])
    days_left_train = np.array([])
    price_train = np.array([])

    departure_time_test = np.array([])
    stops_test = np.array([])
    arrival_time_test = np.array([])
    class_test = np.array([])
    duration_test = np.array([])
    days_left_test = np.array([])
    price_test = np.array([])

    departure_time_train = x_train['departure_time'].to_numpy()
    stops_train = x_train['stops'].to_numpy()
    arrival_time_train = x_train['arrival_time'].to_numpy()
    class_train = x_train['class'].to_numpy()
    duration_train = x_train['duration'].to_numpy()
    days_left_train = x_train['days_left'].to_numpy()
    price_train = y_train['price'].to_numpy()

    departure_time_test = x_test['departure_time'].to_numpy()
    stops_test = x_test['stops'].to_numpy()
    arrival_time_test = x_test['arrival_time'].to_numpy()
    class_test = x_test['class'].to_numpy()
    duration_test = x_test['duration'].to_numpy()
    days_left_test = x_test['days_left'].to_numpy()
    price_test = y_test['price'].to_numpy()

    normalization(departure_time_train, len(x_train.index), min_x_list[0], max_x_list[0])
    normalization(stops_train, len(x_train.index), min_x_list[1], max_x_list[1])
    normalization(arrival_time_train, len(x_train.index), min_x_list[2], max_x_list[2])
    normalization(class_train, len(x_train.index), min_x_list[3], max_x_list[3])
    normalization(duration_train, len(x_train.index), min_x_list[4], max_x_list[4])
    normalization(days_left_train, len(x_train.index), min_x_list[5], max_x_list[5])
    normalization(price_train, len(y_train.index), y_min, y_max)

    normalization(departure_time_test, len(x_test.index), min_x_list[0], max_x_list[0])
    normalization(stops_test, len(x_test.index), min_x_list[1], max_x_list[1])
    normalization(arrival_time_test, len(x_test.index), min_x_list[2], max_x_list[2])
    normalization(class_test, len(x_test.index), min_x_list[3], max_x_list[3])
    normalization(duration_test, len(x_test.index), min_x_list[4], max_x_list[4])
    normalization(days_left_test, len(x_test.index), min_x_list[5], max_x_list[5])
    normalization(price_test, len(y_test.index), y_min, y_max)

    return departure_time_train, stops_train, arrival_time_train, class_train, duration_train, days_left_train, \
           price_train, departure_time_test, stops_test, arrival_time_test, class_test, duration_test,\
           days_left_test, price_test


def normalization(np_array, length, min_items, max_items):
    for i in range(length):
        np.put(np_array, [i], (np_array[i] - max_items) / (max_items - min_items))
