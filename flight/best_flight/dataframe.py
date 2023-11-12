import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
import math
import time

df = pd.read_csv('Flight_Data.csv')


def dataframe():
    df_to_dict = df.to_dict()

    source_airport_dict = df_to_dict['SourceAirport']
    destination_airport_dict = df_to_dict['DestinationAirport']
    distance_dict = df_to_dict['Distance']
    source_airport_latitude_dict = df_to_dict['SourceAirport_Latitude']
    source_airport_altitude_dict = df_to_dict['SourceAirport_Altitude']
    source_airport_longitude_dict = df_to_dict['SourceAirport_Longitude']
    destination_airport_latitude_dict = df_to_dict['DestinationAirport_Latitude']
    destination_airport_altitude_dict = df_to_dict['DestinationAirport_Altitude']
    destination_airport_longitude_dict = df_to_dict['DestinationAirport_Longitude']
    fly_time_dict = df_to_dict['FlyTime']
    price_dict = df_to_dict['Price']

    source_airport_set = set(df['SourceAirport'])
    destination_airport_set = set(df['DestinationAirport'])
    destination_airport_set.update(source_airport_set)

    min_items_list = list(df.min())
    max_items_list = list(df.max())
    length = len(distance_dict)

    distance_dict = normalization(distance_dict, length, min_items_list[13], max_items_list[13])
    source_airport_latitude_dict = normalization(source_airport_latitude_dict, length, min_items_list[5],
                                                 max_items_list[5])
    source_airport_altitude_dict = normalization(source_airport_altitude_dict, length, min_items_list[7],
                                                 max_items_list[7])
    source_airport_longitude_dict = normalization(source_airport_longitude_dict, length, min_items_list[6],
                                                  max_items_list[6])
    destination_airport_latitude_dict = normalization(destination_airport_latitude_dict, length, min_items_list[10],
                                                      max_items_list[10])
    destination_airport_altitude_dict = normalization(destination_airport_altitude_dict, length, min_items_list[12],
                                                      max_items_list[12])
    destination_airport_longitude_dict = normalization(destination_airport_longitude_dict, length, min_items_list[11],
                                                       max_items_list[11])
    fly_time_dict = normalization(fly_time_dict, length, min_items_list[14], max_items_list[14])
    price_dict = normalization(price_dict, length, min_items_list[15], max_items_list[15])

    return source_airport_dict, destination_airport_dict, distance_dict, source_airport_latitude_dict, \
           source_airport_altitude_dict, source_airport_longitude_dict, destination_airport_latitude_dict, \
           destination_airport_altitude_dict, destination_airport_longitude_dict, fly_time_dict, price_dict, source_airport_set


def normalization(dic, length, min_items, max_items):
    for i in range(length):
        dic[i] = (dic[i] - min_items) / (max_items - min_items)

    return dic


def weight_cal():
    costs = []

    source_airport_dict, destination_airport_dict, distance_dict, source_airport_latitude_dict, \
    source_airport_altitude_dict, source_airport_longitude_dict, destination_airport_latitude_dict, \
    destination_airport_altitude_dict, destination_airport_longitude_dict, fly_time_dict, price_dict, source_airport_set = dataframe()
    for i in distance_dict:
        coordinates = math.sqrt(
            (math.pow(destination_airport_latitude_dict[i] - source_airport_latitude_dict[i], 2)) +
            (math.pow(destination_airport_altitude_dict[i] - source_airport_altitude_dict[i], 2)) +
            (math.pow(destination_airport_longitude_dict[i] - source_airport_longitude_dict[i], 2)))

        costs.append(
            (2 * coordinates) + fly_time_dict[i] + (4 * price_dict[i]) + (2 * distance_dict[i]))

    return costs


def create_graph():
    source_airport_dict, destination_airport_dict, distance_dict, source_airport_latitude_dict, \
    source_airport_altitude_dict, source_airport_longitude_dict, destination_airport_latitude_dict, \
    destination_airport_altitude_dict, destination_airport_longitude_dict, fly_time_dict, price_dict, source_airport_set = dataframe()

    costs = weight_cal()

    init_graph = {}
    nodes = list(source_airport_set)
    for node in nodes:
        init_graph[node] = {}

    for i in source_airport_dict:
        init_graph[source_airport_dict[i]][destination_airport_dict[i]] = costs[i]

    return nodes, init_graph


class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = init_graph

    def get_nodes(self):
        return self.nodes

    def get_outgoing_edges(self, node):
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections

    def value(self, node1, node2):
        return self.graph[node1][node2]
