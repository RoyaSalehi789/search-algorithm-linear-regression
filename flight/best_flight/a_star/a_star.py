from best_flight.dataframe import df, create_graph
import math


def h(src, des_latitude, des_longitude):
    df2 = df[df['SourceAirport'] == src]
    src_lat_list = list(set(df2['SourceAirport_Latitude']))
    src_long_list = list(set(df2['SourceAirport_Longitude']))

    radian = math.pi / 180
    radius = 6371
    coordinate = math.acos(
        math.sin(des_latitude * radian) * (math.sin(src_lat_list[0] * radian)) + (math.cos(des_latitude * radian)) * (
            math.cos(src_lat_list[0] * radian)) * (
            math.cos((des_longitude * radian) - (src_long_list[0] * radian)))) * radius

    return coordinate


def a_star_algorithm(src, des):

    nodes, init_graph = create_graph()
    open_lst = set([src])
    closed_lst = set([])

    g = {src: 0}

    parent = {src: src}

    df3 = df[df['DestinationAirport'] == des]
    des_lat_list = list(set(df3['DestinationAirport_Latitude']))
    des_long_list = list(set(df3['DestinationAirport_Longitude']))

    while len(open_lst) > 0:
        n = None

        for v in open_lst:

            if n is None or g[v] + h(v, des_lat_list[0], des_long_list[0]) < g[n] + h(n, des_lat_list[0],
                                                                                      des_long_list[0]):
                n = v
        if n is None:
            print('Path does not exist!')
            return None
        if n == des:
            path = []

            while parent[n] != n:
                path.append(n)
                n = parent[n]

            path.append(src)
            path.reverse()

            print('Path found: {}'.format(path))
            return path

        for (m, weight) in init_graph[n].items():
            if m not in open_lst and m not in closed_lst:
                open_lst.add(m)
                parent[m] = n
                g[m] = g[n] + weight
            else:
                if g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parent[m] = n

                    if m in closed_lst:
                        closed_lst.remove(m)
                        open_lst.add(m)
        open_lst.remove(n)
        closed_lst.add(n)

    print('Path does not exist!')
    return None
