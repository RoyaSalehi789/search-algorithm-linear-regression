from best_flight.dijkstra import dijkstra as dj
from best_flight.dataframe import Graph, create_graph, df
from best_flight.a_star.a_star import a_star_algorithm
import time


def text_write(start_time, path, algorithm):
    total_time = 0
    total_price = 0
    total_duration = 0
    f = open(f"[7]-UIAI4021-PR1-Q1([{algorithm}]).txt", "w")
    f.write(f"{algorithm} \n")
    f.write(f"Execution Time: {(time.time() - start_time): .2}s \n")
    f.write(".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- \n")
    for i in range(len(path) - 1):
        df6 = df[(df['SourceAirport'] == path[i]) & (df['DestinationAirport'] == path[i + 1])]
        f.write(f"Flight #{i + 1} ({df6.iloc[0]['Airline']}): \n")
        f.write(
            f"From: {df6.iloc[0]['SourceAirport']} - {df6.iloc[0]['SourceAirport_City']}, {df6.iloc[0]['SourceAirport_Country']} \n")
        f.write(
            f"To: {df6.iloc[0]['DestinationAirport']} - {df6.iloc[0]['DestinationAirport_City']}, {df6.iloc[0]['DestinationAirport_Country']} \n")
        f.write(f"Duration: {round(df6.iloc[0]['Distance'], 2)}km \n")
        total_duration += df6.iloc[0]['Distance']
        f.write(f"Time: {round(df6.iloc[0]['FlyTime'], 2)}h \n")
        total_time += df6.iloc[0]['FlyTime']
        f.write(f"Price: {round(df6.iloc[0]['Price'], 2)}$ \n")
        total_price += df6.iloc[0]['Price']
        f.write("---------------------------- \n")

    f.write(f"Total Duration: {round(total_duration, 2)}km \n")
    f.write(f"Total Time: {round(total_time, 2)}h \n")
    f.write(f"Total Price: {round(total_price, 2)}$ \n")
    f.close()


def main():
    print("""
    [1]. Dijkstra
    [2]. A*
    ----------------
    """)
    x = input('Enter a number:')

    if x == '1':
        start = input('Enter Your Source Airport')
        target = input('Enter Your Destination Airport')

        nodes, init_graph = create_graph()
        graph = Graph(nodes, init_graph)
        start_time = time.time()
        previous_nodes, shortest_path = dj.dijkstra_algorithm(graph=graph,
                                                              start_node=start)
        path = dj.print_result(previous_nodes, shortest_path, start_node=start,
                               target_node=target)
        text_write(start_time, path, 'Dijkstra')

    if x == '2':
        start = input('Enter Your Source Airport')
        target = input('Enter Your Destination Airport')
        start_time = time.time()
        path = a_star_algorithm(start, target)
        text_write(start_time, path, "A_Star")


if __name__ == '__main__':
    main()
