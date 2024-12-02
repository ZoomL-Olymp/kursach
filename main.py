import osmnx as ox
import numpy as np
import networkx as nx
import itertools
import shapely
import matplotlib.pyplot as plt

# 1. Достопримечательности (теперь только названия)
locations_names = {
    "Эрмитаж",
    "Исаакиевский собор",
    "Петропавловская крепость",
    "Храм Спаса на Крови",
    "Русский музей",
    "Кунсткамера",
    "Дворцовая площадь",
    "Летний сад",
    "Казанский собор",
    "Адмиралтейство",
    "Медный всадник",
    "Стрелка Васильевского острова",
    "Мариинский театр",
    # Добавьте еще достопримечательности по желанию
}

# 2. Функция для получения графа дорог и координат достопримечательностей
def get_graph_and_locations(place="Saint Petersburg, Russia", tags={"tourism": "attraction"}):
    """
    Получает граф дорог и координаты достопримечательностей с помощью OSM.
    """
    try:
        graph = ox.graph_from_place(place, network_type='drive', retain_all=True, simplify=True)
        pois = ox.features_from_place(place, tags=tags)

        print(pois)

        locations = {}
        for name in locations_names.copy(): # Iterate over a copy to avoid RuntimeError
            #  Fuzzy string matching for better results
            matching_pois = pois[pois["name"].str.contains(name, case=False, na=False, regex=False)]
            if not matching_pois.empty:
                location = matching_pois.iloc[0].geometry
                if isinstance(location, shapely.geometry.point.Point): # Check if it's a Point
                    locations[name] = (location.y, location.x)  # Correct order: latitude, longitude
                elif isinstance(location, shapely.geometry.polygon.Polygon):
                    centroid = location.centroid
                    locations[name] = (centroid.y, centroid.x) # Use centroid for polygons
                else:
                    print(f"Warning: Unexpected geometry type for '{name}'. Skipping.")
                    continue
                    
            else:
                print(f"Warning: POI '{name}' not found using fuzzy matching. Skipping.")
                locations_names.discard(name)

        return graph, locations

    except ValueError:
        print(f"Place '{place}' not found in OpenStreetMap.")
        return None



# 3. Функция для получения матрицы расстояний
def get_distance_matrix(graph, locations):
    """Вычисляет матрицу расстояний."""

    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))

    for i, (loc1_name, loc1_coords) in enumerate(locations.items()):
        for j, (loc2_name, loc2_coords) in enumerate(locations.items()):
            if i != j:
                try:
                    origin_node = ox.distance.nearest_nodes(graph,  loc1_coords[1], loc1_coords[0])
                    destination_node = ox.distance.nearest_nodes(graph,  loc2_coords[1], loc2_coords[0])
                    distance = nx.shortest_path_length(graph, origin_node, destination_node, weight='length')
                    distance_matrix[i, j] = distance
                except nx.NetworkXNoPath:
                    print(f"Путь между {loc1_name} and {loc2_name} не найден.")
                    distance_matrix[i, j] = np.inf  # Or a very large number
    return distance_matrix



# 4. Функция для решения TSP (пример с полным перебором - для небольшого числа точек)
def solve_tsp(distance_matrix):
    num_locations = len(distance_matrix)
    best_path = None
    min_distance = float('inf')

    for path in itertools.permutations(range(num_locations)):
        total_distance = 0
        for i in range(num_locations - 1):
            total_distance += distance_matrix[path[i]][path[i+1]]
        total_distance += distance_matrix[path[-1]][path[0]]  # Возвращение в начало

        if total_distance < min_distance:
            min_distance = total_distance
            best_path = path

    return best_path


# 5. Функция для вывода результатов (без изменений)
def print_results(locations, path):
    print("Оптимальный маршрут:")
    for index in path:
        print(list(locations.keys())[index])

def visualize_route(graph, locations, path):
    """Визуализирует маршрут на карте с порядковыми номерами."""

    route_nodes = []
    for i in range(len(path)):
        loc1_name = list(locations.keys())[path[i]]
        loc1_coords = locations[loc1_name]
        route_nodes.append(ox.distance.nearest_nodes(graph, loc1_coords[1], loc1_coords[0]))
        if i < len(path) - 1:
            loc2_name = list(locations.keys())[path[i+1]]
            loc2_coords = locations[loc2_name]
            route = nx.shortest_path(graph, route_nodes[-1], ox.distance.nearest_nodes(graph, loc2_coords[1], loc2_coords[0]), weight='length')
            route_nodes.extend(route[1:])

    route_nodes_coords = np.array([(graph.nodes[node]['x'], graph.nodes[node]['y']) for node in route_nodes])

    fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=0, edge_linewidth=0.5, bgcolor='white')

    xs = [locations[list(locations.keys())[path[i]]][1] for i in range(len(path))]
    ys = [locations[list(locations.keys())[path[i]]][0] for i in range(len(path))]

    ax.scatter(xs, ys, c='red', s=50, zorder=3, label="POIs", edgecolors='white', linewidth=0.5)
    ax.plot(route_nodes_coords[:, 0], route_nodes_coords[:, 1], c='blue', linewidth=2, zorder=2, label="Route")

    for i in range(len(path)):
        loc_name = list(locations.keys())[path[i]]
        x = locations[loc_name][1]
        y = locations[loc_name][0]
        ax.text(x, y, f"{i+1}. {loc_name}", fontsize=8, color="black",
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.2),
                verticalalignment='center', horizontalalignment='center', zorder=4)


    ax.set_title("Optimal Tourist Route")
    plt.legend()
    plt.show()


def get_real_locations(place, tags, locations_names):

    pois = ox.features_from_place(place, tags=tags)
    pois = pois[pois["name"].notna()] #filter out pois with no name
    locations = {}

    for name in locations_names.copy():
        matching_pois = pois[pois["name"].str.contains(name, case=False, na=False, regex=False)]
        if not matching_pois.empty:
            poi = matching_pois.iloc[0]
            location = poi.geometry
            if isinstance(location, shapely.geometry.point.Point):
                locations[poi['name']] = (location.y, location.x) # store the real name
            elif isinstance(location, shapely.geometry.polygon.Polygon):
                centroid = location.centroid
                locations[poi['name']] = (centroid.y, centroid.x)
            else:
                print(f"Warning: Unexpected geometry type for '{name}'. Skipping.")
            locations_names.discard(name) # remove from set only when matched


    # If some locations are not found, add random POIs from available ones:
    num_missing = len(locations_names)
    if num_missing > 0:
        available_pois = list(pois['name'])
        random_pois = np.random.choice(available_pois, size=min(num_missing, len(available_pois)), replace=False)

        for poi_name in random_pois:
            poi = pois[pois["name"] == poi_name].iloc[0]
            location = poi.geometry
            if isinstance(location, shapely.geometry.point.Point):
                locations[poi_name] = (location.y, location.x)
            elif isinstance(location, shapely.geometry.polygon.Polygon):
                centroid = location.centroid
                locations[poi_name] = (centroid.y, centroid.x)
            else:
                 print("Warning: unexpected geometry type for '{poi_name}'. Skipping.")



    return locations

def main():
    place = "Saint Petersburg, Russia"
    tags = {"tourism": "attraction"}
    locations_names = {  # Use a set to add more and ensure uniqueness
        "Эрмитаж", "Исаакиевский собор", "Петропавловская крепость",
        "Храм Спаса на Крови", "Русский музей", "Кунсткамера", "Зимний дворец"
    }
    graph = ox.graph_from_place(place, network_type='drive', retain_all=True, simplify=True)
    locations = get_real_locations(place, tags, locations_names) #Get the locations here
    distance_matrix = get_distance_matrix(graph, locations)
    if distance_matrix is not None:
        path = solve_tsp(distance_matrix)
        print_results(locations, path)
        visualize_route(graph, locations, path)



if __name__ == "__main__":
    main()