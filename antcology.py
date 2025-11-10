import random
import networkx as nx


NUM_ITERATIONS = 30
EVAPORATION_RATE = 0.3
PHEROMONE_DEPOSIT = 1.0
ALPHA = 1   
BETA = 2      


G = nx.Graph()


edges = [
    ('A', 'B', 10),
    ('A', 'C', 5),
    ('B', 'C', 2),
    ('B', 'D', 7),
    ('C', 'D', 3),
    ('C', 'E', 8),
    ('D', 'E', 10)
]

for u, v, bw in edges:
    G.add_edge(u, v, bandwidth=bw, pheromone=1.0)


def select_next_node(current, visited):
    neighbors = list(G.neighbors(current))
    probabilities = []

    for node in neighbors:
        if node in visited:
            probabilities.append(0)
            continue

        pheromone = G[current][node]['pheromone'] ** ALPHA
        bandwidth = G[current][node]['bandwidth'] ** BETA
        probabilities.append(pheromone * bandwidth)

    total = sum(probabilities)
    if total == 0:
        return None

    probabilities = [p / total for p in probabilities]
    return random.choices(neighbors, weights=probabilities, k=1)[0]



def ant_colony(source, destination):
    best_path = None
    best_bw = 0

    for _ in range(NUM_ITERATIONS):
        paths = []
        path_bandwidths = []

        for _ in range(NUM_ANTS):
            current = source
            visited = [current]

            while current != destination:
                next_node = select_next_node(current, visited)
                if next_node is None:
                    visited = []
                    break
                visited.append(next_node)
                current = next_node

            if visited:
                # Path bandwidth = minimum bandwidth in path (bottleneck)
                bw = min(G[visited[i]][visited[i+1]]['bandwidth'] for i in range(len(visited)-1))
                paths.append(visited)
                path_bandwidths.append(bw)

                # Update global best
                if bw > best_bw:
                    best_bw = bw
                    best_path = visited

        # Evaporate pheromone
        for u, v in G.edges():
            G[u][v]['pheromone'] *= (1 - EVAPORATION_RATE)

        # Deposit pheromone for good paths
        for path, bw in zip(paths, path_bandwidths):
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                G[u][v]['pheromone'] += PHEROMONE_DEPOSIT * bw

    return best_path, best_bw

best_path, max_bw = ant_colony('A', 'E')
print("Best Path Found:", best_path)
print("Maximum Achievable Bandwidth:", max_bw)
