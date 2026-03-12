import heapq
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ===============================
# Grid Setup
# ===============================

GRID_SIZE = 30
OBSTACLE_PROB = 0.25

def generate_grid(size, obstacle_prob):
    grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if random.random() < obstacle_prob:
                grid[i][j] = 1
    return grid

def get_neighbors(node, grid):
    x, y = node
    neighbors = []
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx][ny] == 0:
                neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# ===============================
# BFS
# ===============================

def bfs(grid, start, goal):
    queue = deque([start])
    visited = set([start])
    came_from = {}
    nodes_expanded = 0

    start_time = time.time()

    while queue:
        current = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            runtime = time.time() - start_time
            path = reconstruct_path(came_from, start, goal)
            return path, nodes_expanded, runtime

        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return None, nodes_expanded, time.time() - start_time

# ===============================
# Dijkstra
# ===============================

def dijkstra(grid, start, goal):
    pq = [(0, start)]
    distances = {start: 0}
    came_from = {}
    visited = set()
    nodes_expanded = 0

    start_time = time.time()

    while pq:
        cost, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            runtime = time.time() - start_time
            path = reconstruct_path(came_from, start, goal)
            return path, nodes_expanded, runtime

        for neighbor in get_neighbors(current, grid):
            new_cost = cost + 1
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    return None, nodes_expanded, time.time() - start_time

# ===============================
# A*
# ===============================

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    pq = [(0, start)]
    g_cost = {start: 0}
    came_from = {}
    visited = set()
    nodes_expanded = 0

    start_time = time.time()

    while pq:
        _, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            runtime = time.time() - start_time
            path = reconstruct_path(came_from, start, goal)
            return path, nodes_expanded, runtime

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_cost[current] + 1
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(pq, (f_cost, neighbor))

    return None, nodes_expanded, time.time() - start_time

# ===============================
# Visualization
# ===============================

def visualize(grid, path, start, goal, title):
    display = np.copy(grid)
    if path:
        for x, y in path:
            display[x][y] = 0.5
    display[start] = 0.7
    display[goal] = 0.9

    plt.imshow(display, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.show()

# ===============================
# Main Execution
# ===============================

def main():
    grid = generate_grid(GRID_SIZE, OBSTACLE_PROB)

    start = (0, 0)
    goal = (GRID_SIZE - 1, GRID_SIZE - 1)

    grid[start] = 0
    grid[goal] = 0

    results = {}

    for name, algo in [("BFS", bfs), ("Dijkstra", dijkstra), ("A*", astar)]:
        path, nodes, runtime = algo(grid, start, goal)
        if path:
            results[name] = {
                "Path Length": len(path),
                "Nodes Expanded": nodes,
                "Runtime (s)": runtime
            }
            print(f"\n{name} Results:")
            print(f"Path Length: {len(path)}")
            print(f"Nodes Expanded: {nodes}")
            print(f"Runtime: {runtime:.6f} seconds")

            visualize(grid, path, start, goal, name)
        else:
            print(f"{name} failed to find a path.")

    print("\n=== Comparison Table ===")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")

if __name__ == "__main__":
    main()
