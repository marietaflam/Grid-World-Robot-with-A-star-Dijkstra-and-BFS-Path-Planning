import heapq
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ===============================
# Configuration
# ===============================
GRID_SIZE = 30
OBSTACLE_PROB = 0.15  # lower probability to make paths more likely
TERRAIN_WEIGHTS = [1, 2, 3]  # terrain costs

# ===============================
# Grid generation
# ===============================
def generate_grid(size, obstacle_prob):
    grid = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if random.random() < obstacle_prob:
                grid[i][j] = -1  # obstacle
            else:
                grid[i][j] = random.choice(TERRAIN_WEIGHTS)
    return grid

def get_neighbors(node, grid):
    x, y = node
    neighbors = []
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx][ny] != -1:
                neighbors.append((nx, ny))
    return neighbors

# ===============================
# Path reconstruction
# ===============================
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
            return reconstruct_path(came_from, start, goal), nodes_expanded, runtime
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
    pq = [(grid[start], start)]
    distances = {start: grid[start]}
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
            return reconstruct_path(came_from, start, goal), nodes_expanded, runtime
        for neighbor in get_neighbors(current, grid):
            new_cost = cost + grid[neighbor]
            if neighbor not in distances or new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))
    return None, nodes_expanded, time.time() - start_time

# ===============================
# A* with Manhattan heuristic
# ===============================
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    pq = [(grid[start] + heuristic(start, goal), start)]
    g_cost = {start: grid[start]}
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
            return reconstruct_path(came_from, start, goal), nodes_expanded, runtime
        for neighbor in get_neighbors(current, grid):
            tentative_g = g_cost[current] + grid[neighbor]
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
    display[display == -1] = np.nan  # obstacles as NaN for black

    cmap = plt.cm.viridis
    cmap.set_bad(color='black')

    plt.figure(figsize=(6,6))
    plt.imshow(display, cmap=cmap)

    if path:
        for x, y in path:
            plt.plot(y, x, marker='s', color='red', markersize=8)

    plt.plot(start[1], start[0], marker='o', color='green', markersize=10, label='Start')
    plt.plot(goal[1], goal[0], marker='o', color='blue', markersize=10, label='Goal')

    plt.title(title)
    plt.colorbar(label="Terrain Cost")
    plt.legend(loc='upper left')
    plt.show()

# ===============================
# Main Execution with guaranteed path
# ===============================
def main():
    start = (0, 0)
    goal = (GRID_SIZE-1, GRID_SIZE-1)

    # Retry grid generation until a path exists
    while True:
        grid = generate_grid(GRID_SIZE, OBSTACLE_PROB)
        grid[start] = 1
        grid[goal] = 1
        path, _, _ = astar(grid, start, goal)
        if path:
            break

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
