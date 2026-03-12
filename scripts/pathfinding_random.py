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
                grid[i][j] = 1  # obstacle
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
            return reconstruct_path(came_from, start, goal), nodes_expanded, runtime

        for neighbor in get_neighbors(current, grid):
            new_cost = cost + 1
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
            return reconstruct_path(came_from, start, goal), nodes_expanded, runtime

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_cost[current] + 1
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(pq, (f_cost, neighbor))
    return None, nodes_expanded, time.time() - start_time

# ===============================
# Run 10 random grids and evaluate
# ===============================
def evaluate_algorithms(n_grids=10):
    start = (0,0)
    goal = (GRID_SIZE-1, GRID_SIZE-1)
    algorithms = [("BFS", bfs), ("Dijkstra", dijkstra), ("A*", astar)]
    results = {name: {"Path Length": [], "Nodes Expanded": [], "Runtime": []} for name,_ in algorithms}
    grids_and_paths = []  # store grids and paths for visualization

    for i in range(n_grids):
        while True:  # ensure at least one path exists
            grid = generate_grid(GRID_SIZE, OBSTACLE_PROB)
            grid[start] = 0
            grid[goal] = 0
            path, _, _ = astar(grid, start, goal)
            if path:
                break

        grid_result = {"grid": grid, "paths": {}}

        for name, algo in algorithms:
            path, nodes, runtime = algo(grid, start, goal)
            grid_result["paths"][name] = path
            results[name]["Path Length"].append(len(path))
            results[name]["Nodes Expanded"].append(nodes)
            results[name]["Runtime"].append(runtime)

        grids_and_paths.append(grid_result)

    # Print average metrics
    for name in results:
        avg_len = np.mean(results[name]["Path Length"])
        avg_nodes = np.mean(results[name]["Nodes Expanded"])
        avg_runtime = np.mean(results[name]["Runtime"])
        print(f"\n{name} Average over {n_grids} grids:")
        print(f"Average Path Length: {avg_len:.2f}")
        print(f"Average Nodes Expanded: {avg_nodes:.2f}")
        print(f"Average Runtime (s): {avg_runtime:.6f}")

    return grids_and_paths, results

# ===============================
# Show all grids side-by-side for each grid
# ===============================
def show_all_grids_side_by_side(grids_and_paths, algorithms=["BFS","Dijkstra","A*"]):
    n_grids = len(grids_and_paths)
    
    for i, grid_info in enumerate(grids_and_paths):
        fig, axs = plt.subplots(1, len(algorithms), figsize=(5*len(algorithms),5))
        if len(algorithms) == 1:
            axs = [axs]  # ensure iterable
        grid = grid_info["grid"]
        
        for ax, algo_name in zip(axs, algorithms):
            path = grid_info["paths"][algo_name]
            ax.imshow(grid, cmap="viridis")
            
            if path:
                for x, y in path:
                    ax.plot(y, x, marker='s', color='red', markersize=4)
            
            ax.plot(0,0, marker='o', color='green', markersize=6)  # start
            ax.plot(GRID_SIZE-1, GRID_SIZE-1, marker='o', color='blue', markersize=6)  # goal
            ax.set_title(f"{algo_name}")
            ax.axis('off')
        
        plt.suptitle(f"Grid {i+1} - BFS vs Dijkstra vs A*", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    grids_and_paths, results = evaluate_algorithms(n_grids=10)
    show_all_grids_side_by_side(grids_and_paths)
