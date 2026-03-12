import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import matplotlib.pyplot as plt

from collections import deque
import heapq
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===============================
# PARAMETERS
# ===============================

GRID_SIZE = 30
OBSTACLE_PROB = 0.25
N_GRIDS = 500

START = (0,0)
GOAL = (GRID_SIZE-1, GRID_SIZE-1)

# ===============================
# GRID GENERATION
# ===============================

def generate_grid():

    grid = np.zeros((GRID_SIZE,GRID_SIZE))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):

            if random.random() < OBSTACLE_PROB:
                grid[i][j] = 1

    grid[START] = 0
    grid[GOAL] = 0

    return grid


# ===============================
# NEIGHBORS
# ===============================

def get_neighbors(node, grid):

    x,y = node
    neighbors = []

    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    for dx,dy in directions:

        nx = x + dx
        ny = y + dy

        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if grid[nx][ny] == 0:
                neighbors.append((nx,ny))

    return neighbors


# ===============================
# PATH RECONSTRUCTION
# ===============================

def reconstruct_path(came_from):

    path = []
    current = GOAL

    while current != START:
        path.append(current)
        current = came_from[current]

    path.append(START)
    path.reverse()

    return path


# ===============================
# BFS
# ===============================

def bfs(grid):

    queue = deque([START])
    visited = {START}

    came_from = {}

    while queue:

        current = queue.popleft()

        if current == GOAL:
            return reconstruct_path(came_from)

        for n in get_neighbors(current,grid):

            if n not in visited:

                visited.add(n)
                came_from[n] = current
                queue.append(n)

    return None


# ===============================
# DIJKSTRA
# ===============================

def dijkstra(grid):

    pq = [(0,START)]

    dist = {START:0}
    came_from = {}

    visited=set()

    while pq:

        cost,current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        if current == GOAL:
            return reconstruct_path(came_from)

        for n in get_neighbors(current,grid):

            new_cost = cost + 1

            if n not in dist or new_cost < dist[n]:

                dist[n] = new_cost
                came_from[n] = current

                heapq.heappush(pq,(new_cost,n))

    return None


# ===============================
# A*
# ===============================

def heuristic(a,b):

    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def astar(grid):

    pq=[(0,START)]

    g_cost={START:0}
    came_from={}

    visited=set()

    while pq:

        _,current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)

        if current == GOAL:
            return reconstruct_path(came_from)

        for n in get_neighbors(current,grid):

            new_g = g_cost[current] + 1

            if n not in g_cost or new_g < g_cost[n]:

                g_cost[n] = new_g

                f = new_g + heuristic(n,GOAL)

                came_from[n] = current

                heapq.heappush(pq,(f,n))

    return None


# ===============================
# GENERATE SOLVABLE GRIDS
# ===============================

def evaluate_algorithms():

    grids=[]

    while len(grids) < N_GRIDS:

        grid = generate_grid()

        paths = {

            "BFS": bfs(grid),
            "Dijkstra": dijkstra(grid),
            "A*": astar(grid)

        }

        if all(paths.values()):
            grids.append({
                "grid":grid,
                "paths":paths
            })

    return grids


# ===============================
# DATASET
# ===============================

def generate_next_move_data(grids,algo):

    X=[]
    y=[]

    for g in grids:

        grid=g["grid"]
        path=g["paths"][algo]

        for i in range(len(path)-1):

            cur=path[i]
            nxt=path[i+1]

            dx=nxt[0]-cur[0]
            dy=nxt[1]-cur[1]

            if dx==-1 and dy==0: move=0
            elif dx==1 and dy==0: move=1
            elif dx==0 and dy==-1: move=2
            elif dx==0 and dy==1: move=3
            else: continue

            tensor=np.zeros((3,GRID_SIZE,GRID_SIZE))

            tensor[0]=(grid==0)
            tensor[1,cur[0],cur[1]]=1
            tensor[2,GOAL[0],GOAL[1]]=1

            X.append(tensor)
            y.append(move)

    X=torch.tensor(np.array(X),dtype=torch.float32)
    y=torch.tensor(y,dtype=torch.long)

    return X,y


class DatasetMoves(Dataset):

    def __init__(self,X,y):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.X)

    def __getitem__(self,i):
        return self.X[i],self.y[i]


# ===============================
# CNN MODEL
# ===============================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)

        self.fc1 = nn.Linear(32*GRID_SIZE*GRID_SIZE,128)
        self.fc2 = nn.Linear(128,4)

    def forward(self,x):

        x=torch.relu(self.conv1(x))
        x=torch.relu(self.conv2(x))

        x=x.view(x.size(0),-1)

        x=torch.relu(self.fc1(x))

        return self.fc2(x)


# ===============================
# TRAIN
# ===============================

def train_model(X,y):

    dataset=DatasetMoves(X,y)

    loader=DataLoader(dataset,batch_size=32,shuffle=True)

    model=CNN()

    optimizer=optim.Adam(model.parameters(),lr=0.001)

    loss_fn=nn.CrossEntropyLoss()

    for epoch in range(20):

        total=0

        for xb,yb in loader:

            optimizer.zero_grad()

            pred=model(xb)

            loss=loss_fn(pred,yb)

            loss.backward()

            optimizer.step()

            total+=loss.item()

        print("Epoch",epoch+1,"Loss:",total/len(loader))

    return model


# ===============================
# SIMULATE AI PATH
# ===============================

def simulate_path(model,grid):

    current=START

    path=[current]

    for _ in range(GRID_SIZE*GRID_SIZE):

        if current==GOAL:
            break

        tensor=np.zeros((3,GRID_SIZE,GRID_SIZE))

        tensor[0]=(grid==0)
        tensor[1,current[0],current[1]]=1
        tensor[2,GOAL[0],GOAL[1]]=1

        x=torch.tensor(tensor[np.newaxis],dtype=torch.float32)

        with torch.no_grad():
            move=torch.argmax(model(x)).item()

        dx,dy=0,0

        if move==0: dx=-1
        if move==1: dx=1
        if move==2: dy=-1
        if move==3: dy=1

        nxt=(current[0]+dx,current[1]+dy)

        if 0<=nxt[0]<GRID_SIZE and 0<=nxt[1]<GRID_SIZE and grid[nxt]==0:

            current=nxt
            path.append(current)

        else:
            break

    return path


# ===============================
# METRICS
# ===============================

def path_similarity(true,pred):

    return len(set(true)&set(pred)) / len(true)


def path_length_error(true,pred):

    return abs(len(true)-len(pred))


def evaluate_ai(models,grids):

    results={}

    for algo in models:

        success=0
        sim=0
        err=0

        for g in grids:

            grid=g["grid"]
            true=g["paths"][algo]

            pred=simulate_path(models[algo],grid)

            if pred[-1]==GOAL:
                success+=1

            sim+=path_similarity(true,pred)
            err+=path_length_error(true,pred)

        n=len(grids)

        results[algo]={
            "success":success/n,
            "similarity":sim/n,
            "length_error":err/n
        }

    return results


# ===============================
# VISUALIZATION
# ===============================

def visualize(grid,true_paths,ai_paths):

    plt.figure(figsize=(6,6))

    plt.imshow(grid,cmap="gray_r")

    colors={"BFS":"red","Dijkstra":"orange","A*":"purple"}

    for algo in true_paths:

        t=true_paths[algo]
        p=ai_paths[algo]

        xs=[x for x,_ in t]
        ys=[y for _,y in t]

        plt.plot(ys,xs,"--",color=colors[algo],label=algo+" True")

        xs=[x for x,_ in p]
        ys=[y for _,y in p]

        plt.plot(ys,xs,color=colors[algo],label=algo+" AI")

    plt.scatter(START[1],START[0],c="green",s=80,label="Start")
    plt.scatter(GOAL[1],GOAL[0],c="blue",s=80,label="Goal")

    plt.legend()

    plt.show()


# ===============================
# MAIN
# ===============================

grids=evaluate_algorithms()

models={}

for algo in ["BFS","Dijkstra","A*"]:

    print("\nTraining",algo)

    X,y=generate_next_move_data(grids,algo)

    models[algo]=train_model(X,y)

results=evaluate_ai(models,grids)

print("\nAI PERFORMANCE\n")

for k,v in results.items():

    print(k)
    print("Success Rate:",v["success"])
    print("Path Similarity:",v["similarity"])
    print("Length Error:",v["length_error"])
    print()


sample=random.sample(grids,5)

for g in sample:

    grid=g["grid"]

    ai_paths={
        algo:simulate_path(models[algo],grid)
        for algo in models
    }

    visualize(grid,g["paths"],ai_paths)
