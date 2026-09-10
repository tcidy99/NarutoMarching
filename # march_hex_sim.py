import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    try:
        matplotlib.use('Qt5Agg')
    except Exception:
        pass
import matplotlib.pyplot as plt
import os
import tempfile

# detect whether the chosen backend is interactive (may still fail at runtime)
_backend = matplotlib.get_backend().lower()
# consider GUI backend present if typical GUI names appear
GUI_AVAILABLE = any(k in _backend for k in ('tk', 'qt', 'wx', 'gtk', 'macosx'))
print(f"Matplotlib backend: {_backend}, GUI_AVAILABLE={GUI_AVAILABLE}")

import numpy as np
from matplotlib.patches import RegularPolygon
import csv
import json
import random
import math
from collections import deque

def read_csv_to_2d_vector(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

file_path = 'map_S19.csv' 
map = read_csv_to_2d_vector(file_path)
map.reverse()
startRow = 53
startCol = 8
visited = set()
#valid_moves = []
initNeighbors=[]
#Actions = [(r, c) for r in range(len(map)) for c in range(len(map[0]))]

#for row in map[:]:
#    print(row)
# Load the JSON file
with open('landInfo.json', 'r') as file:
    map_properties = json.load(file)

initFood=2000

class Hexagon:
    def __init__(self, x, y, Food, Award, Valid, Occupied, Edge, Face, Pattern, Label, Name):
        self.x = x
        self.y = y
        self.Food = Food
        self.Award = Award
        self.Valid = Valid
        self.Occupied = Occupied
        self.Edge = Edge
        self.Face = Face
        self.Pattern = Pattern
        self.Label = Label
        self.Name = Name


def hexagon_grid(rows, cols, hex_size,startRow,startCol):
    # Calculate width and height of the grid
    print(len(map))
    print(len(map[0]))
    width = cols * 1.5 * hex_size
    height = rows * np.sqrt(3) * hex_size
    StartRow = startRow
    StartCol = startCol

    
    marchMap = []

    for row in range(rows):
        hex_row = []
        for col in range(cols):
            x_offset = col * 1.5 * hex_size
            y_offset = row * np.sqrt(3) * hex_size
            if col % 2 == 1:
                y_offset += np.sqrt(3) / 2 * hex_size
            
            mapCode=map[row][col]
            properties = map_properties.get(mapCode, map_properties["default"])
            # Set the properties
            name = properties["name"]
            edge = properties["edge"]
            face = properties["face"]
            food = properties["food"]
            award = properties["award"]
            pattern = properties["hatch"]
            label = properties["label"]
            name = properties["name"]
            
            if name=="empty":
                valid = False
            else:
                valid = True      
            # Highlight the start position
            if row == rows-StartRow and col == StartCol-1:
                edge = 'red'
                face = 'yellow'
            hexagon_patch = RegularPolygon((x_offset, y_offset), numVertices=6, radius=hex_size, orientation=np.radians(30), edgecolor=edge, facecolor=face,hatch=pattern)
            #ax.add_patch(hexagon_patch)
            
            # Create a Hexagon object with coordinates and random food and award values
            hex_obj = Hexagon(x=row, y=col, Food=food, Award=award,Valid=valid,Occupied=False, Edge=edge,Face=face, Pattern=pattern, Label=label, Name=name)
            hex_row.append(hex_obj)
            #ax.text(x_offset, y_offset, f'({rows-row},{col+1})', ha='center', va='center', fontsize=9)
            #ax.text(x_offset, y_offset, label, fontsize=20)

        marchMap.append(hex_row)
    

    return marchMap

def resetMap(marchMap):
    for row in marchMap:
        for hexagon in row:
            hexagon.Occupied = False


        
# Define the game environment
class Calculator:
    def __init__(self, marchMap):
        self.marchMap = marchMap

    def calculate_award(self, player_pos):
        row, col = player_pos
        hexagon = self.marchMap[row][col]
        return hexagon.Award

    def calculate_food(self, current_food):
        return current_food - 1

class Player:
    
    def __init__(self, marchMap, start_pos=(startRow, startCol), food=initFood):
        self.marchMap = marchMap
        self.player_pos = start_pos
        self.food = food
        self.total_award = 0
        self.rows = len(marchMap)
        self.cols = len(marchMap[0])
        self.done = False
        self.path = [start_pos]  # Initialize the path with the start
        self.pathType = ['m']
        self.valid_moves = []
    def getLandInfo(self, player_pos):
        row, col = player_pos
        #print(f"getLandInfo: player_pos={player_pos}, rows={self.rows}, cols={self.cols}")
        if 0 <= self.rows - row < self.rows and 0 <= col - 1 < self.cols:
            return self.marchMap[self.rows - row][col - 1]
        else:
            raise IndexError("Invalid row or column index")


    def get_state(self):
        return (self.player_pos, self.food, self.total_award)

    def is_valid_move(self, new_pos):
        row, col = new_pos
        #if not(0 < row <= self.rows and 0 < col <= self.cols):
        #    return False
        hexagon = self.getLandInfo(new_pos)
        #if not(hexagon.Valid):
        #    return False
        #if hexagon.Occupied:
        #    return False
        #print(f'position: {new_pos}')
        if not(self.is_adjacent_or_connected(new_pos)):
            n = self.find_min_occupied_path(self.player_pos, new_pos)
            #print(f'n={n}')
            return self.food >= hexagon.Food + n*10
        else:
            return self.food >= hexagon.Food
    
    def is_adjacent_or_connected(self,hex_pos):
            row, col = hex_pos
            row0, col0 = self.player_pos
            if col0%2:#odd col
                if ((row==row0-1 and col==col0) or (row==row0 and col==col0+1) or (row==row0+1 and col==col0+1) or (row==row0+1 and col==col0) or (row==row0+1 and col==col0-1) or (row==row0 and col==col0-1)):
                    #print('adj found')
                    return True
            else:
                if ((row==row0-1 and col==col0) or (row==row0-1 and col==col0+1) or (row==row0 and col==col0+1) or (row==row0+1 and col==col0) or (row==row0 and col==col0-1) or (row==row0-1 and col==col0-1)):
                    return True
            return False
    
    def get_valid_moves(self):
        Actions = [(r, c) for r in range(self.rows) for c in range(self.cols)]  # Cover all hexagons in marchMap
        for action in Actions:
            # Check if the hexagon is valid, not occupied, and not the current position
            row, col = action
            row=self.rows-row
            col=col+1
            hexagon = self.getLandInfo((row,col))
            if (
                hexagon.Valid and not hexagon.Occupied and (row, col) != self.player_pos and 
                self.is_adjacent_or_connected((row, col)) and (row, col) not in self.valid_moves and 
                self.is_valid_move((row, col))
            ):
                self.valid_moves.append((row,col)) 
        #print(f'player_pos:{self.player_pos}')
        if self.player_pos==(startRow,startCol):
            #print('set initNeighbors========================')
            self.initNeighbors=self.valid_moves
        #print(f'initNeighbors:{self.initNeighbors}')
        #print(f'valid_moves:{self.valid_moves}')
        #check if food is enough for next move
        final_valid_moves=[]
        for m in self.valid_moves:
            if (self.is_valid_move(m)):
                final_valid_moves.append(m)
        #print(f'valid_moves after food cal:{final_valid_moves}')  
        self.valid_moves=final_valid_moves
        return final_valid_moves
    
    def find_min_occupied_path(self,start_pos, end_pos):
            #print(f'checking steps between {start_pos} and {end_pos}')
            # Perform Breadth-First Search (BFS) to find the minimum path through occupied hexagons
            queue = [(start_pos, 0)]  # (current_position, occupied_count)
            visited = set()
            while queue:
                current_pos, occupied_count = queue.pop(0)
                #print(current_pos)
                if current_pos == end_pos:
                    return occupied_count
                visited.add(current_pos)
                row, col = current_pos
                if col%2:#odd col
                    neighbors = [
                    (row - 1, col), (row, col + 1), (row + 1, col + 1), (row + 1, col),
                    (row + 1, col - 1), (row, col - 1)  # Add diagonal connections for hex grids
                    ]
                else:
                    neighbors = [
                    (row - 1, col), (row - 1, col + 1), (row, col + 1), (row + 1, col),
                    (row, col - 1), (row - 1, col - 1)  # Add diagonal connections for hex grids
                ]
                #print(f'neighbors: {neighbors}')
                for neighbor in neighbors:
                    n_row, n_col = neighbor
                    #if neighbor==end_pos:
                        #print(f'found: {occupied_count}')
                    if 0 < n_row <= self.rows and 0 < n_col <= self.cols and neighbor not in visited:
                        hexagon = self.getLandInfo((n_row,n_col))
                        #print(f'({n_row},{n_col}), {hexagon.Occupied}')
                        if hexagon.Valid and hexagon.Occupied:  # Pass through occupied hexagons
                            queue.append((neighbor, occupied_count + 1))
                        elif hexagon.Valid and not hexagon.Occupied:  # Unoccupied hexagons
                            if (n_row,n_col) == end_pos:
                                return occupied_count
                            #queue.append((neighbor, occupied_count))
                        #print(f"Visited: {visited}, Queue: {queue}")
                        #print(f'occupied_count={occupied_count}')
            return float('inf')  # No valid path found
            
    def move_player(self, target_pos):
        player1.get_valid_moves()
        if self.done:
            return self.get_state(), 0,0, self.done
        if target_pos not in self.valid_moves:
            return self.get_state(), 0,0, self.done

        #print(f'==================valid target position:{target_pos}')
        row, col = self.player_pos
        target_row, target_col = target_pos
        if self.player_pos==(startRow,startCol):
            for n in self.initNeighbors:
                if n in self.valid_moves:
                    self.valid_moves.remove(n)

        def is_adjacent(pos1, pos2):
            row, col = pos1
            if col%2:#odd col
                neighbors = [
                (row - 1, col), (row, col + 1), (row + 1, col + 1), (row + 1, col),
                (row + 1, col - 1), (row, col - 1)  # Add diagonal connections for hex grids
                ]
            else:
                neighbors = [
                (row - 1, col), (row - 1, col + 1), (row, col + 1), (row + 1, col),
                (row, col - 1), (row - 1, col - 1)  # Add diagonal connections for hex grids
                ]
            return pos2 in neighbors

        
        if is_adjacent(self.player_pos, target_pos):  # If target hexagon is adjacent
            #print(f'==================move to target position:{target_pos}')
            self.player_pos = target_pos
            self.path.append(target_pos)
            self.pathType.append('m')
            hexagon = self.getLandInfo(target_pos)
            self.food -= hexagon.Food
            self.total_award += hexagon.Award
            hexagon.Occupied = True
            #print(f'remove{target_pos}')
            if target_pos in self.valid_moves:
                self.valid_moves.remove(target_pos)
            if self.food <= 0:
                self.done = True
            return self.get_state(), hexagon.Award,hexagon.Food, self.done
        else:  # If target hexagon is connected through occupied hexagons
            #print(f'==================fly to target position:{target_pos}')
            n = self.find_min_occupied_path(self.player_pos, target_pos)
            #print(f'steps={n}')
            if n == float('inf'):  # No valid path found
                return self.get_state(), 0, self.done
            self.player_pos = target_pos
            self.path.append(target_pos)
            self.pathType.append('j')
            hexagon = self.getLandInfo(target_pos)
            additional_food_cost = 10 * n
            self.food -= (hexagon.Food + additional_food_cost)
            self.total_award += hexagon.Award
            hexagon.Occupied = True
            #print(f'remove{target_pos}')
            if target_pos in self.valid_moves:
                self.valid_moves.remove(target_pos)
            if self.food <= 0:
                self.done = True
            return self.get_state(), hexagon.Award,hexagon.Food + additional_food_cost, self.done
    
    def reset(self,start_pos=(startRow, startCol)):
        self.player_pos = (startRow, startCol)
        self.food = initFood
        self.total_award = 0
        self.done = False
        self.path = [start_pos]
        self.pathType = ['m']
        return self.get_state()
     
def plot_hexagon_grid(marchMap, hex_size, startRow, startCol, path, pathType):
    rows = len(marchMap)
    cols = len(marchMap[0])
    width = cols * 1.5 * hex_size
    height = rows * np.sqrt(3) * hex_size

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect(0.618)

    for row in range(rows):
        for col in range(cols):
            x_offset = col * 1.5 * hex_size
            y_offset = row * np.sqrt(3) * hex_size
            if col % 2 == 1:
                y_offset += np.sqrt(3) / 2 * hex_size
            
            hexagon = marchMap[row][col]
            edge = hexagon.Edge
            face = hexagon.Face
            pattern = hexagon.Pattern
            label = hexagon.Label
            name = hexagon.Name
            
            if label=="tower" or label=="bigGua":
                lw=3
            else:
                lw=1
            #print(f"label = {label}, lw={lw}")
            # Highlight the start position
            if row == rows - startRow and col == startCol - 1:
                edge = 'k'
                face = '#920000'

            hexagon_patch = RegularPolygon((x_offset, y_offset), numVertices=6, radius=hex_size, orientation=np.radians(30), edgecolor=edge, facecolor=face, hatch=pattern,linewidth=lw)
            ax.add_patch(hexagon_patch)
            #ax.text(x_offset, y_offset, f'({rows - row},{col + 1})', ha='center', va='center', fontsize=9) #coordinates label
            # ax.text(x_offset, y_offset, label, fontsize=20)
    # Draw the path
    adjusted_path = [(rows - row, col - 1) for row, col in path]
    for i in range(len(adjusted_path) - 1):
        start_pos = adjusted_path[i]
        end_pos = adjusted_path[i + 1]
        start_x = start_pos[1] * 1.5 * hex_size
        start_y = start_pos[0] * np.sqrt(3) * hex_size
        if start_pos[1] % 2 == 1:
            start_y += np.sqrt(3) / 2 * hex_size
        end_x = end_pos[1] * 1.5 * hex_size
        end_y = end_pos[0] * np.sqrt(3) * hex_size
        if end_pos[1] % 2 == 1:
            end_y += np.sqrt(3) / 2 * hex_size
        # Check pathType and set line style
        if pathType[i+1] == 'j':
            ax.plot([start_x, end_x], [start_y, end_y], 'g--', lw=2)  # Dashed line
        else:
            ax.plot([start_x, end_x], [start_y, end_y], 'k-', lw=2)  # Solid line
    ax.set_xlim(-hex_size, width + hex_size)
    ax.set_ylim(-hex_size, height + hex_size)
    plt.axis('off')
    plt.show()
    
    

        
# Generate a 60 x 60 hexagon grid with each hexagon having a size of 1
marchMap = hexagon_grid(len(map), len(map[0]), 5,startRow,startCol)

# player1 = Player(marchMap, start_pos=(startRow, startCol), food=initFood)
# initial_state = player1.get_state()
# # print(f"Initial State: {initial_state}")
# moves = [(9, 1), (10, 2), (9, 2), (11, 2), (9, 3), (10, 3), (11, 3), (8, 1), (8, 2), (8, 3), (10, 4), (8, 4), (7, 2)]
# for m in moves:
    # player1.move_player(m)
    # print(player1.get_state())
  
# plot_hexagon_grid(marchMap, hex_size=5, startRow=9, startCol=1,path=player1.path,pathType=player1.pathType)
# player1.move_player((10,2))
# player1.move_player((10,3))
# player1.move_player((9,3))
# player1.move_player((9,2))
# print(player1.get_state())
# player1.move_player((11,4))
# print(player1.get_state())
# player1.move_player((8,1))
# print(player1.get_state())
# player1.move_player((11,5))
# print(player1.get_state())
# player1.move_player((10,4))
# print(player1.get_state())
#plot_hexagon_grid(marchMap, hex_size=5, startRow=9, startCol=1,path=player1.path)

#============================================================
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, valid_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        else:            
            q_values = [self.get_q_value(state, action) for action in valid_moves]
            max_q = max(q_values)
            return valid_moves[q_values.index(max_q)]

    def learn(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state, action)] = new_q
        
        
    
# Generate a 60 x 60 hexagon grid with each hexagon having a size of 1
marchMap = hexagon_grid(len(map), len(map[0]), 5,startRow,startCol)

# Update the reward function
def calculate_reward(hexagon):
    base_reward = hexagon.Award
    if hexagon.Name in ["Merc", "Tent"]:
        
        base_reward *= 15  # Increase reward for "Merc" and "Tent"
    reward_to_food_ratio = base_reward / hexagon.Food if hexagon.Food > 0 else base_reward
    return reward_to_food_ratio

# player1 = Player(marchMap, start_pos=(startRow, startCol), food=800)
# initial_state = player1.get_state()
# print(f"Initial State: {initial_state}")

# Define the actions


# Create the Q-learning agent
#agent = QLearningAgent(actions)
player1 = Player(marchMap, start_pos=(startRow, startCol), food=initFood)
# Simulate episodes
actions = player1.get_valid_moves()
num_episodes = 1
BestPath = []
BestPathType = []
max_reward = float('-inf')
final_food = float('-inf')
agent = QLearningAgent(actions)  
for episode in range(num_episodes):
    resetMap(marchMap)
    state = player1.reset()
    total_reward = 0

    while True:
        #print(player1.player_pos)
        valid_moves = player1.get_valid_moves()
        if not valid_moves:
            #print("No valid moves available")
            break
        action = agent.choose_action(state, valid_moves)
        #print(f"Predicted Action: {action}")
        next_state, reward,food, done = player1.move_player(action)
        adjusted_reward = calculate_reward(player1.getLandInfo(action))
        agent.learn(state, action, reward/food, next_state)
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}: Total Reward: {total_reward} Food Left: {player1.food}")
    if total_reward > max_reward:
        final_food = player1.food
        max_reward = total_reward
        BestPath = player1.path
        BestPathType = player1.pathType
    elif total_reward == max_reward and player1.food > final_food:
        final_food = player1.food
        max_reward = total_reward
        BestPath = player1.path
        BestPathType = player1.pathType
print(f"The episode with the highest reward has a total reward of {max_reward} with {final_food} food left, and the path is {BestPath}.")



#print(player1.path)

#plot_hexagon_grid(marchMap, hex_size=5, startRow=startRow, startCol=startCol,path=BestPath,pathType=BestPathType)
marchMap = hexagon_grid(len(map), len(map[0]), 5, startRow, startCol)

def interactive_map(marchMap, hex_size, startRow, startCol):
    rows = len(marchMap)
    cols = len(marchMap[0])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')

    centers = {}      # (r,c) -> (x,y)
    patches = {}      # (r,c) -> patch
    for r in range(rows):
        for c in range(cols):
            x_offset = c * 1.5 * hex_size
            y_offset = r * math.sqrt(3) * hex_size
            if c % 2 == 1:
                y_offset += math.sqrt(3) / 2 * hex_size
            hexagon = marchMap[r][c]
            p = RegularPolygon((x_offset, y_offset), numVertices=6, radius=hex_size, orientation=np.radians(30),
                               edgecolor=hexagon.Edge, facecolor=hexagon.Face, hatch=hexagon.Pattern, linewidth=1)
            ax.add_patch(p)
            centers[(r, c)] = (x_offset, y_offset)
            patches[(r, c)] = p

    # helper: odd-q offset <-> axial (q,r)
    def offset_to_axial_oddq(row, col):
        q = col
        r = row - (col - (col & 1)) // 2
        return (q, r)

    def axial_to_offset_oddq(q, r):
        col = q
        row = r + (q - (q & 1)) // 2
        return (row, col)

    # cube helpers
    def axial_to_cube(q, r):
        x = q
        z = r
        y = -x - z
        return (x, y, z)

    def cube_to_axial(x, y, z):
        return (x, z)

    def cube_round(x, y, z):
        rx = round(x)
        ry = round(y)
        rz = round(z)
        x_diff = abs(rx - x)
        y_diff = abs(ry - y)
        z_diff = abs(rz - z)
        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry
        return (rx, ry, rz)

    def lerp(a, b, t):
        return a + (b - a) * t

    def cube_lerp(a, b, t):
        return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))

    def hex_linedraw(a_offset, b_offset):
        # a_offset, b_offset are (row,col) in odd-q offset coords
        aq, ar = offset_to_axial_oddq(a_offset[0], a_offset[1])
        bq, br = offset_to_axial_oddq(b_offset[0], b_offset[1])
        a_cube = axial_to_cube(aq, ar)
        b_cube = axial_to_cube(bq, br)
        # distance
        dist = max(abs(a_cube[0]-b_cube[0]), abs(a_cube[1]-b_cube[1]), abs(a_cube[2]-b_cube[2]))
        if dist == 0:
            return [a_offset]
        results = []
        for i in range(dist + 1):
            t = 0 if dist == 0 else i / dist
            c = cube_lerp(a_cube, b_cube, t)
            rc = cube_round(c[0], c[1], c[2])
            aq2, ar2 = cube_to_axial(rc[0], rc[1], rc[2])
            off = axial_to_offset_oddq(aq2, ar2)
            results.append(off)
        return results

    # stats text
    text_len = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top')
    text_food = ax.text(0.01, 0.95, '', transform=ax.transAxes, va='top')
    text_score = ax.text(0.01, 0.91, '', transform=ax.transAxes, va='top')

    path = []  # list of (r,c)
    lines = []
    cur_food = initFood
    cur_score = 0

    def redraw():
        # remove old lines
        nonlocal lines
        for ln in lines:
            try:
                ln.remove()
            except Exception:
                pass
        lines = []
        # draw path lines between centers (now path already contains every traversed tile center)
        if len(path) >= 2:
            for i in range(len(path) - 1):
                a = centers[path[i]]
                b = centers[path[i+1]]
                ln, = ax.plot([a[0], b[0]], [a[1], b[1]], 'k-', lw=2)
                lines.append(ln)
        # highlight patches
        for k, p in patches.items():
            if k in path:
                p.set_edgecolor('gold')
                p.set_linewidth(3)
            else:
                p.set_edgecolor(marchMap[k[0]][k[1]].Edge)
                p.set_linewidth(1)
        # update stats
        text_len.set_text(f'Path length: {len(path)}')
        text_food.set_text(f'Food: {cur_food}')
        text_score.set_text(f'Score: {cur_score}')
        fig.canvas.draw_idle()

    def neighbors_oddq(pos):
        r, c = pos
        # neighbors in offset coords for odd-q
        if c % 2 == 1:
            nbrs = [(r-1,c), (r,c+1), (r+1,c+1), (r+1,c), (r+1,c-1), (r,c-1)]
        else:
            nbrs = [(r-1,c), (r-1,c+1), (r,c+1), (r+1,c), (r,c-1), (r-1,c-1)]
        # keep in bounds
        res = []
        for nr, nc in nbrs:
            if 0 <= nr < rows and 0 <= nc < cols:
                # obstacle check: valid and not name 'empty'
                hexagon = marchMap[nr][nc]
                if hexagon.Valid:
                    res.append((nr,nc))
        return res

    def bfs_path(start, goal):
        # simple BFS returning list of offset coords from start to goal (inclusive), avoiding invalid tiles
        if start == goal:
            return [start]
        q = deque([start])
        came = {start: None}
        while q:
            cur = q.popleft()
            for nb in neighbors_oddq(cur):
                if nb not in came:
                    came[nb] = cur
                    if nb == goal:
                        # reconstruct
                        path = [goal]
                        cur2 = cur
                        while cur2 is not None:
                            path.append(cur2)
                            cur2 = came[cur2]
                        path.reverse()
                        return path
                    q.append(nb)
        return None

    # on_click event handler
    def on_click(event):
        nonlocal cur_food, cur_score, path
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        # find nearest center within threshold
        sel = None
        thr = hex_size * 1.1
        for k, (cx, cy) in centers.items():
            if (x - cx)**2 + (y - cy)**2 <= thr**2:
                sel = k
                break
        if sel is None:
            return
        # obstacle check: cannot select an empty/invalid tile as target
        if not marchMap[sel[0]][sel[1]].Valid:
            print('目标为障碍，无法选择')
            return
        # backtrack if clicking an earlier node
        if sel in path:
            idx = path.index(sel)
            path[:] = path[:idx+1]
            # recompute resources
            cur_food = initFood
            cur_score = 0
            for (rr, cc) in path:
                h = marchMap[rr][cc]
                cur_food -= h.Food
                cur_score += h.Award
            redraw()
            return
        # otherwise append along hex line from last -> sel
        if not path:
            # first pick must be sel (start anywhere)
            h = marchMap[sel[0]][sel[1]]
            if cur_food - h.Food <= 0:
                print('Not enough food to move')
                return
            cur_food -= h.Food
            cur_score += h.Award
            path.append(sel)
            redraw()
            return
        last = path[-1]
        route = hex_linedraw(last, sel)
        # if route contains invalid tiles, try BFS
        invalid_present = any(not marchMap[t[0]][t[1]].Valid for t in route)
        if invalid_present:
            path_via = bfs_path(last, sel)
            if path_via is None:
                print('无法找到不经过障碍的路径')
                return
            route = path_via
        # route includes last at index 0, target last element
        # if any intermediate tile already in path -> trim to that tile instead
        for t in route[1:]:
            if t in path:
                idx = path.index(t)
                path[:] = path[:idx+1]
                # recompute resources
                cur_food = initFood
                cur_score = 0
                for (rr, cc) in path:
                    h = marchMap[rr][cc]
                    cur_food -= h.Food
                    cur_score += h.Award
                redraw()
                return
        # compute total cost for intermediate tiles (exclude current last)
        cost = 0
        for t in route[1:]:
            h = marchMap[t[0]][t[1]]
            cost += h.Food
        if cur_food - cost <= 0:
            print('Not enough food to move along route')
            return
        # apply and append intermediate tiles
        for t in route[1:]:
            h = marchMap[t[0]][t[1]]
            cur_food -= h.Food
            cur_score += h.Award
            path.append(t)
        redraw()

    def on_key(event):
        nonlocal path, cur_food, cur_score
        if event.key in ('r', 'R'):
            path = []
            cur_food = initFood
            cur_score = 0
            redraw()
        elif event.key in ('u', 'U'):
            if path:
                path.pop()
                # recompute
                cur_food = initFood
                cur_score = 0
                for rr, cc in path:
                    h = marchMap[rr][cc]
                    cur_food -= h.Food
                    cur_score += h.Award
                redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    def on_scroll(event):
        # zoom in/out around mouse position
        if event.inaxes != ax:
            return
        # scale factor per scroll step
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return
        xdata = event.xdata
        ydata = event.ydata
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_width = cur_xlim[1] - cur_xlim[0]
        cur_height = cur_ylim[1] - cur_ylim[0]
        new_width = cur_width * scale_factor
        new_height = cur_height * scale_factor
        relx = (xdata - cur_xlim[0]) / cur_width
        rely = (ydata - cur_ylim[0]) / cur_height
        ax.set_xlim(xdata - relx * new_width, xdata + (1 - relx) * new_width)
        ax.set_ylim(ydata - rely * new_height, ydata + (1 - rely) * new_height)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    ax.set_xlim(-hex_size, cols * 1.5 * hex_size + hex_size)
    ax.set_ylim(-hex_size, rows * math.sqrt(3) * hex_size + hex_size)
    plt.axis('off')
    plt.show()

    interactive_map(marchMap, hex_size=5, startRow=startRow, startCol=startCol)
