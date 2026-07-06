import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
import csv
import json
import random

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
                y_offset += np.sqrt(3)/2 * hex_size
            
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

plot_hexagon_grid(marchMap, hex_size=5, startRow=startRow, startCol=startCol,path=BestPath,pathType=BestPathType)
