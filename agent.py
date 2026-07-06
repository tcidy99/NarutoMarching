import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon
import csv
import json
import random
import marchGame
import marchGame
from marchGame import move_player, plot_hexagon_grid


file_path = 'map.csv' 
map = marchGame.read_csv_to_2d_vector(file_path)
map.reverse()
startRow = 9
startCol = 1
visited = set()
valid_moves = []
initNeighbors=[]
#Actions = [(r, c) for r in range(len(map)) for c in range(len(map[0]))]

#for row in map[:]:
#    print(row)
# Load the JSON file
with open('landInfo.json', 'r') as file:
    map_properties = json.load(file)

initFood=2000

    
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.5):
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
marchMap = marchGame.hexagon_grid(len(map), len(map[0]), 5,startRow,startCol)

# player1 = Player(marchMap, start_pos=(startRow, startCol), food=800)
# initial_state = player1.get_state()
# print(f"Initial State: {initial_state}")

# Define the actions
actions = [1, 2, 3, 4, 5, 6]

# Create the Q-learning agent
agent = QLearningAgent(actions)
player1 = marchGame.Player(marchMap, start_pos=(startRow, startCol), food=initFood)
# Simulate episodes
num_episodes = 1
BestPath = []
max_reward = float('-inf')
final_food = float('-inf')

for episode in range(num_episodes):
    marchGame.resetMap(marchMap)
    state = player1.reset()
    total_reward = 0

    while True:
        print(player1.player_pos)
        valid_moves = player1.get_valid_moves()
        if not valid_moves:
            #print("No valid moves available")
            break
        action = agent.choose_action(state, valid_moves)
        #print(f"Predicted Action: {action}")
        next_state, reward, done = player1.move_player(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}: Total Reward: {total_reward} Food Left: {player1.food}")
    if total_reward > max_reward:
        final_food = player1.food
        max_reward = total_reward
        BestPath = player1.path
    elif total_reward == max_reward and player1.food > final_food:
        final_food = player1.food
        max_reward = total_reward
        BestPath = player1.path
print(f"The episode with the highest reward has a total reward of {max_reward} with {final_food} food left, and the path is {BestPath}.")



#print(player1.path)

plot_hexagon_grid(marchMap, hex_size=5, startRow=9, startCol=1,path=BestPath)
