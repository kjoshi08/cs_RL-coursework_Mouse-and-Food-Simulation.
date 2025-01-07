import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Initialize PyGame and other configurations
pygame.init()

GRID_SIZE = 100
CELL_SIZE = 5
ENERGY_FULL = 200
SCENT_RANGE = 3
NUM_FOOD = 50    # Increase food quantity for more opportunities

# Set up colors and display
WHITE = (255, 255, 255)
RED = (255, 0, 0)        # Color for the mouse
BLUE = (0, 0, 255)      # Color for the food
BLACK = (0, 0, 0)
screen = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
pygame.display.set_caption("Goal vs Hole Game")

# Font for displaying text
font = pygame.font.Font(None, 36)

class Mouse:
        def __init__(self):
                self.x, self.y = 0, 0    # Mouse starts at top-left corner
                self.energy = ENERGY_FULL
                self.food_collected = 0

        def reset(self):
                self.x, self.y = 0, 0
                self.energy = ENERGY_FULL
                self.food_collected = 0

        def move(self, direction, terrain_cost):
                if direction == 0 and self.y > 0:              # North
                        self.y -= 1
                elif direction == 1 and self.y < GRID_SIZE - 1:    # South
                        self.y += 1
                elif direction == 2 and self.x < GRID_SIZE - 1:    # East
                        self.x += 1
                elif direction == 3 and self.x > 0:          # West
                        self.x -= 1
                self.energy -= terrain_cost[self.x][self.y]    # Decrease energy based on terrain cost

        def sense_food(self, grid):
                # Generates the 3x3 scent matrix with stacked values for overlapping food
                scent_matrix = np.zeros((SCENT_RANGE, SCENT_RANGE))
                for i in range(-1, 2):
                        for j in range(-1, 2):
                                if 0 <= self.x + i < GRID_SIZE and 0 <= self.y + j < GRID_SIZE:
                                        scent_matrix[i + 1][j + 1] += grid[self.x + i][self.y + j]
                return scent_matrix.flatten()[[0, 1, 2, 3, 5, 6, 7, 8]]    # Ignore the center

class Environment:
        def __init__(self):
                self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
                self.mouse = Mouse()
                self.terrain_cost = np.ones((GRID_SIZE, GRID_SIZE))    # Uniform energy cost terrain
                self.spawn_food()

        def spawn_food(self):
                self.grid = np.zeros((GRID_SIZE, GRID_SIZE))    # Reset grid
                for _ in range(NUM_FOOD):
                        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                        self.grid[x][y] = 1    # Place food at random locations

        def reset(self):
                self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
                self.mouse.reset()
                self.terrain_cost = np.ones((GRID_SIZE, GRID_SIZE))    # Reset terrain costs
                self.spawn_food()

        def check_food_collection(self):
                # If mouse is on a food tile, collect the food
                if self.grid[self.mouse.x][self.mouse.y] == 1:
                        self.grid[self.mouse.x][self.mouse.y] = 0    # Remove food from the grid
                        self.mouse.food_collected += 1    # Increase food collected count
                        self.mouse.energy = ENERGY_FULL    # Replenish energy upon finding food

        def render(self, game_over=False):
                screen.fill(WHITE)
                
                # Draw the grid and food
                for i in range(GRID_SIZE):
                        for j in range(GRID_SIZE):
                                if self.grid[i][j] == 1:    # Food tile
                                        pygame.draw.rect(screen, BLUE, pygame.Rect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                                else:
                                        pygame.draw.rect(screen, WHITE, pygame.Rect(i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                
                # Draw the borders
                for i in range(GRID_SIZE + 1):    # Draw vertical borders
                        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE * CELL_SIZE))
                for j in range(GRID_SIZE + 1):    # Draw horizontal borders
                        pygame.draw.line(screen, BLACK, (0, j * CELL_SIZE), (GRID_SIZE * CELL_SIZE, j * CELL_SIZE))

                # Draw the mouse icon
                pygame.draw.rect(screen, RED, pygame.Rect(self.mouse.x * CELL_SIZE, self.mouse.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

                # If the game is over, display a "Game Over" message with food count
                if game_over:
                        text = font.render(f"Game Over! Food Eaten: {self.mouse.food_collected}/{NUM_FOOD}", True, BLACK)
                        screen.blit(text, (20, 20))
                
                pygame.display.flip()

class RLModel(nn.Module):
        def __init__(self):
                super(RLModel, self).__init__()
                self.fc1 = nn.Linear(8, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 4)    # Four outputs: N, S, E, W

        def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return torch.softmax(self.fc3(x), dim=-1)

# Define a reward function
def reward_function(state, previous_distance):
        mouse_pos = np.array([state.mouse.x, state.mouse.y])
        food_positions = np.argwhere(state.grid == 1)

        if len(food_positions) == 0:
                return 100, previous_distance    # High reward when all food is eaten

        # Calculate distance to the nearest food
        distances = np.linalg.norm(food_positions - mouse_pos, axis=1)
        min_distance = np.min(distances)

        reward = -1    # Penalize each step to encourage faster completion

        # Reward for moving closer to food
        if min_distance < previous_distance:
                reward += 10

        # Extra reward for eating food
        if state.mouse.food_collected > 0:
                reward += 20

        return reward, min_distance

# Define a function to handle quitting
def handle_quit():
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

# Game loop with RL model
env = Environment()
model = RLModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
memory = deque(maxlen=1000)
batch_size = 64
gamma = 0.99    # Discount factor for future rewards
num_episodes = 10
epsilon = 0.1    # Exploration rate

game_running = True
while game_running:
        for episode in range(num_episodes):
                env.reset()
                total_reward = 0
                previous_distance = float('inf')    # Initialize to a large distance
                game_over = False

                for step in range(ENERGY_FULL):
                        handle_quit()    # Handle quit events

                        # Sense and get action
                        input_sense = torch.tensor(env.mouse.sense_food(env.grid), dtype=torch.float32).unsqueeze(0)

                        if random.random() < epsilon:    # Epsilon-greedy exploration
                                action = random.randint(0, 3)
                        else:
                                action_probs = model(input_sense)
                                action = torch.argmax(action_probs).item()

                        # Move the mouse
                        env.mouse.move(action, env.terrain_cost)
                        env.check_food_collection()

                        # Calculate reward
                        reward, previous_distance = reward_function(env, previous_distance)
                        total_reward += reward

                        # Store experience in memory
                        memory.append((input_sense, action, reward))

                        # Render the environment
                        if env.mouse.energy <= 0 or env.mouse.food_collected == NUM_FOOD:
                                game_over = True
                                env.render(game_over=True)
                                break
                        else:
                                env.render()

                # Experience replay
                if len(memory) >= batch_size:
                        batch = random.sample(memory, batch_size)
                        for state, action, reward in batch:
                                target = reward + gamma * torch.max(model(state))
                                predicted = model(state)[0, action]
                                loss = nn.MSELoss()(predicted, target)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                print(f"Episode {episode + 1}: Food Eaten: {env.mouse.food_collected}/{NUM_FOOD}, Total Reward: {total_reward}, Game Over")

        handle_quit()    # Allow user to quit manually
