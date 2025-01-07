
# Mouse and Food Simulation - Reinforcement Learning

## Project Overview

This project involves designing a **Reinforcement Learning (RL)** algorithm to guide a mouse in locating and consuming food within a simulated grid environment. The simulation uses **PyGame** for visualization and includes challenges like limited energy and sensory inputs.

A comparison with a non-RL arithmetic-based solution is encouraged.

## Features

- **Grid Environment**: A 100x100 grid with arbitrarily placed food items.
- **Mouse Sensory Input**: A 3x3 "smell" matrix indicating food proximity, excluding the mouse's current position.
- **Energy Management**: Energy depletes with movement and replenishes upon consuming food.
- **Movement**: Mouse moves in cardinal directions (N, S, E, W).

## Tasks

1. **Reward Function**:
   - Develop a function that rewards food discovery and penalizes inefficiency.
   - Use sparse or shaped rewards to guide the mouse's behavior.

2. **Model Development**:
   - Create a model that processes sensory inputs (8 values) and outputs probabilities for movement.
   - Handle reward backpropagation and decision updates.

3. **Optional Enhancements**:
   - Experiment with variables like reduced scent range or energy-cost terrain tiles.

## Setup and Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the simulation:
   ```bash
   python main.py
   ```

## Questions to Explore

- What constitutes an effective reward function?
- How do changes in variables (e.g., scent range, terrain cost) impact performance?
- How does the RL solution compare to a non-RL algorithm?
