import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Constants for trap and reward effects
TRAP_COST = 2  # Cost multiplier for traps
REWARD_COST = 0.5  # Step cost for rewards
NORMAL_COST = 1  # Default step cost

# Energy consumption constants
NORMAL_ENERGY_CONSUMPTION = 1
TRAP_ENERGY_CONSUMPTION = 2
REWARD_ENERGY_CONSUMPTION = 0.5

# Define the grid (rows, columns)
grid_rows = 6
grid_cols = 10

# Define trap/reward effects
trap_effects = {
    (3, 8): NORMAL_COST,
    (4, 1): TRAP_COST,
    (1, 2): TRAP_COST,
    (2, 5): NORMAL_COST,
    (4, 6): NORMAL_COST,
    (4, 3): 1000000,
}

reward_effects = {
    (0, 5): REWARD_COST,
    (3, 7): REWARD_COST,
    (5, 4): NORMAL_COST,
}

# New energy effect dictionaries
energy_trap_effects = {
    (3, 8): TRAP_ENERGY_CONSUMPTION,
}

energy_reward_effects = {
    (2, 1): REWARD_ENERGY_CONSUMPTION,
    (5, 4): REWARD_ENERGY_CONSUMPTION,
}

# Treasures' Coordinates
treasures = [(1, 3), (4, 4), (2, 7), (2, 9)]

# Obstacles' Coordinates
obstacles = [(2, 0), (3, 2), (2, 3), (1, 4), (3, 4), (1, 6), (2, 6), (1, 7), (4, 8)]

def hexagon(x_center, y_center, size):
    """Generate the vertices of a hexagon centered at (x_center, y_center) with a given size."""
    angles = np.linspace(0, 2 * np.pi, 7)
    x_hexagon = x_center + size * np.cos(angles)
    y_hexagon = y_center + size * np.sin(angles)
    return x_hexagon, y_hexagon

def plot_honeycomb_grid(rows, cols, size=1, marked_coordinates=None, path_coordinates=None, current_node=None):
    plt.figure(figsize=(cols, rows))
    plt.title("Virtual World Grid", fontsize=30, pad=35)
    
    for row in range(rows):
        for col in range(cols):
            x_offset = col * 1.5 * size
            y_offset = row * np.sqrt(3) * size + (col % 2) * (np.sqrt(3) / 2 * size)
            x_hexagon, y_hexagon = hexagon(x_offset, y_offset, size)

            plt.fill(x_hexagon, y_hexagon, color='white', edgecolor='black')

            if marked_coordinates is not None and (row, col) in marked_coordinates:
                if (row, col) in [(4, 1), (4, 3), (4, 6), (3, 8), (2, 5), (1, 2)]:
                    plt.fill(x_hexagon, y_hexagon, color='plum', edgecolor='black')
                if (row, col) in [(2, 1), (5, 4), (0, 5), (3, 7)]:
                    plt.fill(x_hexagon, y_hexagon, color='lightseagreen', edgecolor='black')
                if (row, col) in [(1, 3), (4, 4), (2, 7), (2, 9)]:
                    plt.fill(x_hexagon, y_hexagon, color='gold', edgecolor='black')
                if (row, col) in [(2, 0), (3, 2), (2, 3), (1, 4), (3, 4), (1, 6), (2, 6), (1, 7), (4, 8)]:
                    plt.fill(x_hexagon, y_hexagon, color='gray', edgecolor='black')

            if path_coordinates is not None and (row, col) in path_coordinates:
                plt.fill(x_hexagon, y_hexagon, color='yellow', edgecolor='black')
                
            if current_node is not None and (row, col) == current_node:
                plt.fill(x_hexagon, y_hexagon, color='purple', edgecolor='black')

    symbol_coordinates = {
        '⊖': [(3, 8)],
        '⊕': [(4, 1), (1, 2)],
        '⊗': [(2, 5), (4, 6)],
        '⊘': [(4, 3)],
        '⊞': [(2, 1), (5, 4)],
        '⊠': [(0, 5), (3, 7)],
    }

    for symbol, coords in symbol_coordinates.items():
        for (row, col) in coords:
            x_offset = col * 1.5 * size
            y_offset = row * np.sqrt(3) * size + (col % 2) * (np.sqrt(3) / 2 * size)
            plt.text(x_offset, y_offset, symbol, ha='center', va='center', fontsize=25, color='black')

    plt.gca().set_aspect('equal')
    plt.axis('off')
    
    # Add custom legend
    trap_patch = mpatches.Patch(color='plum', label='Trap')
    reward_patch = mpatches.Patch(color='lightseagreen', label='Reward')
    treasure_patch = mpatches.Patch(color='gold', label='Treasure')
    obstacle_patch = mpatches.Patch(color='gray', label='Obstacle')
    path_patch = mpatches.Patch(color='yellow', label='Path')
    current_node_patch = mpatches.Patch(color='purple', label='Current Node')

    plt.legend(handles=[trap_patch, reward_patch, treasure_patch, obstacle_patch, path_patch, current_node_patch], loc='upper right', bbox_to_anchor=(1.2, 1.0))
    # Add text for initial energy
    plt.figtext(0.5, 0.90, "Note: You are given 20 units of energy. Finish the game before your energy runs out.", ha="center", fontsize=12, color="maroon")

    plt.show()

# Define marked coordinates with priority for purple, then green, then gold
marked_coordinates = [
    (4, 1), (4, 3), (4, 6), (3, 8), (2, 5), (1, 2),  # Purple
    (2, 1), (5, 4), (0, 5), (3, 7),                  # Green
    (1, 3), (4, 4), (2, 7), (2, 9),                  # Gold
    (2, 0), (3, 2), (2, 3), (1, 4), (3, 4), (1, 6), (2, 6), (1, 7), (4, 8)  # Gray (Obstacles)
]

# Define the grid with costs and traps/rewards
grid = np.ones((grid_rows, grid_cols), dtype=np.float32) * NORMAL_COST

# Apply trap and reward effects to the grid
for (row, col), multiplier in trap_effects.items():
    grid[row, col] *= multiplier

for (row, col), divisor in reward_effects.items():
    grid[row, col] = REWARD_COST

start = (5, 0)
goal = (0, 9)

# Implementation of Uniform Cost Search (UCS)
def uniform_cost_search(grid, start, goal, treasures, obstacles):
    # Directions for hexagonal grid movement
    directions_odd = [(1, 0), (1, -1), (0, -1), (-1, 0), (1, 1), (0, 1)]
    directions_even = [(-1, -1), (0, -1), (-1, 0), (1, 0), (0, 1), (-1, 1)]

    # Priority queue to store (cost, current_node, collected_treasures)
    pq = [(0, start, set())]

    # Dictionary to keep track of visited nodes with their collected treasures
    visited = {}

    # Dictionary to keep track of the parent of each node to reconstruct the path
    parent = {}

    # Dictionary to keep track of the cost to reach each node with its collected treasures
    cost_so_far = {(start, frozenset()): 0}

    while pq:
        pq.sort()  # Sort the priority queue by cost
        current_cost, current_node, collected_treasures = pq.pop(0)
        
        # Trap 4: Check if the current node is (4, 3) to stop the game
        if current_node == (4, 3):
            print("Game stopped! You stepped on Trap 4.")
            goal = current_node  # Set the current node as the goal
            break

        # Check if all treasures have been collected
        if len(collected_treasures) == len(treasures):
            print("Yay! You have collected all the treasures 🎉🤩👏")
            goal = current_node  # Set the current node as the goal
            break

        # If the node has already been visited with the same collected treasures, skip it
        if (current_node, frozenset(collected_treasures)) in visited:
            continue

        # Mark the current node with its collected treasures as visited
        visited[(current_node, frozenset(collected_treasures))] = True

        # Determine the directions based on the column parity
        directions = directions_odd if current_node[1] % 2 != 0 else directions_even

        for direction in directions:
            # Define the neighbor based on the direction
            neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])
    
            # Trap 3: if current node is (2, 5), move to (3, 7)
            if current_node == (2, 5):
                neighbor = (3, 7)
                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                    new_cost = current_cost + grid[current_node[0]][current_node[1]] + grid[neighbor[0]][neighbor[1]]
                else:
                    continue  # Skip if the neighbor is out of bounds
            else:
                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                    new_cost = current_cost + grid[neighbor[0]][neighbor[1]]
                else:
                    continue  # Skip if the neighbor is out of bounds

            # Skip the neighbor if it's an obstacle
            if neighbor in obstacles:
                continue

            # Update the set of collected treasures
            new_collected_treasures = collected_treasures.copy()
            if neighbor in treasures:
                new_collected_treasures.add(neighbor)

            # Update the cost and path if it's the first time visiting or found a cheaper path
            if (neighbor, frozenset(new_collected_treasures)) not in cost_so_far or new_cost < cost_so_far[(neighbor, frozenset(new_collected_treasures))]:
                cost_so_far[(neighbor, frozenset(new_collected_treasures))] = new_cost
                pq.append((new_cost, neighbor, new_collected_treasures))
                parent[(neighbor, frozenset(new_collected_treasures))] = (current_node, frozenset(collected_treasures))

    # If goal state was never reached
    if (goal, frozenset(treasures)) not in parent:
        return None, None

    # Reconstruct the path and step-by-step costs
    path = []
    step_by_step_costs = []
    state = (goal, frozenset(treasures))
    while state[0] != start:
        path.append(state[0])
        step_by_step_costs.append(cost_so_far[state])
        state = parent[state]
    path.append(start)
    path.reverse()
    step_by_step_costs.reverse()
    return path, step_by_step_costs

# Find the optimal path using uniform cost search
path, step_by_step_costs = uniform_cost_search(grid, start, goal, set(treasures), set(obstacles))

def calculate_energy_consumption(path):
    energy = 20  # Starting energy
    energy_consumption = [energy]  # Initialize with starting energy
    
    # Iterate through the path, skipping the first element (starting point)
    for i in range(1, len(path)):
        step = path[i]
        if step in energy_trap_effects:
            energy -= TRAP_ENERGY_CONSUMPTION
        elif step in energy_reward_effects:
            energy -= REWARD_ENERGY_CONSUMPTION
        else:
            energy -= NORMAL_ENERGY_CONSUMPTION
        energy_consumption.append(energy)
        if energy < 0:
            return energy_consumption, True  # Return energy consumption and a flag indicating energy depletion
    return energy_consumption, False


if path:
    print("Path found:", path)
    total_cost = step_by_step_costs[-1]
    
    print("Step-by-step costs:", step_by_step_costs)
    print("Total path cost:", total_cost)

    # Calculate energy consumption and check for depletion
    energy_consumption, energy_depleted = calculate_energy_consumption(path)
    print("Energy consumption step-by-step:", energy_consumption)
    
    if energy_depleted:
        print("Energy already been used up‼️ 😔⚡🪫 ")
    else:
        # Print remaining energy
        remaining_energy = energy_consumption[-1]  # Last element in energy_consumption list
        print("Remaining energy:", remaining_energy)

        # Visualize the path step by step
        for i in range(len(path)):
            plot_honeycomb_grid(grid_rows, grid_cols, size=1, marked_coordinates=marked_coordinates, path_coordinates=path[:i+1], current_node=path[i])
else:
    print("No path found.")
