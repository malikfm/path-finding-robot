import math
import random
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

# Define the environment and GA parameters
# tkinter canvas coordinate starts from top-left
start_point = (1.0, 10.0)
goal_point = (10.0, 1.0)
chromosome_length = 10
population_size = 40
elitism_rate = 0.05
mutation_probability = 0.5
mutation_range = 1.0

# obstacles = [(2.5, 3.0), (4.5, 4.0), (3.0, 5.0)]  # Example obstacles
obstacles = []
obstacles_type = []
for i in range(random.randint(5, 10)):
    obstacle = (float(random.randint(2, 9)), float(random.randint(2, 9)))

    # Prevent creating obstacle in the same coordinate.
    while obstacle in obstacles:
        obstacle = (float(random.randint(2, 9)), float(random.randint(2, 9)))
    
    obstacles.append(obstacle)
    obstacles_type.append(random.choice([1, 2, 3]))

# Helper functions for Genetic Algorithm
def initialize_population():
    population = []
    for _ in range(population_size):
        chromosome = [start_point]  # Start point
        for _ in range(chromosome_length - 2):
            chromosome.append((random.uniform(0, 10), random.uniform(0, 10)))
        chromosome.append(goal_point)  # Goal point
        population.append(pad_chromosome(chromosome))
    return population

def pad_chromosome(chromosome):
    while len(chromosome) < chromosome_length:
        chromosome.insert(-1, chromosome[-2])
    return chromosome

def fitness(chromosome):
    total_cost = 0
    collisions = 0
    path_distance = 0
    smoothness = 0
    
    for i in range(len(chromosome) - 1):
        point1 = chromosome[i]
        point2 = chromosome[i + 1]
        path_distance += np.linalg.norm(np.array(point2) - np.array(point1))
        smoothness += abs(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
        if check_collision(point1, point2):
            collisions += 1
    
    min_distance_to_obstacle = min([distance_to_obstacle(chromosome) for obstacle in obstacles])
    total_cost = (collisions * 1000) + min_distance_to_obstacle + smoothness + path_distance
    return 1 / total_cost

def point_line_distance(point, line_start, line_end):
    """Calculate the perpendicular distance from a point to a line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Vector AB
    dx, dy = x2 - x1, y2 - y1

    # Project point P onto line segment AB, but limit it to the segment
    if dx == 0 and dy == 0:
        # The segment is actually a point.
        return np.linalg.norm(np.array([px - x1, py - y1]))

    # Project point onto line (not limited to the segment)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # Limit t to [0, 1] to restrict to segment
    t = max(0, min(1, t))

    # Find the closest point on the segment
    nearest_x, nearest_y = x1 + t * dx, y1 + t * dy

    # Return the distance from P to the nearest point on the segment
    return np.linalg.norm(np.array([px - nearest_x, py - nearest_y]))

def check_collision(p1, p2, threshold=0.5):
    """Return True if the line segment from p1 to p2 intersects any obstacles."""
    for obs in obstacles:
        # Check if the perpendicular distance to the line is below the threshold
        if point_line_distance(obs, p1, p2) < threshold:
            return True
    return False

def distance_to_obstacle(chromosome):
    min_dist = float('inf')
    for point in chromosome:
        for obstacle in obstacles:
            dist = np.linalg.norm(np.array(point) - np.array(obstacle))
            if dist < min_dist:
                min_dist = dist
    return math.exp(-1 * min_dist)

def rank_selection(population):
    population.sort(key=lambda x: fitness(x), reverse=True)
    return population[:int(population_size * 0.5)]

def crossover(parent1, parent2):
    alpha = 0.5
    child = [start_point]
    for i in range(1, chromosome_length - 1):
        new_x = alpha * parent1[i][0] + (1 - alpha) * parent2[i][0]
        new_y = alpha * parent1[i][1] + (1 - alpha) * parent2[i][1]
        child.append((new_x, new_y))
    child.append(goal_point)
    return pad_chromosome(child)

def mutate(chromosome):
    for i in range(1, chromosome_length - 1):
        if random.random() < mutation_probability:
            new_x = chromosome[i][0] + random.uniform(-mutation_range, mutation_range)
            new_y = chromosome[i][1] + random.uniform(-mutation_range, mutation_range)
            chromosome[i] = (new_x, new_y)
    return chromosome

def elitism(population):
    elite_size = int(elitism_rate * population_size)
    return population[:elite_size]

# Visualize the Genetic Algorithm process with tkinter
class GeneticPathfinderUI:
    def __init__(self, generations=100):
        self.max_generations = generations
        self.generations = generations
        self.population = initialize_population()
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=1000, height=1000)
        self.canvas.pack()
        
        # Run GA and visualize
        self.best_fitness = 0
        self.best_fitness_each_gen = []
        self.best_solution = []
        self.update_generation()

    def update_generation(self):
        self.population = rank_selection(self.population)
        best_path = self.population[0]
        best_fitness = fitness(best_path)
        
        # Logging best fitness for each generation
        print(f"Generation {self.max_generations - self.generations + 1}: Best fitness = {best_fitness}")

        # Store the best solution and its fitness
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_solution = best_path

        self.draw_path(best_path)
        self.best_fitness_each_gen.append(self.best_fitness)
        
        new_population = elitism(self.population)
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(self.population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        self.population = new_population

        self.generations -= 1
        if self.generations > 0:
            self.root.after(100, self.update_generation)
        else:
            # Log the final best solution at the end
            print("Best solution found:", self.best_solution)
            self.plot_fitness()

    def plot_fitness(self):
        """Plot the best fitness per generation using matplotlib."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitness_each_gen, label="Best Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Best Fitness per Generation")
        plt.legend()
        plt.show()

    def draw_path(self, path):
        self.canvas.delete("all")
        self.draw_obstacles()
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            self.canvas.create_line(
                x1 * 70, y1 * 70, x2 * 70, y2 * 70, fill="blue", width=2
            )

        self.canvas.create_oval(
            start_point[0] * 70 - 5, start_point[1] * 70 - 5,
            start_point[0] * 70 + 5, start_point[1] * 70 + 5,
            fill="green"
        )
        self.canvas.create_oval(
            goal_point[0] * 70 - 5, goal_point[1] * 70 - 5,
            goal_point[0] * 70 + 5, goal_point[1] * 70 + 5,
            fill="red"
        )

    def draw_obstacles(self):
        for idx, obs in enumerate(obstacles):
            x, y = obs

            # 1 = rectangle, 2 = triangle, 3 = oval
            obs_type = random.choice([1, 2, 3])

            if obstacles_type[idx] == 1:
                self.canvas.create_rectangle(
                    x * 70 - 10, y * 70 - 10,
                    x * 70 + 10, y * 70 + 10,
                    fill="grey",
                    outline="grey",
                    width=15
                )
            elif obstacles_type[idx] == 2:
                self.canvas.create_oval(
                    x * 70 - 10, y * 70 - 10,
                    x * 70 + 10, y * 70 + 10,
                    fill="grey",
                    outline="grey",
                    width=15
                )
            else:
                self.canvas.create_polygon(
                    x * 70, y * 70 - 20,
                    x * 70 - 20, y * 70 + 20,
                    x * 70 + 20, y * 70 + 20,
                    fill="grey",
                    outline="grey"
                )

# Run the UI application
ui = GeneticPathfinderUI(generations=200)
ui.root.mainloop()
