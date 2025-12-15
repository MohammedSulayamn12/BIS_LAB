import random
import numpy as np

# Define the problem (optimization of flight control parameters)
# Control parameters: pitch_angle, roll_angle, yaw_angle
# Target: Minimize deviation from desired control behavior (flight stability)

# Define bounds for the control parameters
PITCH_BOUNDS = (-15, 15)   # degrees
ROLL_BOUNDS = (-30, 30)    # degrees
YAW_BOUNDS = (-10, 10)     # degrees

# Define the fitness function (simple example: how close the control angles are to ideal angles)
def fitness(control_params):
    ideal_params = np.array([0, 0, 0])  # Ideal pitch, roll, and yaw for stability
    control_params = np.array(control_params)
    deviation = np.abs(ideal_params - control_params)  # Deviation from the ideal
    return np.sum(deviation)  # The lower, the better

# Initialize population (each individual has pitch, roll, and yaw control values)
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = [
            random.uniform(*PITCH_BOUNDS),
            random.uniform(*ROLL_BOUNDS),
            random.uniform(*YAW_BOUNDS)
        ]
        population.append(individual)
    return population

# Selection: Tournament selection method (randomly pick a few individuals, select the best)
def selection(population, fitness_values):
    selected = random.choices(
        population=population,
        weights=[1 / (f + 1e-6) for f in fitness_values],  # inverse fitness
        k=len(population) // 2
    )
    return selected

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation: Mutate an individual by slightly adjusting the values
def mutate(individual):
    mutation_rate = 0.1
    if random.random() < mutation_rate:
        mutation_index = random.randint(0, 2)
        mutation_value = random.uniform(-5, 5)  # small mutation in control values
        individual[mutation_index] += mutation_value
    # Ensure bounds are respected
    individual[0] = np.clip(individual[0], *PITCH_BOUNDS)
    individual[1] = np.clip(individual[1], *ROLL_BOUNDS)
    individual[2] = np.clip(individual[2], *YAW_BOUNDS)
    return individual

# Main Genetic Algorithm loop
def genetic_algorithm(pop_size=50, generations=100):
    population = initialize_population(pop_size)
    
    for generation in range(generations):
        # Evaluate fitness for each individual in the population
        fitness_values = [fitness(individual) for individual in population]
        
        # Select the best individuals for reproduction
        selected_individuals = selection(population, fitness_values)
        
        # If we have an odd number of selected individuals, add one more
        if len(selected_individuals) % 2 != 0:
            selected_individuals.append(random.choice(population))
        
        # Generate the next generation
        next_population = []
        for i in range(0, len(selected_individuals), 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[i + 1]
            child1, child2 = crossover(parent1, parent2)
            next_population.extend([mutate(child1), mutate(child2)])
        
        # Calculate fitness for the new population
        fitness_values_new_population = [fitness(individual) for individual in next_population]
        
        # Find the best individual in the new population
        best_fitness = min(fitness_values_new_population)
        best_individual = next_population[fitness_values_new_population.index(best_fitness)]
        
        # Replace old population with new population
        population = next_population

        # Output the best solution in each generation with integer output
        best_fitness_int = int(best_fitness)  # Convert fitness to integer
        best_individual_int = [int(param) for param in best_individual]  # Convert control params to integers
        
        print(f"Generation {generation}: Best Fitness = {best_fitness_int}, Best Params = {best_individual_int}")
    
    # Return the best solution found as integers
    best_fitness_final = int(best_fitness)
    best_individual_final = [int(param) for param in best_individual]
    return best_fitness_final, best_individual_final

# Run the Genetic Algorithm
best_fitness, best_control_params = genetic_algorithm(pop_size=50, generations=100)
print(f"\nBest Control Parameters: {best_control_params}, Best Fitness: {best_fitness}")