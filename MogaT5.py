import random
import numpy as np


# 1. Data Loading and Parsing

def load_products(filename):
    """Loads product data from a file."""

    products = []
    with open(filename, 'r') as f:
        for line in f:
            family, quantity, length, width, height = map(int, line.strip().split())
            products.append({
                'family': family,
                'quantity': quantity,
                'dimensions': (length, width, height)
            })
    return products


def load_bay(filename):
    """Loads bay data from a file."""

    bays = []
    with open(filename, 'r') as f:
        for line in f:
            width, height, depth, available_height = map(int, line.strip().split())
            bays.append({
                'width': width,
                'height': height,
                'depth': depth,
                'available_height': available_height
            })
    return bays


def load_shelves(filename):
    """Loads shelf data from a file."""

    shelves = []
    with open(filename, 'r') as f:
        for line in f:
            number, thickness, position, top_gap, left_gap, inter_gap, right_gap = map(
                int, line.strip().split())
            shelves.append({
                'number': number,
                'thickness': thickness,
                'position': position,
                'top_gap': top_gap,
                'left_gap': left_gap,
                'inter_gap': inter_gap,
                'right_gap': right_gap
            })
    return shelves


# 2. Chromosome Representation and Initialization

def create_chromosome(products, bays, shelves):
    """Creates a random chromosome representing a solution."""

    chromosome = []
    product_order = list(range(len(products)))
    random.shuffle(product_order)  # Randomize the order of products

    for product_index in product_order:
        # Randomly choose orientation (6 possibilities)
        orientation = random.randint(0, 5)
        # Randomly assign to a bay
        bay_assignment = random.randint(0, len(bays) - 1)
        # Randomly assign to a shelf (within the bay)
        shelf_assignment = random.randint(0, len(shelves) - 1)

        chromosome.append({
            'product_index': product_index,
            'orientation': orientation,
            'bay_assignment': bay_assignment,
            'shelf_assignment': shelf_assignment
        })
    return chromosome


def initial_population(pop_size, products, bays, shelves):
    """Generates an initial population of chromosomes."""

    return [create_chromosome(products, bays, shelves) for _ in range(pop_size)]


# 3. Fitness Evaluation

def get_product_dimensions(product, orientation):
    """Gets the dimensions of a product based on its orientation."""

    l, w, h = product['dimensions']
    if orientation == 0:
        return (l, w, h)
    elif orientation == 1:
        return (l, h, w)
    elif orientation == 2:
        return (w, l, h)
    elif orientation == 3:
        return (w, h, l)
    elif orientation == 4:
        return (h, l, w)
    elif orientation == 5:
        return (h, w, l)
    else:
        raise ValueError("Invalid orientation")


# def check_placement_constraints(chromosome, products, bays, shelves):
#     """Checks if the placement of products in a chromosome is valid."""
#
#     for gene in chromosome:
#         product = products[gene['product_index']]
#         bay = bays[gene['bay_assignment']]
#         shelf = shelves[gene['shelf_assignment']]
#
#         l, w, h = get_product_dimensions(product, gene['orientation'])
#
#         # Check if product fits within bay
#         if l > bay['width'] or w > bay['depth'] or h > bay['available_height']:
#             return False
#
#         # Check shelf height constraint (product height vs. shelf position)
#         if h > shelf['position']:
#             return False
#
#         #  More complex checks (gaps, overlaps) would go here
#         #  This is a simplified version!
#
#     return True
def check_placement_constraints(chromosome, products, bays, shelves):
    """Checks if the placement of products in a chromosome is valid."""

    print("--- Checking Constraints ---")  # Overall check start

    for gene in chromosome:
        product = products[gene['product_index']]
        bay = bays[gene['bay_assignment']]
        shelf = shelves[gene['shelf_assignment']]

        l, w, h = get_product_dimensions(product, gene['orientation'])

        print(f"  Product: {gene['product_index']}, Bay: {gene['bay_assignment']}, Shelf: {gene['shelf_assignment']}")
        print(f"    Product dims: {l}, {w}, {h}")
        print(f"    Bay dims: {bay['width']}, {bay['depth']}, {bay['available_height']}")
        print(f"    Shelf position: {shelf['position']}")

        # Check if product fits within bay
        bay_fit = l <= bay['width'] and w <= bay['depth'] and h <= bay['available_height']
        print(f"    Bay fit check: {bay_fit}")
        if not bay_fit:
            print("    Constraint failed: Product does not fit in bay")
            return False

        # Check shelf height constraint (product height vs. shelf position)
        shelf_fit = h <= shelf['position']
        print(f"    Shelf fit check: {shelf_fit}")
        if not shelf_fit:
            print("    Constraint failed: Product is taller than shelf position")
            return False

        print("    Basic constraints passed")  # If it gets here, basic checks passed

    print("--- All constraints passed ---")  # If all genes pass
    return True

def calculate_fitness(chromosome, products, bays, shelves):
    """Calculates the fitness of a chromosome (solution)."""

    if not check_placement_constraints(chromosome, products, bays, shelves):
        #  Assign a very bad fitness if constraints are violated
        return float('inf'), 0, float('inf')  # High values for bad fitness

    total_product_volume = 0
    for gene in chromosome:
        product = products[gene['product_index']]
        l, w, h = get_product_dimensions(product, gene['orientation'])
        total_product_volume += l * w * h

    total_shelf_volume = 0
    used_shelves = set()
    for gene in chromosome:
        bay = bays[gene['bay_assignment']]
        shelf = shelves[gene['shelf_assignment']]
        #  This should be shelf position * bay area
        total_shelf_volume += bay['width'] * bay['depth'] * shelf['position']
        used_shelves.add((gene['bay_assignment'], gene['shelf_assignment']))

    space_utilization = total_product_volume / total_shelf_volume if total_shelf_volume else 0
    num_used_shelves = len(used_shelves)
    #  A very basic proxy for wasted space (can be improved)
    wasted_space = total_shelf_volume - total_product_volume

    #  Multi-objective: We want to MAXIMIZE space_utilization and MINIMIZE num_used_shelves and wasted_space
    #  Since GA's typically minimize, we can invert space_utilization
    return -space_utilization, num_used_shelves, wasted_space


# 4. Selection

def selection(population, fitnesses, num_parents):
    """Selects parents for crossover based on fitness (using tournament selection)."""

    parents = []
    for _ in range(num_parents):
        idx1 = random.randint(0, len(population) - 1)
        idx2 = random.randint(0, len(population) - 1)
        if is_better(fitnesses[idx1], fitnesses[idx2]):
            parents.append(population[idx1])
        else:
            parents.append(population[idx2])
    return parents


def is_better(fitness1, fitness2):
    """Checks if fitness1 is better than fitness2 (for multi-objective)."""

    #  In our fitness: (negative space_utilization, num_used_shelves, wasted_space)
    #  Lower is better for all objectives (except space_utilization, which is negated)
    if fitness1[0] < fitness2[0]:  # Better space utilization
        return True
    elif fitness1[0] == fitness2[0] and fitness1[1] < fitness2[1]:  # Fewer shelves
        return True
    elif fitness1[0] == fitness2[0] and fitness1[1] == fitness2[1] and fitness1[2] < fitness2[2]:  # Less wasted space
        return True
    else:
        return False


# 5. Crossover

def crossover(parent1, parent2):
    """Performs one-point crossover to create two offspring."""

    if len(parent1) < 2:
        return parent1[:], parent2[:]  # Return parents if too short for crossover

    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[:crossover_point:]
    return offspring1, offspring2


# 6. Mutation

def mutate(chromosome, products, bays, shelves, mutation_rate):
    """Mutates a chromosome."""

    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            mutation_type = random.randint(0, 2)  # Choose mutation type

            if mutation_type == 0:
                # Change orientation
                chromosome[i]['orientation'] = random.randint(0, 5)
            elif mutation_type == 1:
                # Change bay assignment
                chromosome[i]['bay_assignment'] = random.randint(0, len(bays) - 1)
            elif mutation_type == 2:
                # Change shelf assignment
                chromosome[i]['shelf_assignment'] = random.randint(0, len(shelves) - 1)
    return chromosome


# 7. Replacement

def replacement(population, offspring, fitnesses, offspring_fitnesses, pop_size):
    """Replaces the old population with a combination of parents and offspring (elitism)."""

    combined = list(zip(population, fitnesses)) + list(zip(offspring, offspring_fitnesses))
    combined.sort(key=lambda x: x[1])  # Sort by fitness (best first)
    new_population = [ind[0] for ind in combined[:pop_size]]
    new_fitnesses = [ind[1] for ind in combined[:pop_size]]
    return new_population, new_fitnesses


# 8. Main GA Function

def genetic_algorithm(products, bays, shelves, pop_size, num_generations, mutation_rate,
                      num_parents):
    """Executes the genetic algorithm."""

    population = initial_population(pop_size, products, bays, shelves)
    best_fitness_history = []  # To track best fitness over generations

    for generation in range(num_generations):
        fitnesses = [calculate_fitness(ind, products, bays, shelves) for ind in population]
        best_fitness = min(fitnesses)
        best_fitness_history.append(best_fitness)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

        parents = selection(population, fitnesses, num_parents)
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = crossover(parents[i], parents[i + 1])
                offspring.extend([child1, child2])

        offspring = [mutate(child, products, bays, shelves, mutation_rate) for child in offspring]
        offspring_fitnesses = [calculate_fitness(child, products, bays, shelves) for child in
                              offspring]

        population, fitnesses = replacement(population, offspring, fitnesses, offspring_fitnesses,
                                          pop_size)

    best_solution = population[fitnesses.index(min(fitnesses))]
    best_fitness = min(fitnesses)
    return best_solution, best_fitness, best_fitness_history


# 9. Main Execution

if __name__ == "__main__":
    products = load_products('products.txt')
    bays = load_bay('baytp2.txt')  # Or 'baytp2.txt'
    shelves = load_shelves('shelves.txt')

    pop_size = 100
    num_generations = 50
    mutation_rate = 0.1
    num_parents = 50

    best_solution, best_fitness, best_fitness_history = genetic_algorithm(products, bays, shelves,
                                                                          pop_size,
                                                                          num_generations, mutation_rate,
                                                                          num_parents)

    print("\nBest Solution:")
    print(best_solution)
    print("\nBest Fitness:")
    print(best_fitness)
    #  You can add code here to visualize the results or save them to a file