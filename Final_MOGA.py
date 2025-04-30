import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import seaborn as sns
import os
import logging
from typing import List, Dict, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 1. Configuration

CONFIG = {
    'population_size': 100,
    'num_generations': 50,
    'crossover_prob': 0.7,
    'mutation_prob': 0.4,
    'mutation_indpb': 0.05,  # Individual probability for mutation
    'random_seed': 21,
    'data_file': r'C:\Users\sudee\OneDrive - University of Plymouth\Desktop\COM5012\ptask (1)\data_4_23_59_33.dat'
    # 'data_file': r'C:\Users\sudee\OneDrive - University of Plymouth\Desktop\COM5012\ptask (1)\data_5_25_60_33.dat'
    }




def extract_data_from_dat_file(file_path: str) -> Tuple[List[Tuple[int, int]], Dict[int, List[int]]]:

    job_durations = []
    qualifications = {}

    try:
        with open(file_path) as file:
            lines = file.readlines()
            logger.info(f"Successfully opened file: {file_path}")
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        logger.info("Using generated sample data instead")
        print('File not found')
        return


    reading_jobs = False
    reading_quals = False

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue  # Skip comments and empty lines

        if line.startswith("Jobs"):
            num_jobs = int(line.split("=")[1].strip())
            reading_jobs = True
            continue

        if reading_jobs:
            if line.startswith("Qualifications"):
                num_quals = int(line.split("=")[1].strip())
                reading_jobs = False
                reading_quals = True
                continue
            else:
                parts = line.split()
                if len(parts) == 2:
                    start, end = map(int, parts)
                    job_durations.append((start, end))

        elif reading_quals:
            if ':' in line:
                parts = line.split(':')
                worker_id = len(qualifications)
                qualified_jobs = list(map(int, parts[1].split()))
                qualifications[worker_id] = qualified_jobs

    # Verify data was loaded correctly
    if not job_durations or not qualifications:
        logger.warning("Data file was empty or improperly formatted")
        return

    logger.info(f"Loaded {len(job_durations)} jobs and {len(qualifications)} workers from file")
    return job_durations, qualifications


def load_data(file_path: Optional[str] = None) -> Tuple[List[Tuple[int, int]], Dict[int, List[int]]]:

    if file_path is None:
        file_path = CONFIG['data_file']

    try:
        return extract_data_from_dat_file(file_path)
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        return


# 2. GA Algorithm Components

def setup_ga(job_durations: List[Tuple[int, int]], qualifications: Dict[int, List[int]]) -> base.Toolbox:

    # Clear existing types if needed
    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "Individual" in creator.__dict__:
        del creator.Individual

    # Create fitness and individual types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Set up toolbox
    toolbox = base.Toolbox()
    num_workers = len(qualifications)
    num_jobs = len(job_durations)

    logger.debug(f"Setting up GA with {num_workers} workers and {num_jobs} jobs")

    # Registration of operators
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: np.random.randint(0, num_workers), n=num_jobs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual,
                     n=CONFIG['population_size'])

    # Define evaluation function as a closure to capture the problem data
    def evaluate(individual: List[int]) -> Tuple[int, int, int]:

        shifts_used = len(set(individual))  # Number of unique workers used
        penalties = 0

        # Use a more efficient data structure for tracking worker schedules
        worker_jobs = [[] for _ in range(num_workers)]

        # Assign jobs to workers and calculate penalties
        for job_index, worker_id in enumerate(individual):
            # Check qualification penalty
            if job_index not in qualifications.get(worker_id, []):
                penalties += 1

            # Add job to worker's schedule
            worker_jobs[worker_id].append(job_durations[job_index])

        # Calculate makespan (maximum end time across all workers)
        makespan = 0
        for worker_id, jobs in enumerate(worker_jobs):
            if not jobs:
                continue

            # Sort jobs by start time
            jobs.sort(key=lambda x: x[0])

            # Calculate end time for this worker
            # Just take the maximum end time for simplicity
            worker_end_time = max(job[1] for job in jobs)
            makespan = max(makespan, worker_end_time)

        return shifts_used, penalties, makespan

    # Register operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_workers - 1,
                     indpb=CONFIG['mutation_indpb'])
    toolbox.register("select", tools.selNSGA2)

    return toolbox


def run_nsga2(toolbox: base.Toolbox) -> Tuple[List[Any], tools.Logbook]:

    # Set random seed for reproducibility
    random.seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    logger.info("Starting NSGA-II algorithm run")

    # Initialize population
    pop = toolbox.population()

    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)

    # Run algorithm
    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CONFIG['crossover_prob'],
        mutpb=CONFIG['mutation_prob'],
        ngen=CONFIG['num_generations'],
        stats=stats,
        verbose=True
    )

    logger.info(f"NSGA-II algorithm completed after {CONFIG['num_generations']} generations")

    return pop, logbook



# 3. Visualization Functions

def plot_pareto_front_3d(population: List[Any]) -> List[Any]:


    logger.info("Plotting 3D Pareto front")

    # Extract fitness values
    fits = [ind.fitness.values for ind in population]
    shifts = [fit[0] for fit in fits]
    penalties = [fit[1] for fit in fits]
    makespans = [fit[2] for fit in fits]

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot - we use size and color to enhance visibility
    scatter = ax.scatter(shifts, penalties, makespans, c=makespans, cmap='viridis',
                         s=100, alpha=0.6, edgecolors='w')

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Makespan')

    # Set labels and title
    ax.set_xlabel('Number of Shifts Used')
    ax.set_ylabel('Qualification Penalties')
    ax.set_zlabel('Makespan')
    plt.title('3D Pareto Front - NSGA-II Optimization Results')

    # Identify and mark non-dominated solutions
    non_dominated = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    non_dom_shifts = [ind.fitness.values[0] for ind in non_dominated]
    non_dom_penalties = [ind.fitness.values[1] for ind in non_dominated]
    non_dom_makespans = [ind.fitness.values[2] for ind in non_dominated]

    # Highlight Pareto front points
    ax.scatter(non_dom_shifts, non_dom_penalties, non_dom_makespans,
               color='red', s=150, alpha=0.8, marker='*', label='Pareto Front')

    # Connect Pareto front points with lines
    # Sort points for cleaner visualization
    pareto_points = list(zip(non_dom_shifts, non_dom_penalties, non_dom_makespans))

    # Sort by first objective (shifts)
    pareto_points.sort()

    # Draw lines connecting Pareto points
    if pareto_points:
        x_pareto, y_pareto, z_pareto = zip(*pareto_points)
        ax.plot(x_pareto, y_pareto, z_pareto, 'r-', linewidth=2, alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig('pareto_front_3d.png')  # Save figure
    plt.show()

    return non_dominated


def plot_shifts_penalties_heatmap(population: List[Any]) -> None:

    logger.info("Plotting shifts vs penalties heatmap")

    # Extract fitness values
    fits = [ind.fitness.values for ind in population]
    shifts = [int(fit[0]) for fit in fits]
    penalties = [int(fit[1]) for fit in fits]

    # Determine the range for shifts and penalties
    min_shifts, max_shifts = min(shifts), max(shifts)
    min_penalties, max_penalties = min(penalties), max(penalties)

    # Create a 2D histogram (counts in each bin)
    heatmap_data = np.zeros((max_shifts - min_shifts + 1, max_penalties - min_penalties + 1))

    # Count solutions in each bin
    for s, p in zip(shifts, penalties):
        heatmap_data[s - min_shifts, p - min_penalties] += 1

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="g",
                cbar_kws={'label': 'Number of Solutions'})

    # Set labels and title
    plt.xlabel("Qualification Penalties")
    plt.ylabel("Shifts Used")
    plt.title("Heatmap of Solutions Distribution: Shifts vs Penalties")

    # Adjust the tick labels to match the actual values
    plt.yticks(np.arange(0.5, max_shifts - min_shifts + 1),
               range(min_shifts, max_shifts + 1))
    plt.xticks(np.arange(0.5, max_penalties - min_penalties + 1),
               range(min_penalties, max_penalties + 1))

    plt.tight_layout()
    plt.savefig('shifts_vs_penalties_heatmap.png')
    plt.show()





def plot_objective_relationships(non_dominated_solutions: List[Any]) -> None:

    logger.info("Plotting objective relationships")

    # Extract fitness values from Pareto front
    fitnesses = [ind.fitness.values for ind in non_dominated_solutions[:5]]

    shifts_list = [f[0] for f in fitnesses]
    penalties_list = [f[1] for f in fitnesses]
    makespans_list = [f[2] for f in fitnesses]

    # --------- Figure 1: Shifts vs Penalties ---------
    plt.figure(figsize=(7, 5))
    plt.scatter(shifts_list, penalties_list, c='blue')
    plt.xlabel('Shifts Used')
    plt.ylabel('Penalties')
    plt.title('Shifts vs Penalties')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('shifts_vs_penalties.png')
    plt.show()

    # --------- Figure 2: Shifts vs Makespan and Penalties vs Makespan ---------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Shifts vs Makespan
    axes[0].scatter(shifts_list, makespans_list, c='green')
    axes[0].set_xlabel('Shifts Used')
    axes[0].set_ylabel('Makespan')
    axes[0].set_title('Shifts vs Makespan')
    axes[0].grid(True)

    # Penalties vs Makespan
    axes[1].scatter(penalties_list, makespans_list, c='red')
    axes[1].set_xlabel('Penalties')
    axes[1].set_ylabel('Makespan')
    axes[1].set_title('Penalties vs Makespan')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('makespan_relationships.png')
    plt.show()



# 4. Analysis Functions

def analyze_solution(solution: List[int], job_durations: List[Tuple[int, int]],
                     qualifications: Dict[int, List[int]]) -> None:

    # Print solution name and detailed schedule

    num_workers = len(qualifications)
    num_jobs = len(job_durations)

    # Create worker schedules
    worker_schedules = [[] for _ in range(num_workers)]
    for job_index, worker_id in enumerate(solution):
        job = job_durations[job_index]
        qualified = job_index in qualifications.get(worker_id, [])
        worker_schedules[worker_id].append((job_index, job[0], job[1], qualified))

    # Print detailed schedule
    print("\nDetailed Schedule Analysis:")
    print("==========================")



    total_penalties = 0
    max_end_time = 0
    workers_used = set()

    for worker_id, jobs in enumerate(worker_schedules):
        if not jobs:
            continue

        workers_used.add(worker_id)
        jobs.sort(key=lambda x: x[1])  # Sort by start time

        print(f"\nWorker {worker_id + 1}:")
        print("  Job ID  | Start | End   | Duration | Qualified")
        print("  --------|-------|-------|----------|----------")

        worker_penalties = 0
        for job_id, start, end, qualified in jobs:
            duration = end - start
            qualified_str = "Yes" if qualified else "No"
            if not qualified:
                worker_penalties += 1
            print(f"  {job_id + 1:<7} | {start:<5} | {end:<5} | {duration:<8} | {qualified_str}")

            max_end_time = max(max_end_time, end)

        print(f"  Total jobs: {len(jobs)}, Qualification penalties: {worker_penalties}")
        total_penalties += worker_penalties

    print("\nSummary:")
    print(f"  Workers used: {len(workers_used)} out of {num_workers}")
    print(f"  Total qualification penalties: {total_penalties}")
    print(f"  Makespan (total completion time): {max_end_time}")
    print("==========================")



# 5. Main Execution Functions

def main(file_path: Optional[str] = None, visualize: bool = True) -> Tuple[List[Any], tools.Logbook]:

    logger.info("Starting personnel scheduling optimization")

    # 1. Load data
    job_durations, qualifications = load_data(file_path)

    # Print data summary
    print(f"\nData Summary:")
    print(f"  Jobs: {len(job_durations)}")
    print(f"  Workers: {len(qualifications)}")

    # 2. Setup GA
    toolbox = setup_ga(job_durations, qualifications)

    # 3. Run algorithm
    final_pop, logbook = run_nsga2(toolbox)

    # 4. Extract Pareto front
    non_dominated = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]

    # 5. Visualize results
    if visualize:
        non_dominated = plot_pareto_front_3d(final_pop)
        plot_shifts_penalties_heatmap(final_pop)
        # plot_solution_heatmap(non_dominated, len(qualifications), len(job_durations))
        plot_objective_relationships(non_dominated)
        # plot_convergence(logbook)

    # 6. Print best solutions
    print("\nPareto Front Solutions:")
    for i, solution in enumerate(non_dominated[:5]):  # Show top 5 solutions
        print(f"Solution {i + 1}: Shifts={solution.fitness.values[0]}, "
              f"Penalties={solution.fitness.values[1]}, Makespan={solution.fitness.values[2]}")

    # 7. Detailed analysis of the best solution (by makespan)
    if non_dominated:
        best_solution = min(non_dominated, key=lambda x: x.fitness.values[2])
        print("\nDetailed analysis of the solution with minimum makespan:")
        analyze_solution(best_solution, job_durations, qualifications)

    return non_dominated, logbook


# Entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")