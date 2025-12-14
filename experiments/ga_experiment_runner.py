"""
Run GA experiments across TSPLIB instances and save generated trajectories.

Usage (example):
python experiments/ga_experiment_runner.py --n 100 --generations 2000 --outdir experiments --seed 42

This script:
- finds `.tsp` files under `data/problem_instances` (or `--data-dir`)
- for each instance, runs `TSPGeneticAlgorithm.evolve()` `n` times with random initial populations
- saves results to `outputs/<instance>_ga_only_evolved_trajectories.json` (JSON list of trajectories)
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from data.tsplib_loader import TSPLIBLoader
from utils.visualization import TableBuilder
from algorithms.genetic_algorithm.ga_agent import TSPGeneticAlgorithm


def list_instances(data_dir):
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.tsp'):
            yield os.path.splitext(f)[0]


def generate_random_initial_routes(dimension, population_size):
    """Generate random initial routes for GA (for GA-only algorithm).
       Starts with 1, end with 1
    """
    routes = []
    for _ in range(population_size):
        # 1-indexed cities excluding the fixed start city (1)
        inner_cities = list(range(2, dimension + 1))
        random.shuffle(inner_cities)

        # Create the N+1 route: [1] + [shuffled N-1 cities] + [1]
        route = [1] + inner_cities + [1]
        routes.append(route)
    return routes


def _single_run_worker(args_tuple):
    """Module-level helper for ProcessPoolExecutor (must be picklable).

    args_tuple: (instance_name, data_dir, generations, seed, run_index, ga_config, initial_routes)
    Returns a serializable dict with run result.
    """
    instance_name, data_dir, generations, seed, run_index, ga_config, initial_routes = args_tuple

    # Each worker creates its own loader to avoid shared state
    loader = TSPLIBLoader(data_dir=data_dir)
    data = loader.load_instance(instance_name)

    # Set per-worker RNG
    np.random.seed(seed + run_index)
    random.seed(seed + run_index)

    # Initialize GA with config parameters
    ga = TSPGeneticAlgorithm(
        population_size=ga_config['population_size'],
        mutation_rate=ga_config['mutation_rate'],
        selection_method=ga_config['selection_method']
    )

    # Run evolution
    result = ga.evolve(data, generations, initial_routes)

    # Extract best route and cost
    best_route = result['best_route']
    best_distance = result['best_distance']

    # Convert route to 1-based if needed (should already be 1-based from initial_routes)
    route_1based = [int(city) for city in best_route]

    return {
        'run': run_index,
        'cost': float(best_distance),
        'route': route_1based,
        'cost_per_gen': result['cost_per_gen']
    }


def run_for_instance(instance_name, loader, n, generations, seed, outdir, ga_config, initial_routes, workers=None):
    """Run `n` independent GA evolution runs and save results. Runs are executed in parallel.
    Args:
        instance_name: Name of TSP instance
        loader: TSPLIBLoader instance
        n: Number of runs
        generations: Number of GA generations per run
        seed: Base random seed
        outdir: Output directory
        ga_config: GA configuration dict (required, no defaults)
        initial_routes: List of initial routes for GA population
        workers: Number of worker processes (defaults to os.cpu_count())
    """
    data_dir = loader.data_dir
    data = loader.load_instance(instance_name)
    optimal_value = data.get('optimal_value', None)
    trajectories = []

    # Build args for each run
    tasks = [(instance_name, data_dir, generations, seed, run, ga_config, initial_routes) for run in range(n)]

    max_workers = workers or os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_single_run_worker, t): t[4] for t in tasks}

        for fut in as_completed(futures):
            run_index = futures[fut]
            try:
                res = fut.result()
                trajectories.append(res)
            except Exception as e:
                print(f"Run {run_index} failed for {instance_name}: {e}")

    # Sort by run index to keep deterministic ordering
    trajectories.sort(key=lambda x: x['run'])

    # Build filename with GA config info
    config_suffix = f"_sel_{ga_config['selection_method']}_pop_{ga_config['population_size']}_mut_{ga_config['mutation_rate']}"
    outpath = Path(outdir) / f"{instance_name}_ga_only{config_suffix}.json"

    with open(outpath, 'w') as fh:
        json.dump({
            'instance': instance_name,
            'optimal_value': optimal_value,
            'n': n,
            'trajectories': trajectories
        }, fh, indent=2)

    print(f"Saved {len(trajectories)} GA trajectories for {instance_name} -> {outpath}")


def main(argv=None):

    parser = argparse.ArgumentParser(description='Run GA experiments to generate evolved trajectories')
    parser.add_argument('--data-dir', default='../data/problem_instances', help='Path to .tsp files')
    parser.add_argument('--n', type=int, default=100, help='Number of trajectories per instance')
    parser.add_argument('--generations', type=int, default=2000, help='GA generations per run')
    parser.add_argument('--outdir', default='../outputs', help='Directory to save trajectories')
    parser.add_argument('--seed', type=int, default=42, help='Base RNG seed')
    parser.add_argument('--instances', default=None, help='Comma-separated list of instance names to run (without .tsp)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes for parallel runs')

    # GA-specific parameters
    parser.add_argument('--population-size', type=int, default=100, help='GA population size')
    parser.add_argument('--mutation-rate', type=float, default=0.01, help='GA mutation rate')

    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    loader = TSPLIBLoader(data_dir=args.data_dir)

    # Determine which instances to run
    if args.instances:
        instances = [s.strip() for s in args.instances.split(',') if s.strip()]
    else:
        instances = list(list_instances(args.data_dir))

    # Define parameter combinations (all as lists for future extensibility)
    ga_selection_methods = ['roulette_wheel', 'elitist']
    ga_mutation_rates = [args.mutation_rate]
    ga_population_sizes = [args.population_size]

    # Pre-generate initial routes for each instance (same routes used across all GA configs)
    initial_routes_by_instance = {}
    random.seed(args.seed)
    for instance in instances:
        data = loader.load_instance(instance)
        initial_routes_by_instance[instance] = generate_random_initial_routes(
            data['dimension'],
            args.population_size
        )

    for selection_method in ga_selection_methods:
        for mutation_rate in ga_mutation_rates:
            for population_size in ga_population_sizes:

                # Build GA configuration
                ga_config = {
                    'population_size': population_size,
                    'mutation_rate': mutation_rate,
                    'selection_method': selection_method
                }

                print(f"GA Configuration: {ga_config}")
                print(f"Generations: {args.generations}")

                for instance in instances:
                    print(f"Running GA for instance: {instance} (n={args.n}, generations={args.generations})")
                    try:
                        run_for_instance(
                            instance, loader, args.n, args.generations, args.seed,
                            args.outdir, ga_config, initial_routes_by_instance[instance], workers=args.workers
                        )
                    except Exception as e:
                        print(f"Error while processing {instance}: {e}")

    # results table
    df = TableBuilder.build_table_algorithm_comparison(
        results_dir=args.outdir,
        output_csv="../results/tsp_instances_table2.csv",
        print_table=True)


if __name__ == '__main__':
    main()