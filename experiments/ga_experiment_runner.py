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
    """Generate random initial routes for GA (since we're testing GA only)."""
    routes = []
    for _ in range(population_size):
        route = list(range(1, dimension + 1))  # 1-indexed cities
        random.shuffle(route)
        routes.append(route)
    return routes


def _single_run_worker(args_tuple):
    """Module-level helper for ProcessPoolExecutor (must be picklable).

    args_tuple: (instance_name, data_dir, generations, seed, run_index, ga_config)
    Returns a serializable dict with run result.
    """
    instance_name, data_dir, generations, seed, run_index, ga_config = args_tuple

    # Each worker creates its own loader to avoid shared state
    loader = TSPLIBLoader(data_dir=data_dir)
    data = loader.load_instance(instance_name)

    # Set per-worker RNG
    np.random.seed(seed + run_index)
    random.seed(seed + run_index)

    # Initialize GA with config parameters
    ga = TSPGeneticAlgorithm(
        population_size=ga_config.get('population_size'),
        mutation_rate=ga_config.get('mutation_rate', 0.01),
        selection_method=ga_config.get('selection_method', 'roulette_wheel')
    )

    # Generate random initial routes (GA-only approach)
    initial_routes = generate_random_initial_routes(
        data['dimension'],
        ga_config.get('population_size')
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
        'route': route_1based
    }


def run_for_instance(instance_name, loader, n, generations, seed, outdir, ga_config=None, workers=None):
    """Run `n` independent GA evolution runs and save results. Runs are executed in parallel.

    - `workers` controls the max number of worker processes (defaults to os.cpu_count()).
    """
    data_dir = loader.data_dir
    data = loader.load_instance(instance_name)
    optimal_value = data.get('optimal_value', None)
    trajectories = []

    # Default GA config if none provided
    if ga_config is None:
        ga_config = {
            'population_size': 40,
            'mutation_rate': 0.01,
            'selection_method': 'roulette_wheel'
        }

    # Build args for each run
    tasks = [(instance_name, data_dir, generations, seed, run, ga_config) for run in range(n)]

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
    config_str_parts = []
    if ga_config:
        if 'population_size' in ga_config and ga_config['population_size'] != 40:
            config_str_parts.append(f"pop_{ga_config['population_size']}")
        if 'mutation_rate' in ga_config and ga_config['mutation_rate'] != 0.01:
            config_str_parts.append(f"mut_{ga_config['mutation_rate']}")
        if 'selection_method' in ga_config and ga_config['selection_method'] != 'roulette_wheel':
            config_str_parts.append(f"sel_{ga_config['selection_method']}")

    config_suffix = "_" + "_".join(config_str_parts) if config_str_parts else ""
    outpath = Path(outdir) / f"{instance_name}_ga_only{config_suffix}_evolved_trajectories.json"

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
    parser.add_argument('--instances', default=None,
                        help='Comma-separated list of instance names to run (without .tsp)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes for parallel runs')

    # GA-specific parameters
    parser.add_argument('--population-size', type=int, default=100, help='GA population size')
    parser.add_argument('--mutation-rate', type=float, default=0.01, help='GA mutation rate')
    parser.add_argument('--selection-method', default='roulette_wheel',
                        choices=['roulette_wheel', 'elitist'], help='GA selection method')

    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    loader = TSPLIBLoader(data_dir=args.data_dir)

    # Determine which instances to run
    if args.instances:
        instances = [s.strip() for s in args.instances.split(',') if s.strip()]
    else:
        instances = list(list_instances(args.data_dir))

    # Build GA configuration
    ga_config = {
        'population_size': args.population_size,
        'mutation_rate': args.mutation_rate,
        'selection_method': args.selection_method
    }

    print(f"GA Configuration: {ga_config}")
    print(f"Generations: {args.generations}")

    for instance in instances:
        print(f"Running GA for instance: {instance} (n={args.n}, generations={args.generations})")
        try:
            run_for_instance(
                instance, loader, args.n, args.generations, args.seed,
                args.outdir, ga_config, workers=args.workers
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