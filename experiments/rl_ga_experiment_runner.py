"""
Run Hybrid RL+GA experiments across TSPLIB instances.

Usage (example):
python experiments/rl_ga_experiment_runner.py --data-dir ../data/problem_instances --outdir ../outputs --seed 42

This script:
1. Runs RL agent for all parameter combinations and saves outputs
2. Waits for all RL runs to complete
3. Uses RL output routes directly as initial population for GA evolution
4. Saves results with combined configuration in filename

Parameter combinations:
- RL: gamma in [0.01, 0.15, 0.3], reward_type in [1], epsilon_greedy_type in [1], alpha in [0.01], episodes=1000
- GA: population=40, selection_method in ['roulette_wheel', 'elitist'], mutation_rate in [0.01], generations=2000
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from data.tsplib_loader import TSPLIBLoader
from utils.visualization import TableBuilder

# Import from existing experiment runners
from experiments.rl_experiment_runner import (
    list_instances,
    run_for_instance as run_rl_for_instance
)
from experiments.ga_experiment_runner import _single_run_worker as ga_single_run_worker


def validate_rl_config(rl_config):
    """Validate that all required RL config parameters are present."""
    required_keys = ['episodes', 'alpha', 'gamma', 'reward_type', 'epsilon_greedy_type']
    for key in required_keys:
        if key not in rl_config:
            raise ValueError(f"RL config missing required parameter: {key}")


def validate_ga_config(ga_config):
    """Validate that all required GA config parameters are present."""
    required_keys = ['population_size', 'selection_method', 'mutation_rate']
    for key in required_keys:
        if key not in ga_config:
            raise ValueError(f"GA config missing required parameter: {key}")


def get_rl_output_filename(instance_name, rl_config):
    """Generate RL output filename based on config."""
    return (
        f"{instance_name}"
        f"_gamma_{rl_config['gamma']}"
        f"_reward_type_{rl_config['reward_type']}"
        f"_epsilon_type_{rl_config['epsilon_greedy_type']}"
        f"_trajectories.json"
    )


def get_hybrid_output_filename(instance_name, rl_config, ga_config):
    """Generate hybrid output filename based on configs."""
    return (
        f"{instance_name}_rl_ga"
        f"_rl_gamma_{rl_config['gamma']}"
        f"_reward_type_{rl_config['reward_type']}"
        f"_epsilon_type_{rl_config['epsilon_greedy_type']}"
        f"_ga_pop_{ga_config['population_size']}"
        f"_sel_{ga_config['selection_method']}"
        f"_mut_{ga_config['mutation_rate']}"
        f".json"
    )


def run_hybrid_ga_on_rl_output(rl_output_path, loader, n, generations, seed, outdir,
                               rl_config, ga_config, workers=None):
    """
    Run GA using RL output routes as initial population:
    takes the best routes
    """
    validate_rl_config(rl_config)
    validate_ga_config(ga_config)

    # Load RL output
    with open(rl_output_path, 'r') as fh:
        rl_data = json.load(fh)

    instance_name = rl_data['instance']
    rl_trajectories = rl_data['trajectories']

    population_size = ga_config['population_size']

    # Sort RL trajectories by cost (best first) and take top `population_size`
    sorted_trajectories = sorted(rl_trajectories, key=lambda x: x['cost'])
    best_rl_routes = [traj['route'] for traj in sorted_trajectories[:population_size]]

    # If we have fewer RL routes than population_size, use what we have
    if len(best_rl_routes) < population_size:
        print(f"Warning: Only {len(best_rl_routes)} RL routes available, "
              f"but population_size is {population_size}")

    data = loader.load_instance(instance_name)
    optimal_value = data.get('optimal_value', None)
    trajectories = []

    # Build args for each run
    # all runs use the same initial_routes (best RL routes)
    tasks = [
        (instance_name, loader.data_dir, generations, seed, run, ga_config, best_rl_routes)
        for run in range(n)
    ]

    max_workers = workers or os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(ga_single_run_worker, t): t[4] for t in tasks}

        for fut in as_completed(futures):
            run_index = futures[fut]
            try:
                res = fut.result()
                trajectories.append(res)
            except Exception as e:
                print(f"GA Run {run_index} failed for {instance_name}: {e}")

    # Sort by run index
    trajectories.sort(key=lambda x: x['run'])

    # Save hybrid output
    filename = get_hybrid_output_filename(instance_name, rl_config, ga_config)
    outpath = Path(outdir) / filename

    with open(outpath, 'w') as fh:
        json.dump({
            'instance': instance_name,
            'optimal_value': optimal_value,
            'n': len(trajectories),
            'rl_config': rl_config,
            'ga_config': ga_config,
            'generations': generations,
            'trajectories': trajectories
        }, fh, indent=2)

    print(f"Saved {len(trajectories)} hybrid trajectories for {instance_name} -> {outpath}")
    return outpath


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run Hybrid RL+GA experiments')
    parser.add_argument('--data-dir', default='../data/problem_instances', help='Path to .tsp files')
    parser.add_argument('--n', type=int, default=100, help='Number of trajectories per instance')
    parser.add_argument('--generations', type=int, default=2000, help='GA generations per run')
    parser.add_argument('--outdir', default='../outputs', help='Directory to save trajectories')
    parser.add_argument('--seed', type=int, default=42, help='Base RNG seed')
    parser.add_argument('--instances', default=None, help='Comma-separated list of instance names to run (without .tsp)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes for parallel runs')

    args = parser.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    loader = TSPLIBLoader(data_dir=args.data_dir)

    # Determine which instances to run
    if args.instances:
        instances = [s.strip() for s in args.instances.split(',') if s.strip()]
    else:
        instances = list(list_instances(args.data_dir))

    # =========================================================================
    # Define parameter combinations (all as lists for future extensibility)
    # =========================================================================

    # RL parameters
    rl_gammas = [0.01, 0.15, 0.3]
    rl_reward_types = [1]
    rl_epsilon_greedy_types = [1]
    rl_alphas = [0.01]
    rl_episodes_list = [1000]

    # GA parameters
    ga_population_sizes = [40]
    ga_selection_methods = ['roulette_wheel', 'elitist']
    ga_mutation_rates = [0.01]

    # =========================================================================
    # Phase 1: Run RL for all parameter combinations
    # =========================================================================

    print("=" * 70)
    print("PHASE 1: Running RL experiments")
    print("=" * 70)

    rl_output_files = []  # Track all RL output files with their configs

    for gamma in rl_gammas:
        for reward_type in rl_reward_types:
            for epsilon_greedy_type in rl_epsilon_greedy_types:
                for alpha in rl_alphas:
                    for episodes in rl_episodes_list:

                        rl_config = {
                            'gamma': gamma,
                            'reward_type': reward_type,
                            'epsilon_greedy_type': epsilon_greedy_type,
                            'alpha': alpha,
                            'episodes': episodes
                        }

                        validate_rl_config(rl_config)

                        print("-" * 70)
                        print(f"RL Config: gamma={gamma}, reward_type={reward_type}, "
                              f"epsilon_type={epsilon_greedy_type}, alpha={alpha}, episodes={episodes}")
                        print("-" * 70)

                        for instance in instances:
                            print(f"Running RL for instance: {instance} (n={args.n})")
                            try:
                                # Use imported run_rl_for_instance from rl_experiment_runner
                                run_rl_for_instance(
                                    instance, loader, args.n, episodes, args.seed,
                                    args.outdir, rl_config, workers=args.workers
                                )

                                # Track output file path
                                outpath = Path(args.outdir) / get_rl_output_filename(instance, rl_config)
                                rl_output_files.append({
                                    'path': outpath,
                                    'instance': instance,
                                    'rl_config': rl_config.copy()
                                })
                            except Exception as e:
                                print(f"Error while processing {instance}: {e}")

    print("\n" + "=" * 70)
    print(f"PHASE 1 COMPLETE: Generated {len(rl_output_files)} RL output files")
    print("=" * 70 + "\n")

    # =========================================================================
    # Phase 2: Run GA on all RL outputs for all GA parameter combinations
    # =========================================================================

    print("=" * 70)
    print("PHASE 2: Running GA experiments on RL outputs")
    print("=" * 70)

    hybrid_output_files = []

    for selection_method in ga_selection_methods:
        for mutation_rate in ga_mutation_rates:
            for population_size in ga_population_sizes:

                ga_config = {
                    'population_size': population_size,
                    'selection_method': selection_method,
                    'mutation_rate': mutation_rate
                }

                validate_ga_config(ga_config)

                print("-" * 70)
                print(f"GA Config: pop={population_size}, selection={selection_method}, "
                      f"mutation={mutation_rate}")
                print("-" * 70)

                for rl_output in rl_output_files:
                    instance = rl_output['instance']
                    rl_config = rl_output['rl_config']
                    rl_path = rl_output['path']

                    print(f"Running GA for instance: {instance} "
                          f"(rl_gamma={rl_config['gamma']}, generations={args.generations})")
                    try:
                        outpath = run_hybrid_ga_on_rl_output(
                            rl_path, loader, args.n, args.generations, args.seed,
                            args.outdir, rl_config, ga_config, workers=args.workers
                        )
                        hybrid_output_files.append(outpath)
                    except Exception as e:
                        print(f"Error while processing {instance}: {e}")

    print("\n" + "=" * 70)
    print(f"PHASE 2 COMPLETE: Generated {len(hybrid_output_files)} hybrid output files")
    print("=" * 70 + "\n")

    print("All hybrid experiments completed.\n")

    # results table
    df = TableBuilder.build_table_algorithm_comparison(
        results_dir=args.outdir,
        output_csv="../results/tsp_instances_table2.csv",
        print_table=True)


if __name__ == '__main__':
    main()