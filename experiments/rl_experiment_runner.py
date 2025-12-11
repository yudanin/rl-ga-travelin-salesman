"""
Run RL agent experiments across TSPLIB instances and save generated trajectories.

Usage (example):
python experiments/rl_experiment_runner.py --n 100 --episodes 1000 --outdir outputs --seed 42

This script:
- finds `.tsp` files under `data/problem_instances` (or `--data-dir`)
- for each instance, runs `TSPRLAgent.train()` and then `TSPRLAgent.optimal_route()` `n` times
- saves results to `outputs/<instance>_trajectories.json` (JSON list of trajectories)
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from data.tsplib_loader import TSPLIBLoader
from algorithms.reinforcement_learning.rl_agent import TSPRLAgent


def list_instances(data_dir):
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.tsp'):
            yield os.path.splitext(f)[0]


def _single_run_worker(args_tuple):
    """Module-level helper for ProcessPoolExecutor (must be picklable).

    args_tuple: (instance_name, data_dir, episodes, seed, run_index, config_overrides)
    Returns a serializable dict with run result.
    """
    instance_name, data_dir, episodes, seed, run_index, config_overrides = args_tuple

    # Each worker creates its own loader and agent to avoid shared state
    loader = TSPLIBLoader(data_dir=data_dir)
    data = loader.load_instance(instance_name)

    config = {'episodes': episodes}
    if config_overrides:
        config.update(config_overrides)

    # Set per-worker RNG
    np.random.seed(seed + run_index)

    agent = TSPRLAgent(data, config)
    agent.train()
    cost, route = agent.optimal_route()

    # Convert route to 1-based node ids for compatibility with TSPLIB conventions
    route_1based = [int(r + 1) for r in route]

    return {'run': run_index, 'cost': float(cost), 'route': route_1based}


def run_for_instance(instance_name, loader, n, episodes, seed, outdir, config_overrides=None, workers=None):
    """Run `n` independent train+eval runs and save results. Runs are executed in parallel.

    - `workers` controls the max number of worker processes (defaults to os.cpu_count()).
    """
    data_dir = loader.data_dir
    loader = TSPLIBLoader(data_dir=data_dir)
    data = loader.load_instance(instance_name)
    optimal_value = data.get('optimal_value', None)
    trajectories = []

    # Build args for each run
    tasks = [ (instance_name, data_dir, episodes, seed, run, config_overrides) for run in range(n) ]

    max_workers = workers or os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = { exe.submit(_single_run_worker, t): t[4] for t in tasks }

        for fut in as_completed(futures):
            run_index = futures[fut]
            try:
                res = fut.result()
                trajectories.append(res)
            except Exception as e:
                print(f"Run {run_index} failed for {instance_name}: {e}")

    # Sort by run index to keep deterministic ordering
    trajectories.sort(key=lambda x: x['run'])

    outpath = Path(outdir) / f"{instance_name}_gamma_{config_overrides['gamma']}_reward_type_{config_overrides['reward_type']}_epsilon_type_{config_overrides['epsilon_greedy_type']}.json"
    with open(outpath, 'w') as fh:
        json.dump({'instance': instance_name, 'optimal_value' : optimal_value, 'n': n, 'trajectories': trajectories}, fh, indent=2)

    print(f"Saved {len(trajectories)} trajectories for {instance_name} -> {outpath}")


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run RL experiments to generate trajectories')
    parser.add_argument('--data-dir', default='data/problem_instances', help='Path to .tsp files')
    parser.add_argument('--n', type=int, default=100, help='Number of trajectories per instance')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes per run')
    parser.add_argument('--outdir', default='outputs', help='Directory to save trajectories')
    parser.add_argument('--seed', type=int, default=42, help='Base RNG seed')
    parser.add_argument('--instances', default=None, help='Comma-separated list of instance names to run (without .tsp)')
    parser.add_argument('--configs', default=None, help='Comma-separated list of config JSON paths under `configs/` or glob (overrides --instances)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes for parallel runs')
    parser.add_argument('--alpha', type=float, default=None, help='Optional alpha override')
    parser.add_argument('--gamma', type=float, default=None, help='Optional gamma override')

    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)

    loader = TSPLIBLoader(data_dir=args.data_dir)
    print(args.configs)

    # If configs provided, load them and run per-config. Otherwise fallback to instances discovery.
    if args.configs:
        # configs may be comma-separated paths or globs under configs/
        config_paths = []
        for part in args.configs.split(','):
            part = part.strip()
            if not part:
                continue
            # expand simple glob if used
            if '*' in part:
                from glob import glob
                config_paths.extend(glob(part))
            else:
                config_paths.append(part)

        # Normalize paths inside configs/ if bare filenames passed
        config_paths = [p if os.path.isabs(p) or os.path.dirname(p) else os.path.join('configs', p) for p in config_paths]

        for cfg_path in config_paths:
            try:
                with open(cfg_path, 'r') as fh:
                    cfg = json.load(fh)
            except Exception as e:
                print(f"Failed to load config {cfg_path}: {e}")
                continue

            instance = cfg.get('instance')
            if not instance:
                print(f"Config {cfg_path} missing 'instance' key; skipping")
                continue

            episodes = cfg.get('episodes', args.episodes)
            # pass the rest of cfg as overrides (remove instance and episodes)
            cfg_overrides = {k: v for k, v in cfg.items() if k not in ('instance', 'episodes')}

            print(f"Running config {cfg_path} for instance {instance} (n={args.n}, episodes={episodes})")
            try:
                run_for_instance(instance, loader, args.n, episodes, args.seed, args.outdir, cfg_overrides, workers=args.workers)
            except Exception as e:
                print(f"Error while processing {instance} from config {cfg_path}: {e}")

    else:
        if args.instances:
            instances = [s.strip() for s in args.instances.split(',') if s.strip()]
        else:
            instances = list(list_instances(args.data_dir))

        config_overrides = {}
        if args.alpha is not None:
            config_overrides['alpha'] = args.alpha
        if args.gamma is not None:
            config_overrides['gamma'] = args.gamma

        for instance in instances:
            print(f"Running instance: {instance} (n={args.n}, episodes={args.episodes})")
            try:
                run_for_instance(instance, loader, args.n, args.episodes, args.seed, args.outdir, config_overrides, workers=args.workers)
            except Exception as e:
                print(f"Error while processing {instance}: {e}")


if __name__ == '__main__':
    main()
