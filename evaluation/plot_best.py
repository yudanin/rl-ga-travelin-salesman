"""Plot best trajectories for all result JSON files in a folder.

Usage:
  python evaluation/plot_best.py --results-dir outputs --data-dir data/problem_instances

For each `*.json` file in `--results-dir` this script will:
- load the JSON, find the minimum-cost trajectory
- plot it using `utils.visualization.TSPVisualizer`
- save the figure to `<results-dir>/outputs/<json_basename>_best.png`

"""
import argparse
import sys
from pathlib import Path
import os

# Ensure repo root on sys.path so imports work when running from repo root
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.visualization import TSPVisualizer


def find_json_files(results_dir, pattern="*.json"):
    p = Path(results_dir)
    if not p.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    return sorted([f for f in p.glob(pattern) if f.is_file()])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot best trajectories for result JSONs")
    parser.add_argument('--results-dir', required=True, help='Directory containing result JSON files')
    parser.add_argument('--data-dir', default='data/problem_instances', help='TSPLIB data directory')
    parser.add_argument('--outdir', default=None, help='Directory to save figures; defaults to <results-dir>/outputs')
    parser.add_argument('--pattern', default='*.json', help='Glob pattern to select files (default: *.json)')
    parser.add_argument('--show', action='store_true', help='Show plots interactively (default: save only)')

    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir) if args.outdir else results_dir / 'outputs'
    outdir.mkdir(parents=True, exist_ok=True)

    viz = TSPVisualizer()

    files = find_json_files(results_dir, args.pattern)
    if not files:
        print(f"No files found in {results_dir} matching pattern {args.pattern}")
        return 1

    for f in files:
        try:
            basename = f.stem
            savepath = outdir / f"{basename}_best.png"
            print(f"Plotting {f} -> {savepath}")
            # visualization will save the figure if savepath provided; pass show based on args
            res = viz.plot_best_trajectory_from_results(str(f), data_dir=args.data_dir, savepath=str(savepath), show=args.show)
            # In case the visualization did not save (it does), ensure save when returned fig
            if res and not args.show:
                fig, ax = res
                fig.savefig(savepath, bbox_inches='tight')

        except Exception as e:
            print(f"Error plotting {f}: {e}")

    print(f"Done. Figures saved to {outdir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
