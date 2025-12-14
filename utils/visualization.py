"""
Visualization and plotting utilities for TSP results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import pandas as pd
from typing import Dict, List
import sys
import os
import json
import re

# Path for TSPLIB loader
sys.path.append('../data')
try:
    from data.tsplib_loader import TSPLIBLoader
except ImportError:
    try:
        from tsplib_loader import TSPLIBLoader
    except ImportError:
        print("Warning: Could not import TSPLIBLoader")

class TSPVisualizer:
    """Visualization tools for TSP experiments and results."""
    
    def __init__(self, style='seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_route(self, cities, route, title="TSP Route"):
        """Plot TSP route on 2D city layout."""
        pass
    
    def plot_convergence(self, solution_history, title="Convergence Analysis"):
        """Plot algorithm convergence over iterations."""
        pass
    
    def plot_performance_comparison(self, results_dict, metric="route_length"):
        """Compare performance across different algorithms."""
        pass
    
    def plot_parameter_sensitivity(self, parameter_results):
        """Visualize parameter sensitivity analysis."""
        pass
    
    def plot_computation_time_analysis(self, timing_results):
        """Analyze and plot computation time comparisons."""
        pass
    
    def create_performance_heatmap(self, results_matrix, x_labels, y_labels):
        """Create heatmap for parameter combinations."""
        pass
    
    def animate_route_construction(self, construction_steps):
        """Create animation of route construction process."""
        pass
    
    def generate_publication_figures(self, experiment_results):
        """Generate publication-ready figures."""
        pass

    def plot_best_trajectory_from_results(self, results_json_path: str, data_dir: str = "data/problem_instances",
                                         instance_name: str = None, title: str = None, savepath: str = None,
                                         show: bool = True):
        """Load an experiments JSON file, find the trajectory with minimum cost, and plot it.

        Args:
            results_json_path: Path to a results JSON produced by the experiment runner.
            data_dir: Directory containing TSPLIB `.tsp` files (used by `TSPLIBLoader`).
            instance_name: Optional override of the instance name stored in results JSON.
            title: Optional plot title. If None, a sensible default is used.
            savepath: If provided, save the figure to this path.
            show: If True, call `plt.show()`; otherwise return the figure and axes.

        Returns:
            (fig, ax) tuple when `show` is False, otherwise None.
        """
        # Load results JSON
        try:
            with open(results_json_path, 'r') as fh:
                results = json.load(fh)
        except Exception as e:
            print(f"❌ Could not read results file {results_json_path}: {e}")
            return None

        inst = instance_name or results.get('instance')
        if not inst:
            print("❌ Instance name not found in results and no override provided")
            return None

        trajectories = results.get('trajectories', [])
        if not trajectories:
            print(f"❌ No trajectories found in {results_json_path}")
            return None

        # Find min-cost trajectory
        best = min(trajectories, key=lambda t: t.get('cost', float('inf')))
        best_cost = best.get('cost')
        best_route = best.get('route')  # expected to be 1-based node IDs

        if not best_route:
            print("❌ Best trajectory has no route data")
            return None

        # Load instance coordinates using TSPLIBLoader
        try:
            loader = TSPLIBLoader(data_dir=data_dir)
            data = loader.load_instance(inst)
            coords = data['coordinates']  # mapping: node_id (1-based) -> (x,y)
        except Exception as e:
            print(f"❌ Could not load instance {inst} from {data_dir}: {e}")
            return None

        # Build ordered lists of points from route
        xs = []
        ys = []
        labels = []
        for node in best_route:
            # route may already be 1-based; ensure int
            node_id = int(node)
            xy = coords.get(node_id)
            if xy is None:
                print(f"⚠️ Missing coordinates for node {node_id}; skipping")
                continue
            x, y = xy
            xs.append(x)
            ys.append(y)
            labels.append(node_id)

        if not xs:
            print("❌ No valid coordinates to plot")
            return None

        # Use Plotly for interactive, higher-quality visuals. Fall back to HTML if image export fails.
        try:
            import plotly.graph_objects as go
        except Exception:
            print("❌ Plotly is not installed. Please `pip install plotly` to use this function.")
            return None

        # Ensure cost is a float and round to 2 decimals for title
        try:
            cost_val = float(best_cost)
        except Exception:
            cost_val = best_cost

        plot_title = title or f"Best route for {inst} — cost={cost_val:.2f}"

        # Build Plotly figure: line + markers + labels
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='lines+markers+text',
            text=[str(l) for l in labels],
            textposition='top right',
            marker=dict(size=8),
            line=dict(width=2)
        ))

        fig.update_layout(
            title=plot_title,
            xaxis_title='X',
            yaxis_title='Y',
            showlegend=False,
            autosize=False,
            width=700,
            height=500,
            template='plotly_white'
        )

        # Save if requested. Prefer image (PNG) via kaleido; fallback to HTML if not available.
        if savepath:
            try:
                # Attempt to write an image (requires kaleido)
                fig.write_image(savepath)
                print(f"✅ Figure saved to {savepath}")
            except Exception as e:
                print(f"⚠️ Could not write image to {savepath} (image engine missing). Falling back to HTML: {e}")
                # Fallback: save HTML (append .html if same extension)
                html_path = savepath
                if not str(html_path).lower().endswith('.html'):
                    html_path = str(savepath) + '.html'
                try:
                    fig.write_html(html_path)
                    print(f"✅ Figure saved as HTML to {html_path}")
                except Exception as ee:
                    print(f"❌ Could not save fallback HTML to {html_path}: {ee}")

        if show:
            try:
                fig.show()
            except Exception as e:
                print(f"⚠️ Could not display figure inline: {e}")
            return None

        return fig

    def plot_convergence_comparison(self, results_dir: str, instance_name: str = None,
                                     savepath: str = None, show: bool = True):
        """Compare convergence rates between GA-only and RL+GA methods.

        Scans `results_dir` for JSON files, finds GA-only and RL+GA pairs that differ only in
        the method name (e.g., 'berlin52_ga_only_...' paired with 'berlin52_rl_ga_...'). For each
        pair, finds the best trajectory (lowest final cost) and plots its cost_per_gen convergence
        curve. Uses Plotly for interactive plots.

        Args:
            results_dir: Directory containing experiment result JSON files.
            instance_name: Optional filter to compare only a specific instance (e.g., 'berlin52').
                          If None, finds and plots all available pairs.
            savepath: If provided, save the figure to this path (PNG or fallback HTML).
            show: If True, display the figure; otherwise return the Plotly figure object.

        Returns:
            Plotly figure object if show=False, otherwise None.
        """
        try:
            import plotly.graph_objects as go
        except Exception:
            print("❌ Plotly is not installed. Please `pip install plotly` to use this function.")
            return None

        # Find all JSON files
        if not os.path.exists(results_dir):
            print(f"❌ Results directory not found: {results_dir}")
            return None

        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        if not json_files:
            print(f"❌ No JSON files found in {results_dir}")
            return None

        # Parse filenames to find GA-only and RL+GA pairs
        ga_only_files = {}
        rl_ga_files = {}

        for fname in json_files:
            if '_ga_only_' in fname:
                # Extract base name (everything before '_ga_only_')
                base = fname.split('_ga_only_')[0]
                ga_only_files[base] = fname
            elif '_rl_ga_' in fname:
                # Extract base name (everything before '_rl_ga_')
                base = fname.split('_rl_ga_')[0]
                rl_ga_files[base] = fname

        # Find matching pairs
        pairs = []
        for base_name in ga_only_files:
            if base_name in rl_ga_files:
                # Optional: filter by instance name
                if instance_name and not base_name.startswith(instance_name):
                    continue
                pairs.append((base_name, ga_only_files[base_name], rl_ga_files[base_name]))

        if not pairs:
            print(f"❌ No GA-only/RL+GA pairs found in {results_dir}" +
                  (f" for instance '{instance_name}'" if instance_name else ""))
            return None

        # If multiple pairs, just plot the first one (or all on subplots)
        # For simplicity, we'll plot all pairs on a single figure with subplots if there are many
        num_pairs = len(pairs)
        print(f"✅ Found {num_pairs} GA-only/RL+GA pair(s): {[p[0] for p in pairs]}")

        # Create Plotly subplots
        try:
            from plotly.subplots import make_subplots
        except Exception:
            print("❌ Could not import plotly.subplots")
            return None

        if num_pairs == 1:
            # Single pair: use a simple figure
            fig = go.Figure()
        else:
            # Multiple pairs: use subplots (2 columns, ceil(num_pairs/2) rows)
            rows = (num_pairs + 1) // 2
            cols = 2
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f"{p[0]}" for p in pairs],
                shared_yaxes=False,
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

        # Process each pair
        for pair_idx, (base_name, ga_file, rl_ga_file) in enumerate(pairs):
            ga_path = os.path.join(results_dir, ga_file)
            rl_ga_path = os.path.join(results_dir, rl_ga_file)

            try:
                with open(ga_path, 'r') as f:
                    ga_data = json.load(f)
                with open(rl_ga_path, 'r') as f:
                    rl_ga_data = json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load pair {base_name}: {e}")
                continue

            # Find best trajectory in each file (lowest final cost)
            ga_trajs = ga_data.get('trajectories', [])
            rl_ga_trajs = rl_ga_data.get('trajectories', [])

            if not ga_trajs or not rl_ga_trajs:
                print(f"⚠️ No trajectories found in pair {base_name}")
                continue

            ga_best = min(ga_trajs, key=lambda t: t.get('cost', float('inf')))
            rl_ga_best = min(rl_ga_trajs, key=lambda t: t.get('cost', float('inf')))

            ga_cost_per_gen = ga_best.get('cost_per_gen', [])
            rl_ga_cost_per_gen = rl_ga_best.get('cost_per_gen', [])

            if not ga_cost_per_gen or not rl_ga_cost_per_gen:
                print(f"⚠️ No cost_per_gen data for pair {base_name}")
                continue

            # Prepare data
            ga_gens = list(range(len(ga_cost_per_gen)))
            rl_ga_gens = list(range(len(rl_ga_cost_per_gen)))

            # Format costs for display
            ga_final_cost = float(ga_best.get('cost', 0))
            rl_ga_final_cost = float(rl_ga_best.get('cost', 0))

            # Determine subplot position
            if num_pairs == 1:
                row, col = None, None
            else:
                row = (pair_idx // cols) + 1
                col = (pair_idx % cols) + 1

            # Add traces
            if num_pairs == 1:
                fig.add_trace(go.Scatter(
                    x=ga_gens,
                    y=ga_cost_per_gen,
                    mode='lines',
                    name=f'GA-only (final cost={ga_final_cost:.2f})',
                    line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=rl_ga_gens,
                    y=rl_ga_cost_per_gen,
                    mode='lines',
                    name=f'RL+GA (final cost={rl_ga_final_cost:.2f})',
                    line=dict(width=2)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=ga_gens,
                    y=ga_cost_per_gen,
                    mode='lines',
                    name=f'GA-only (cost={ga_final_cost:.2f})',
                    line=dict(width=2),
                    showlegend=(pair_idx == 0)  # Only show legend for first trace
                ), row=row, col=col)
                fig.add_trace(go.Scatter(
                    x=rl_ga_gens,
                    y=rl_ga_cost_per_gen,
                    mode='lines',
                    name=f'RL+GA (cost={rl_ga_final_cost:.2f})',
                    line=dict(width=2),
                    showlegend=(pair_idx == 0)
                ), row=row, col=col)

                # Update axes for subplots
                fig.update_xaxes(title_text="Generation", row=row, col=col)
                fig.update_yaxes(title_text="Cost", row=row, col=col)

        # Update layout
        if num_pairs == 1:
            title = f"Convergence Comparison: GA-only vs RL+GA ({pairs[0][0]})"
            fig.update_layout(
                title=title,
                xaxis_title='Generation',
                yaxis_title='Cost',
                template='plotly_white',
                hovermode='x unified',
                width=800,
                height=500,
                showlegend=True
            )
        else:
            title = f"Convergence Comparison: GA-only vs RL+GA (All Instances)"
            fig.update_layout(
                title=title,
                template='plotly_white',
                hovermode='closest',
                width=1200,
                height=400 * ((num_pairs + 1) // 2),
                showlegend=True
            )

        # Save if requested
        if savepath:
            try:
                fig.write_image(savepath)
                print(f"✅ Figure saved to {savepath}")
            except Exception as e:
                print(f"⚠️ Could not write image to {savepath}. Falling back to HTML: {e}")
                html_path = savepath
                if not str(html_path).lower().endswith('.html'):
                    html_path = str(savepath) + '.html'
                try:
                    fig.write_html(html_path)
                    print(f"✅ Figure saved as HTML to {html_path}")
                except Exception as ee:
                    print(f"❌ Could not save fallback HTML to {html_path}: {ee}")

        if show:
            try:
                fig.show()
            except Exception as e:
                print(f"⚠️ Could not display figure inline: {e}")
            return None

        return fig


class TableBuilder:
    @staticmethod
    def build_table_TSP_instances(data_dir="../data/problem_instances", output_csv=None, print_table=True):
        """
        Build Table 1 using TSPLIBLoader to get all data directly.

        Structure:
        Instance    n    Optimal solution
        eil51      51         426

        Args:
            data_dir: Path to TSPLIB data directory
            output_csv: Optional path to save CSV file
            print_table: Whether to print formatted table

        Returns:
            pandas.DataFrame with table data
        """
        print("Building TSP Instances Table using TSPLIBLoader")
        print("=" * 55)

        # Initialize loader
        try:
            loader = TSPLIBLoader(data_dir=data_dir)
        except Exception as e:
            print(f"❌ Could not initialize TSPLIBLoader: {e}")
            return pd.DataFrame()

        # Find all .tsp files
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return pd.DataFrame()

        tsp_files = [f for f in os.listdir(data_dir) if f.endswith('.tsp')]
        if not tsp_files:
            print(f"❌ No .tsp files found in {data_dir}")
            return pd.DataFrame()

        print(f"✅ Found {len(tsp_files)} .tsp files in {data_dir}")

        table_data = []

        for tsp_file in sorted(tsp_files):
            instance_name = tsp_file.replace('.tsp', '')

            try:
                # Load instance using TSPLIBLoader
                data = loader.load_instance(instance_name)

                # Extract the data we need
                dimension = data['dimension']

                # Use TSPLIBLoader's internal method to get optimal value
                optimal_value = loader._get_optimal_value(instance_name)

                table_data.append({
                    'Instance': instance_name,
                    'n': dimension,
                    'Optimal solution': optimal_value if optimal_value else 'N/A'
                })

                status = f"{optimal_value}" if optimal_value else "No optimal value"
                print(f"  {instance_name:<12} | {dimension:>3} cities | {status:>12}")

            except Exception as e:
                print(f"  ❌ Error loading {instance_name}: {e}")
                continue

        if not table_data:
            print("❌ No valid TSP instances found")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(table_data)

        if print_table:
            TableBuilder.print_table1_formatted(df)

        if output_csv:
            TableBuilder.save_table1_csv(df, output_csv)

        return df

    @staticmethod
    def print_table1_formatted(df):
        """Print Table 1 in Ruan et al. paper format."""
        print(f"\n📋 TABLE 1 - TSP Instances")
        print("=" * 50)
        print(f"{'Instance':<12} | {'n':<3} | {'Optimal solution':<15}")
        print("-" * 50)

        for _, row in df.iterrows():
            print(f"{row['Instance']:<12} | {row['n']:<3} | {row['Optimal solution']:<15}")

        print("=" * 50)
        print("Note: n = number of cities")

    @staticmethod
    def save_table1_csv(df, filename):
        """Save Table 1 to CSV file."""
        table_df = df[['Instance', 'n', 'Optimal solution']].copy()

        try:
            table_df.to_csv(filename, index=False)
            print(f"✅ Table 1 saved to: {filename}")
        except Exception as e:
            print(f"❌ Could not save CSV: {e}")

    @staticmethod
    def parse_algorithm_description(json_file):
        """
        Parse algorithm description from filename using a mapping dictionary.
        """
        # Mapping: pattern -> display format (--value-- is replaced with extracted value)
        param_mappings = [
            # Algorithm type (order matters - check these first)
            ('rl_ga_', 'RL+GA'),
            ('rl_only_', 'RL'),
            ('ga_only_', 'GA'),
            # RL parameters
            ('rl_method_([a-z_]+?)_gamma', 'Method=--value--'),
            ('gamma_([\d.]+)', 'Gamma=--value--'),
            ('reward_type_(\d+)', 'Reward=--value--'),
            ('epsilon_type_(\d+)', 'Epsilon=--value--'),
            # GA parameters
            ('sel_roulette_wheel', 'Roulette'),
            ('sel_elitist', 'Elitist'),
            ('pop_(\d+)', 'Pop=--value--'),
            ('mut_([\d.]+)', 'Mut=--value--'),
        ]

        parts = []

        for pattern, display in param_mappings:
            if '--value--' in display:
                # Pattern has a capture group
                match = re.search(pattern, json_file)
                if match:
                    parts.append(display.replace('--value--', match.group(1)))
            else:
                # Simple pattern match
                if re.search(pattern, json_file):
                    parts.append(display)

        return ', '.join(parts) if parts else 'Unknown'

    @staticmethod
    def build_table_algorithm_comparison(results_dir="experiments", output_csv=None, print_table=True):
        """
        Build Table 2 comparison of algorithm results (similar to Ruan et al.).

        Columns:
        - TSP instances (optimal): eil51(426)
        - Algorithm: Description based on filename parameters
        - Min Cost: 445
        - Inaccuracy: 4.3%

        Args:
            results_dir: Directory containing JSON result files
            output_csv: Optional path to save CSV file
            print_table: Whether to print formatted table

        Returns:
            pandas.DataFrame with comparison data
        """
        print("Building Algorithm Comparison Table")
        print("=" * 55)

        # Find JSON result files
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

        if not json_files:
            print(f"❌ No JSON files found in {results_dir}")
            return pd.DataFrame()

        table_data = []

        for json_file in sorted(json_files):
            file_path = os.path.join(results_dir, json_file)

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                instance = data['instance']
                optimal_value = data['optimal_value']
                trajectories = data['trajectories']

                # Find minimum cost
                min_cost = min(traj['cost'] for traj in trajectories)

                # Calculate average cost
                avg_cost = sum(traj['cost'] for traj in trajectories) / len(trajectories)

                # Calculate inaccuracy percentage
                if optimal_value and optimal_value > 0:
                    inaccuracy = ((min_cost - optimal_value) / optimal_value) * 100
                else:
                    inaccuracy = None

                # Parse algorithm description from filename
                algorithm = TableBuilder.parse_algorithm_description(json_file)

                table_data.append({
                    'TSP instances (optimal)': f"{instance}({optimal_value})" if optimal_value else instance,
                    'Algorithm': algorithm,
                    'Min Cost': int(min_cost),
                    'Avg Cost': int(avg_cost),
                    'Inaccuracy': round(inaccuracy, 1) if inaccuracy is not None else 'N/A'
                })

            except Exception as e:
                print(f"❌ Error processing {json_file}: {e}")
                continue

        if not table_data:
            print("❌ No valid result files found")
            return pd.DataFrame()

        df = pd.DataFrame(table_data)

        if print_table:
            TableBuilder.print_table2_formatted(df)

        if output_csv:
            try:
                df.to_csv(output_csv, index=False)
                print(f"✅ Table 2 saved to: {output_csv}")
            except Exception as e:
                print(f"❌ Could not save CSV: {e}")

        return df

    @staticmethod
    def print_table2_formatted(df):
        """Print Table 2 in Ruan et al. paper format."""

        print(f"\n📋 TABLE 2 - Algorithm Comparison")
        print("=" * 130)
        print(
            f"{'TSP instances (optimal)':<25} | {'Algorithm':<70} | {'Min Cost':<8} | {'Avg Cost':<8} | {'Inaccuracy':<10}")
        print("-" * 130)

        for _, row in df.iterrows():
            inaccuracy_str = f"{row['Inaccuracy']}%" if row['Inaccuracy'] != 'N/A' else 'N/A'
            print(
                f"{row['TSP instances (optimal)']:<25} | {row['Algorithm']:<70} | {row['Min Cost']:<8} | {row['Avg Cost']:<8} | {inaccuracy_str:<10}")

        print("=" * 130)

