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

