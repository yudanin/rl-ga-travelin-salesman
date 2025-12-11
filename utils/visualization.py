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
        Build Table of TSP instances with cities and optimal solutions
        (Table 1 from Ruan et al.)

        Structure:
        Instance    n    Optimal solution
        eil51      51         426
        berlin52   52        7542

        Reads DIMENSION from .tsp files and calculates optimal distance from .opt.tour files.

        Args:
            data_dir: Path to TSPLIB data directory containing .tsp and .opt.tour files
            output_csv: Optional path to save CSV file
            print_table: Whether to print formatted table

        Returns:
            pandas.DataFrame with table data
        """
        print("Building TSP Instances Table")
        print("=" * 55)

        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return pd.DataFrame()

        # Find all .tsp files in the directory
        tsp_files = [f for f in os.listdir(data_dir) if f.endswith('.tsp')]

        if not tsp_files:
            print(f"❌ No .tsp files found in {data_dir}")
            return pd.DataFrame()

        print(f"✅ Found {len(tsp_files)} .tsp files in {data_dir}")

        table_data = []

        for tsp_file in sorted(tsp_files):
            instance_name = tsp_file.replace('.tsp', '')
            tsp_path = os.path.join(data_dir, tsp_file)
            tour_path = os.path.join(data_dir, f"{instance_name}.opt.tour")

            try:
                # Read dimension from .tsp file
                dimension = TableBuilder.read_tsp_dimension(tsp_path)

                # Calculate optimal distance from tour file
                optimal_distance = None
                if os.path.exists(tour_path):
                    optimal_distance = TableBuilder.calculate_optimal_distance(tsp_path, tour_path)

                table_data.append({
                    'Instance': instance_name,
                    'n': dimension,
                    'Optimal solution': optimal_distance if optimal_distance else 'N/A'
                })

                status = f"{optimal_distance}" if optimal_distance else "No tour file"
                print(f"  {instance_name:<12} | {dimension:>3} cities | {status:>12}")

            except Exception as e:
                print(f"  ❌ Error processing {instance_name}: {e}")
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
    def calculate_optimal_distance(tsp_path, tour_path):
        """
        Read the optimal distance from the COMMENT line in .opt.tour file.

        Example: "COMMENT : Optimal tour for eil51.tsp  (426)" -> returns 426

        Args:
            tsp_path: Path to .tsp file (not used, kept for compatibility)
            tour_path: Path to .opt.tour file

        Returns:
            int: Optimal tour distance from file
        """
        with open(tour_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('COMMENT'):
                    # Look for number in parentheses like "(426)"
                    import re
                    match = re.search(r'\((\d+)\)', line)
                    if match:
                        return int(match.group(1))

        raise ValueError(f"Optimal distance not found in {tour_path}")

    @staticmethod
    def read_tsp_dimension(tsp_path):
        """
        Read the DIMENSION from a .tsp file.

        Args:
            tsp_path: Path to .tsp file

        Returns:
            int: Number of cities (dimension)
        """
        with open(tsp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('DIMENSION'):
                    # Format: "DIMENSION: 52" or "DIMENSION : 52"
                    return int(line.split(':')[1].strip())

        raise ValueError(f"DIMENSION not found in {tsp_path}")

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
        # Select only the columns that appear in Ruan et al. table
        table_df = df[['Instance', 'n', 'Optimal solution']].copy()

        try:
            table_df.to_csv(filename, index=False)
            print(f"✅ Table 1 saved to: {filename}")
        except Exception as e:
            print(f"❌ Could not save CSV: {e}")


def main():
    """Test the table builder."""

    # Build Table 1 from actual files
    df = TableBuilder.build_table_TSP_instances(
        data_dir="../data/problem_instances",
        output_csv="../results/tsp_instances_table1.csv",
        print_table=True
    )


if __name__ == "__main__":
    main()

