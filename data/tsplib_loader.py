"""
TSPLIB instance loader and parser for TSP problems.
"""

import tsplib95
import numpy as np
from typing import Dict, List, Tuple, Optional


class TSPLIBLoader:
    """Clean, minimalistic TSPLIB loader for TSP instances and optimal solutions."""

    def __init__(self, data_dir: str = "data/problem_instances"):
        """Initialize loader with data directory path."""
        self.data_dir = data_dir

    def load_instance(self, instance_name: str) -> Dict:
        """
        Load a TSP instance and return standardized data structure.

        Args:
            instance_name: Name without extension (e.g., 'berlin52')

        Returns:
            Dict with keys: 'name', 'dimension', 'coordinates', 'distance_matrix', 'optimal_value'
        """
        tsp_path = f"{self.data_dir}/{instance_name}.tsp"

        # Load problem
        problem = tsplib95.load(tsp_path)

        # Extract coordinates
        coordinates = self._extract_coordinates(problem)

        # Build distance matrix
        distance_matrix = self._build_distance_matrix(problem, coordinates)

        # Try to get optimal value
        optimal_value = self._get_optimal_value(instance_name)

        return {
            'name': instance_name,
            'dimension': problem.dimension,
            'coordinates': coordinates,
            'distance_matrix': distance_matrix,
            'optimal_value': optimal_value,
            'edge_weight_type': problem.edge_weight_type
        }

    def load_optimal_tour(self, instance_name: str) -> Optional[List[int]]:
        """
        Load optimal tour if available.

        Args:
            instance_name: Name without extension

        Returns:
            List of city indices representing optimal tour, or None
        """
        tour_path = f"{self.data_dir}/{instance_name}.opt.tour"

        try:
            solution = tsplib95.load(tour_path)
            return solution.tours[0] if solution.tours else None
        except:
            return None

    def _extract_coordinates(self, problem) -> Dict[int, Tuple[float, float]]:
        """Extract city coordinates from problem."""
        return problem.node_coords

    def _build_distance_matrix(self, problem, coordinates: Dict[int, Tuple[float, float]]) -> np.ndarray:
        """Build distance matrix using problem's weight function."""
        n = problem.dimension
        nodes = list(coordinates.keys())

        # Create matrix with proper indexing
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i, j] = problem.get_weight(nodes[i], nodes[j])

        return distance_matrix

    def _get_optimal_value(self, instance_name: str) -> Optional[float]:
        """Get known optimal value for Ruan et al. instances."""
        optimal_values = {
            'eil51': 426,
            'berlin52': 7542,
            'st70': 675,
            'eil76': 538,
            'kroA100': 21282,
            'pr107': 44303
        }
        return optimal_values.get(instance_name)

    def get_node_list(self, coordinates: Dict[int, Tuple[float, float]]) -> List[int]:
        """Get sorted list of node indices."""
        return sorted(coordinates.keys())

    def calculate_tour_length(self, tour: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate total length of a tour."""
        if not tour:
            return float('inf')

        total_distance = 0
        n = len(tour)

        for i in range(n):
            start_idx = tour[i] - 1  # Convert to 0-based indexing
            end_idx = tour[(i + 1) % n] - 1
            total_distance += distance_matrix[start_idx, end_idx]

        return total_distance