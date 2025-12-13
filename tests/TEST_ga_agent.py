"""
Tests Genetic Algorithm
Uses instance-specific parameters and mock RL routes.
"""

import sys
import random
import numpy as np
import time
from typing import List, Dict

# Add paths for imports
sys.path.append('../data')
sys.path.append('../algorithms/genetic_algorithm')

from data.tsplib_loader import TSPLIBLoader
from algorithms.genetic_algorithm.ga_agent import TSPGeneticAlgorithm


class GATester:
    """Test suite following Ruan et al.'s methodology."""

    def __init__(self, data_dir="../data/problem_instances"):
        """Initialize tester with data directory."""
        self.loader = TSPLIBLoader(data_dir=data_dir)
        self.test_results = []

        # Ruan et al.'s standard GA parameters
        self.params = {
            'population_size': 100,
            'mutation_rate': 0.01,  # Ruan et al. used 0.01
            'selection_method': 'roulette_wheel',
            'generations': 2000  # As used in Ruan et al.
        }

    def generate_high_quality_mock_rl_routes(self, tsp_instance: Dict, num_routes: int = 50) -> List[List[int]]:
        """
        Generate high-quality mock RL routes simulating trained RL agent output.

        Args:
            tsp_instance: TSP instance data
            num_routes: Number of routes to generate (Ruan et al. used 40)

        Returns:
            List of high-quality mock RL routes
        """
        print(f"🤖 Generating {num_routes} high-quality mock RL routes (simulating trained agent)...")

        n_cities = tsp_instance['dimension']
        distance_matrix = tsp_instance['distance_matrix']
        routes = []
        optimal = tsp_instance['optimal_value']

        for i in range(num_routes):
            # Start with random route
            route = list(range(1, n_cities + 1))
            random.shuffle(route)

            # Apply extensive 2-opt improvements (simulating RL learning)
            # More improvements for larger instances
            max_improvements = min(50, n_cities * 2)
            improvements_made = 0

            for _ in range(max_improvements):
                best_improvement = 0
                best_i, best_j = 0, 0

                # Try multiple random 2-opt moves
                for _ in range(10):
                    i_pos, j_pos = sorted(random.sample(range(n_cities), 2))
                    if j_pos - i_pos < 2:
                        continue

                    # Calculate improvement
                    current_cost = (distance_matrix[route[i_pos] - 1, route[i_pos + 1] - 1] +
                                    distance_matrix[route[j_pos] - 1, route[(j_pos + 1) % n_cities] - 1])
                    new_cost = (distance_matrix[route[i_pos] - 1, route[j_pos] - 1] +
                                distance_matrix[route[i_pos + 1] - 1, route[(j_pos + 1) % n_cities] - 1])

                    improvement = current_cost - new_cost
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_j = i_pos, j_pos

                # Apply best improvement found
                if best_improvement > 0:
                    route[best_i + 1:best_j + 1] = reversed(route[best_i + 1:best_j + 1])
                    improvements_made += 1
                else:
                    # No improvement found, try random perturbation occasionally
                    if random.random() < 0.1:  # 10% chance of random move
                        i_pos, j_pos = sorted(random.sample(range(n_cities), 2))
                        if j_pos - i_pos > 1:
                            route[i_pos:j_pos + 1] = reversed(route[i_pos:j_pos + 1])

            # Additional Or-opt style moves (simulating advanced RL strategies)
            for _ in range(5):
                if n_cities > 10:
                    # Move a segment of cities to different position
                    seg_start = random.randint(0, n_cities - 3)
                    seg_length = random.randint(1, min(3, n_cities - seg_start - 1))
                    seg_end = seg_start + seg_length
                    insert_pos = random.randint(0, n_cities - seg_length)

                    if insert_pos != seg_start and insert_pos != seg_end:
                        segment = route[seg_start:seg_end]
                        remaining = route[:seg_start] + route[seg_end:]
                        route = remaining[:insert_pos] + segment + remaining[insert_pos:]

            routes.append(route)

        # Calculate and report quality
        distances = []
        for route in routes:
            total_distance = 0
            for k in range(n_cities):
                start_idx = route[k] - 1
                end_idx = route[(k + 1) % n_cities] - 1
                total_distance += distance_matrix[start_idx, end_idx]
            distances.append(total_distance)

        best_distance = min(distances)
        avg_distance = np.mean(distances)
        best_gap = ((best_distance - optimal) / optimal) * 100
        avg_gap = ((avg_distance - optimal) / optimal) * 100

        print(f"   ✅ Best mock RL route: {best_distance:.1f} (gap: {best_gap:.1f}%)")
        print(f"   ✅ Average mock RL route: {avg_distance:.1f} (gap: {avg_gap:.1f}%)")
        print(f"   📊 Quality simulation: RL agent producing routes {best_gap:.1f}-{avg_gap:.1f}% from optimal")

        return routes

    def test_instance_with_params(self, instance_name: str, custom_params: Dict = None) -> Dict:
        """
        Test single instance using Ruan et al.'s methodology.

        Args:
            instance_name: TSPLIB instance name
            custom_params: Optional parameter overrides

        Returns:
            Test results
        """
        print(f"\n📋 Testing {instance_name}")
        print("=" * 60)

        try:
            # Load TSP instance
            data = self.loader.load_instance(instance_name)
            optimal_tour = self.loader.load_optimal_tour(instance_name)

            print(f"   📊 Cities: {data['dimension']}")
            print(f"   📊 Optimal value: {data['optimal_value']}")

            # Use Ruan params with any custom overrides
            params = self.params.copy()
            if custom_params:
                params.update(custom_params)

            print(f"   🔧 Population: {params['population_size']}")
            print(f"   🔧 Mutation rate: {params['mutation_rate']}")
            print(f"   🔧 Generations: {params['generations']}")
            print(f"   🔧 Selection: {params['selection_method']}")

            # Generate high-quality mock RL routes (ensure we have enough)
            required_routes = max(params['population_size'], 50)
            mock_routes = self.generate_high_quality_mock_rl_routes(data, required_routes)

            # Initialize GA with Ruan et al.'s parameters
            ga = TSPGeneticAlgorithm(
                population_size=params['population_size'],
                mutation_rate=params['mutation_rate'],
                selection_method=params['selection_method']
            )

            # Run evolution
            start_time = time.time()
            print(f"\n   🧬 Running evolution for {params['generations']} generations...")
            result = ga.evolve(data, params['generations'], mock_routes)
            evolution_time = time.time() - start_time

            # Calculate metrics
            best_distance = result['best_distance']
            optimal_distance = data['optimal_value']
            gap_percent = ((best_distance - optimal_distance) / optimal_distance) * 100

            # Get initial mock route quality for comparison
            best_mock_distance = min([
                sum(data['distance_matrix'][route[i] - 1, route[(i + 1) % len(route)] - 1] for i in range(len(route)))
                for route in mock_routes[:10]  # Check first 10 mock routes
            ])
            improvement_percent = ((best_mock_distance - best_distance) / best_mock_distance) * 100

            print(f"\n   📊 RESULTS:")
            print(f"   🎯 Best GA distance: {best_distance:.1f}")
            print(f"   🎯 Best initial (mock RL): {best_mock_distance:.1f}")
            print(f"   🎯 Optimal distance: {optimal_distance:.1f}")
            print(f"   🎯 GA improvement over initial: {improvement_percent:.2f}%")
            print(f"   🎯 Gap from optimal: {gap_percent:.2f}%")
            print(f"   ⏱️ Evolution time: {evolution_time:.1f}s ({evolution_time / 60:.1f} min)")

            # Check convergence
            final_gen_stats = result['generation_stats'][-10:]  # Last 10 generations
            final_distances = [gs['best_distance'] for gs in final_gen_stats]
            convergence_improvement = final_distances[0] - final_distances[-1]
            print(f"   📈 Final convergence: {convergence_improvement:.1f} improvement in last 10 generations")

            return {
                'instance': instance_name,
                'cities': data['dimension'],
                'best_distance': best_distance,
                'optimal_distance': optimal_distance,
                'gap_percent': gap_percent,
                'improvement_percent': improvement_percent,
                'evolution_time': evolution_time,
                'generations': params['generations'],
                'converged': abs(convergence_improvement) < 1.0,  # Converged if < 1 unit improvement
                'success': True
            }

        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                'instance': instance_name,
                'success': False,
                'error': str(e)
            }

    def test_instances(self):
        """Test the main instances used by Ruan et al."""
        print(f"\n🌍 Testing Instances")
        print("=" * 70)

        # Instances
        instances = ["eil51", "berlin52", "st70", "eil76", "kroA100", "pr107"]
        results = []

        for instance in instances:
            try:
                result = self.test_instance_with_params(instance)
                if result['success']:
                    results.append(result)
                    self.test_results.append(result)
            except Exception as e:
                print(f"   ❌ Failed to test {instance}: {e}")

        # Summary comparison with Ruan et al. results
        print(f"\n📊 FINAL COMPARISON WITH RUAN ET AL.")
        print("=" * 80)
        print(f"{'Instance':<12} | {'Cities':<7} | {'Gap %':<8} | {'Time (min)':<10} | {'Converged':<10}")
        print("-" * 80)

        total_gap = 0
        converged_count = 0

        for result in results:
            converged_str = "✅ Yes" if result['converged'] else "❌ No"
            time_min = result['evolution_time'] / 60

            print(
                f"{result['instance']:<12} | {result['cities']:<7} | {result['gap_percent']:>6.2f}% | {time_min:>8.1f} min | {converged_str:<10}")

            total_gap += result['gap_percent']
            if result['converged']:
                converged_count += 1

        avg_gap = total_gap / len(results) if results else 0
        convergence_rate = (converged_count / len(results)) * 100 if results else 0

        print("-" * 80)
        print(f"Average gap from optimal: {avg_gap:.2f}%")
        print(f"Convergence rate: {convergence_rate:.1f}%")
        print(f"Expected gap (Ruan et al.): ~7.1% (reported in their Table 2)")

        if avg_gap < 20:
            print("🎉 Results are reasonable for GA-only (without real RL training)")
        else:
            print("⚠️ Gap suggests need for longer evolution or better initial routes")

        return results

    def test_enhanced_configurations(self):
        """Test the enhancements from your proposal."""
        print(f"\n🚀 Testing Proposal Enhancements")
        print("=" * 50)

        # Test enhanced configurations on eil51
        instance = "eil51"

        enhancements = [
            {
                'name': 'Elitist Selection',
                'params': {'selection_method': 'elitist'}
            },
            {
                'name': 'Small Population (20)',
                'params': {'population_size': 20}
            },
            {
                'name': 'Large Population (60)',
                'params': {'population_size': 60}
            },
            {
                'name': 'Higher Mutation (0.05)',
                'params': {'mutation_rate': 0.05}
            },
            {
                'name': 'Shorter Evolution (500 gen)',
                'params': {'generations': 500}
            }
        ]

        baseline_result = self.test_instance_with_params(instance)
        enhanced_results = []

        for enhancement in enhancements:
            print(f"\n--- Testing {enhancement['name']} ---")
            result = self.test_instance_with_params(instance, enhancement['params'])
            if result['success']:
                enhanced_results.append({
                    'name': enhancement['name'],
                    'gap': result['gap_percent'],
                    'time': result['evolution_time'],
                    'improvement': result['improvement_percent']
                })

        # Comparison
        print(f"\n📊 ENHANCEMENT COMPARISON (vs Baseline)")
        print("=" * 70)
        print(f"{'Enhancement':<25} | {'Gap %':<8} | {'vs Baseline':<12} | {'Time':<8}")
        print("-" * 70)

        baseline_gap = baseline_result['gap_percent'] if baseline_result['success'] else float('inf')

        for result in enhanced_results:
            vs_baseline = result['gap'] - baseline_gap
            vs_str = f"{vs_baseline:+.1f}%" if abs(vs_baseline) < 100 else f"{vs_baseline:+.0f}%"
            print(f"{result['name']:<25} | {result['gap']:>6.2f}% | {vs_str:<12} | {result['time']:>6.1f}s")

        return enhanced_results


def main():
    """Main test function following Ruan et al.'s methodology."""
    print("🧬 TSP GENETIC ALGORITHM TEST")
    print("=" * 70)
    print("Using proper parameters: 40 population, 0.01 mutation, 2000 generations")
    print("With high-quality mock RL routes simulating trained agent")
    print("=" * 70)

    random.seed(42)  # For reproducible results

    # Initialize tester
    tester = GATester()

    # Test main instances
    main_results = tester.test_instances()

    # Test enhancements
    print("\n" + "=" * 70)
    enhancement_results = tester.test_enhanced_configurations()

    print(f"\n🎉 COMPLETE TEST RESULTS")
    print("=" * 50)
    print(f"✅ Instances tested: {len(main_results)}")
    print(f"✅ Enhancements tested: {len(enhancement_results)}")

    if main_results:
        best_main = min(main_results, key=lambda x: x['gap_percent'])
        print(f"🏆 Best main result: {best_main['instance']} with {best_main['gap_percent']:.2f}% gap")

    if enhancement_results:
        best_enhancement = min(enhancement_results, key=lambda x: x['gap'])
        print(f"🏆 Best enhancement: {best_enhancement['name']} with {best_enhancement['gap']:.2f}% gap")

    print(f"\n📝 Note: Real RL-GA hybrid would require trained RL agent for better initial routes")


if __name__ == "__main__":
    main()