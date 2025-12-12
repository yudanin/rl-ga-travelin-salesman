"""
Genetic algorithm implementation for TSP optimization.

Following Ruan et al.'s Algorithm 2 with enhancements from proposal.

Algorithm:

1 Set the parameters: Population Size,Number of Generations, Mutation Probability
2 Initialize populations based on population size
3 repeat
4      Calculation of fitness based on the fitness function
5      Selection operation
6      if Apply Crossover operator then
7           Perform crossover operations
8      if Apply Mutation operator then
9           Perform mutation operations
10    if current number of generations >target number of generations then
11.         break;
12 until end ;
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional


class TSPGeneticAlgorithm:
    """Genetic algorithm solver for TSP following Ruan et al. Algorithm 2"""

    def __init__(self, population_size: int, mutation_rate: float, selection_method: str):
        """
        Initialize GA with parameters from Ruan et al.

        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation (Ruan et al. used 0.01)
            selection_method: 'roulette_wheel' or 'elitist'
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.fitness_history = []
        self.best_route = None
        self.best_distance = float('inf')

    def initialize_population(self, tsp_instance: Dict, initial_routes: List[List[int]]) -> List[List[int]]:
        """
        Initialize population with provided routes only.

        Args:
            tsp_instance: TSP instance data with 'dimension' and 'coordinates'
            initial_routes: Initial routes from RL component (required)

        Returns:
            List of routes (each route is a list of city indices)
        """
        if not initial_routes:
            raise ValueError("initial_routes must be provided")

        if len(initial_routes) < self.population_size:
            raise ValueError(f"Need at least {self.population_size} initial routes, got {len(initial_routes)}")

        # Use only the provided routes (up to population size)
        population = []

        for route in initial_routes[:self.population_size]:
            route_copy = route.copy()

            # Ensure first and last city is 1
            if route_copy[0] != 1:
                print(f"!!! Route starts with {route_copy[0]}, should start with 1")
            last_index = len(route_copy) - 1
            if route_copy[last_index] != 1:
                print(f"!!! Route ends with {route_copy[last_index]}, should end with 1")

            population.append(route_copy)

        return population

    def calculate_fitness(self, route: List[int], distance_matrix: np.ndarray) -> float:
        """
        Calculate fitness based on total route distance.

        Args:
            route: List of city indices (1-indexed)
            distance_matrix: Distance matrix (0-indexed)

        Returns:
            Fitness value (inverse of total distance)
        """
        total_distance = 0
        n = len(route)

        for i in range(n-1):
            # Convert to 0-indexed for distance matrix
            start_idx = route[i] - 1
            end_idx = route[i + 1] - 1
            total_distance += distance_matrix[start_idx, end_idx]

        # Fitness is inverse of distance (higher fitness = shorter route)
        return 1.0 / total_distance if total_distance > 0 else 0

    def roulette_wheel_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Roulette wheel selection method:
        the probability of selecting an individual is directly proportional to its fitness,
        as in Ruan et al.

        Args:
            population: List of routes
            fitness_scores: Corresponding fitness values

        Returns:
            Selected route
        """
        total_fitness = sum(fitness_scores)

        if total_fitness == 0:
            return random.choice(population)

        # Spin the roulette wheel
        spin = random.uniform(0, total_fitness)
        current_sum = 0

        for i, fitness in enumerate(fitness_scores):
            current_sum += fitness
            if current_sum >= spin:
                return population[i].copy()

        # Fallback (should not happen)
        print("roulette_wheel_selection: does not return anything")
        return population[-1].copy()

    def elitist_selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Elitist selection method through Tournament selection
        (enhancement to Ruan et al)

        Args:
            population: List of routes
            fitness_scores: Corresponding fitness values

        Returns:
            Best route from population
        """
        #best_idx = np.argmax(fitness_scores)
        #return population[best_idx].copy()
        tournament_size = 5
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()

    # def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    #     """
    #     Order Crossover (OX) for TSP - preserves relative order.
    #     A random crossover point is selected in both parents, and
    #     the portion of the route after the crossover point is
    #     exchanged between the two parents,
    #     as used by Ruan et al.
    #     Rtesdtircted to indices 1 to n-1 to make sure the route starts and ends in city 1.
    #
    #     Args:
    #         parent1: First parent route
    #         parent2: Second parent route
    #
    #     Returns:
    #         Two child routes
    #     """
    #     n = len(parent1)
    #     if n < 4:
    #         return parent1.copy(), parent2.copy()
    #
    #     # Choose two random crossover points between index 1 and n-1
    #     start, end = sorted(random.sample(range(1, n), 2))
    #
    #     # Create children
    #     child1 = [-1] * n
    #     child2 = [-1] * n
    #     child1[0] = parent1[0]
    #     child1[n-1] = parent1[n-1]
    #     child2[0] = parent2[0]
    #     child2[n-1] = parent2[n-1]
    #
    #     # Copy segments between crossover points
    #     child1[start:end + 1] = parent1[start:end + 1]
    #     child2[start:end + 1] = parent2[start:end + 1]
    #
    #     # Fill remaining positions maintaining order
    #     def fill_child(child, other_parent, segment_start, segment_end):
    #         # Get cities not in the copied segment
    #         segment_cities = set(child[segment_start:segment_end + 1])
    #
    #         # The 'remaining' list must contain cities from 'other_parent' that are:
    #         # 1. NOT in the copied segment
    #         # 2. NOT City 1 (since City 1 is fixed at child[0])
    #         remaining = [city for city in other_parent if city not in segment_cities and city != 1]
    #
    #         # Fill remaining positions in order
    #         pos = 0
    #         for i in range(1, n-1):
    #             if child[i] == -1:
    #                 child[i] = remaining[pos]
    #                 pos += 1
    #
    #     fill_child(child1, parent2, start, end)
    #     fill_child(child2, parent1, start, end)
    #
    #     return child1, child2

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Single-point crossover with repair to ensure no duplicate cities.
        A random crossover point is selected in both parents, and
        the portion of the route after the crossover point is
        exchanged between the two parents,
        as used by Ruan et al.
        Rtesdtircted to indices 1 to n-1 to make sure the route starts and ends in city 1.

        Args:
            parent1: First parent route
            parent2: Second parent route

        Returns:
            Two child routes
        """
        n = len(parent1)
        if n < 4:
            return parent1.copy(), parent2.copy()

        point = random.randint(1, n - 2) # do not touch firsty and last positions, which are city 1
        child1 = list(parent1[:point]) + list(parent2[point:])
        child2 = list(parent2[:point]) + list(parent1[point:])

        def repair(child):
            # Skip first and last position occupied by city 1
            seen = {child[0]}  # City 1 at the first position
            duplicates = []

            for i in range(1, n - 1):  # Skip first and last
                city = child[i]
                if city in seen:
                    duplicates.append(i)
                else:
                    seen.add(city)

            # Missing cities (excluding city 1 which is first and last)
            all_inner_cities = set(range(2, n))  # Cities 2 to n-1
            missing = [c for c in all_inner_cities if c not in seen]
            random.shuffle(missing)

            for i, pos in enumerate(duplicates):
                child[pos] = missing[i]

            return child

        return repair(child1), repair(child2)


    def insertion_mutation(self, route: List[int]) -> List[int]:
        """
        Insertion mutation by reversing the route
        between two randomly selected positions,
        as used by Ruan et al.
        Restricted to indices 1 to n-1 to make sure the route starts and ends in city 1.

        Args:
            route: Route to mutate

        Returns:
            Mutated route
        """
        if random.random() > self.mutation_rate:
            return route.copy()

        mutated = route.copy()
        n = len(mutated)
        if n < 4:
            return mutated

        # Choose two random positions from 1 to n-1 (i.e., range(1, n))
        pos1, pos2 = sorted(random.sample(range(1, n-1), 2))

        # Reverse the segment between positions
        mutated[pos1:pos2 + 1] = list(reversed(mutated[pos1:pos2 + 1]))

        return mutated

    def select_parents(self, population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
        """
        Select parents based on selection method.

        Args:
            population: Current population
            fitness_scores: Fitness values

        Returns:
            Selected parents for next generation
        """
        parents = []

        # Choose selection method
        if self.selection_method == 'elitist':
            selection_func = self.elitist_selection
        else:
            selection_func = self.roulette_wheel_selection

        # Select pairs of parents for crossover
        for _ in range(self.population_size // 2):
            parent1 = selection_func(population, fitness_scores)
            parent2 = selection_func(population, fitness_scores)

            # Ensure the parents are different
            attempts = 0
            while parent2 == parent1 and attempts < 10:
                parent2 = selection_func(population, fitness_scores)
                attempts += 1

            parents.extend([parent1, parent2])

        return parents

    def evolve(self, tsp_instance: Dict, generations: int, initial_routes: List[List[int]]) -> Dict:
        """
        Main evolution loop following Ruan et al. Algorithm 2.

        Args:
            tsp_instance: TSP instance data
            generations: Number of generations to evolve
            initial_routes: Initial routes from RL (required)

        Returns:
            Evolution results and best solution
        """
        distance_matrix = tsp_instance['distance_matrix']

        # Step 2: Initialize population
        population = self.initialize_population(tsp_instance, initial_routes)

        # Track evolution
        self.fitness_history = []
        generation_stats = []

        for generation in range(generations):
            # Step 4: Calculate fitness
            fitness_scores = []
            for route in population:
                fitness = self.calculate_fitness(route, distance_matrix)
                fitness_scores.append(fitness)

            # Track best solution
            best_gen_fitness = max(fitness_scores)
            best_gen_idx = fitness_scores.index(best_gen_fitness)
            best_gen_distance = 1.0 / best_gen_fitness if best_gen_fitness > 0 else float('inf')

            if best_gen_distance < self.best_distance:
                self.best_distance = best_gen_distance
                self.best_route = population[best_gen_idx].copy()

            # Store generation statistics
            avg_fitness = np.mean(fitness_scores)
            self.fitness_history.append(avg_fitness)
            generation_stats.append({
                'generation': generation,
                'best_fitness': best_gen_fitness,
                'avg_fitness': avg_fitness,
                'best_distance': 1.0 / best_gen_fitness if best_gen_fitness > 0 else float('inf')
            })

            # Step 10: Check termination condition
            if generation >= generations - 1:
                break

            # Save the best individual before crossover/mutation
            elite = population[best_gen_idx].copy()

            # Step 5: Selection operation
            parents = self.select_parents(population, fitness_scores)

            # Step 6-7: Crossover operations
            new_population = []

            # Apply crossover to pairs of parents
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1 = parents[i]
                    parent2 = parents[i + 1]

                    # Apply crossover
                    child1, child2 = self.order_crossover(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    # Odd number of parents, carry over last one
                    new_population.append(parents[i])

            # Step 8-9: Mutation operations
            mutated_population = []
            for individual in new_population:
                mutated = self.insertion_mutation(individual)
                mutated_population.append(mutated)

            #population = mutated_population[:self.population_size]
            population = [elite] + mutated_population[:self.population_size - 1]

        return {
            'best_route': self.best_route,
            'best_distance': self.best_distance,
            'fitness_history': self.fitness_history,
            'generation_stats': generation_stats,
            'final_population': population
        }