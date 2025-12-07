#!/usr/bin/env python3
"""
Test the TSPLIB loader with downloaded files.
"""

from data.tsplib_loader import TSPLIBLoader


def test_loader():
    """Test the TSPLIB loader with all instances."""
    print("Testing TSPLIB Loader")
    print("=" * 30)

    # Use correct path from tests directory to data directory
    loader = TSPLIBLoader(data_dir="../data/problem_instances")
    instances = ['eil51', 'berlin52', 'st70', 'eil76', 'kroA100', 'pr107']

    for instance in instances:
        print(f"\n📋 Testing {instance}:")

        # Load instance
        data = loader.load_instance(instance)
        print(f"  ✅ Loaded: {data['dimension']} cities")
        print(f"  ✅ Optimal value: {data['optimal_value']}")

        # Load optimal tour
        tour = loader.load_optimal_tour(instance)
        if tour:
            print(f"  ✅ Tour loaded: {len(tour)} cities")

            # Calculate tour length
            length = loader.calculate_tour_length(tour, data['distance_matrix'])
            expected = data['optimal_value']
            difference = abs(length - expected)

            print(f"  📏 Calculated: {length}")
            print(f"  📏 Expected: {expected}")
            print(f"  📏 Difference: {difference}")

            if difference < 1:
                print(f"  🎉 PERFECT MATCH!")
            else:
                print(f"  ⚠️  Mismatch")
        else:
            print(f"  ❌ No tour loaded")

    print(f"\n✅ Loader testing complete!")


if __name__ == "__main__":
    test_loader()