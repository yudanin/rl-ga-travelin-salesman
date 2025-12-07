

# #!/usr/bin/env python3
# """
# Download and decompress TSPLIB files from Rice University.
# """
#
# import os
# import urllib.request
# import gzip
# import shutil
#
# # Create data directory
# os.makedirs('data/problem_instances', exist_ok=True)
#
# # Ruan et al.'s instances
# instances = ['eil51', 'berlin52', 'st70', 'eil76', 'kroA100', 'pr107']
# base_url = "https://softlib.rice.edu/pub/tsplib/tsp/"
#
#
# def download_file(filename):
#     """Download a file from Rice University."""
#     url = f"{base_url}{filename}"
#     output_path = f"data/problem_instances/{filename}"
#
#     print(f"Downloading {filename}...")
#     urllib.request.urlretrieve(url, output_path)
#
#     size = os.path.getsize(output_path)
#     print(f"✅ Downloaded {filename} ({size} bytes)")
#
#
# def decompress_file(gz_filename):
#     """Decompress a .gz file."""
#     gz_path = f"data/problem_instances/{gz_filename}"
#     output_filename = gz_filename[:-3]  # Remove .gz extension
#     output_path = f"data/problem_instances/{output_filename}"
#
#     print(f"Decompressing {gz_filename}...")
#
#     with gzip.open(gz_path, 'rb') as f_in:
#         with open(output_path, 'wb') as f_out:
#             shutil.copyfileobj(f_in, f_out)
#
#     size = os.path.getsize(output_path)
#     print(f"✅ Created {output_filename} ({size} bytes)")
#
#     # Remove .gz file
#     os.remove(gz_path)
#
#
# # Download and decompress all files
# print("Downloading TSPLIB files from Rice University...")
# print("=" * 50)
#
# for instance in instances:
#     print(f"\nProcessing {instance}:")
#
#     # Process .tsp file
#     tsp_file = f"{instance}.tsp.gz"
#     download_file(tsp_file)
#     decompress_file(tsp_file)
#
#     # Process .opt.tour file
#     tour_file = f"{instance}.opt.tour.gz"
#     download_file(tour_file)
#     decompress_file(tour_file)
#
# print("\n✅ All files downloaded and decompressed!")