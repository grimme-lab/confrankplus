import argparse
from ase.io import read
import csv


def extract_metadata_with_ase(input_xyz_file, output_csv_file):
    """
    Extracts metadata from an .xyz file using ASE and writes it to a CSV file.

    Args:
        input_xyz_file (str): Path to the input .xyz file.
        output_csv_file (str): Path to the output .csv file.
    """
    # Read all structures in the .xyz file
    structures = read(input_xyz_file, index=":")  # ':' reads all entries

    # List to store extracted metadata
    data = []

    # Iterate over each structure and extract metadata
    for atoms in structures:
        metadata = {}

        # Extract metadata from the ASE `info` dictionary
        for key, value in atoms.info.items():
            if key in ["ensbid", "confid", "total_charge"] or "energy" in key:
                metadata[key] = value

        data.append(metadata)

    # Collect all unique keys to ensure consistent column order
    all_keys = set()
    for row in data:
        all_keys.update(row.keys())

    # Write metadata to a CSV file
    with open(output_csv_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(data)

    print(f"Metadata successfully extracted to {output_csv_file}")


if __name__ == "__main__":
    # Use argparse to parse input and output paths
    parser = argparse.ArgumentParser(description="Extract metadata from an .xyz file and save it to a .csv file.")
    parser.add_argument("--input", type=str, help="Path to the input .xyz file.")
    parser.add_argument("--output", type=str, help="Path to the output .csv file.")

    # Parse the arguments
    args = parser.parse_args()

    # Pass the parsed arguments to the function
    extract_metadata_with_ase(args.input, args.output)
