#!/bin/bash

declare -A datasets
datasets=(
  ["deprotogeom_test"]="total_energy_ref" # r2SCAN-3c ref
  ["qm9star_test"]="total_energy_ref"  # r2SCAN-3c ref
  ["spice_test"]="total_energy_ref"  # wB97M-D3 ref
  ["confclean"]="total_energy_lit"  # various ref (see paper)
  ["folmsbee"]="total_energy_ref"  # r2SCAN-3c ref
  ["wiggle150"]="total_energy_dlpno_ccsdt_cbs"  # DLPNO_CCSD(T)/CBS ref
)
precision=64 # paper results were generated with 64-bit precision during inference
batch_size=10
# Iterate through the datasets
for dataset in "${!datasets[@]}"; do
  ref_col=${datasets[$dataset]}


  echo "Processing dataset: $dataset with ref_col: $ref_col"

  if [ -d $dataset ]; then
    rm -r $dataset
  fi

  # Create a directory for the dataset
  mkdir -p $dataset

  # r2SCAN-3C fidelity
  xyz_output_r2scan3c=${dataset}/predictions_r2scan3c.xyz
  csv_output_r2scan3c=${dataset}/predictions_r2scan3c.csv
  confrankplus --files test_sets/${dataset}.h5 --output_path $xyz_output_r2scan3c --precision $precision --fidelity r2SCAN-3c --write_additional_energies True --batch_size $batch_size
  python scripts/xyz_to_csv.py --input $xyz_output_r2scan3c --output $csv_output_r2scan3c

  # wB07M-D3 fidelity
  xyz_output_wB97M_D3=${dataset}/predictions_wB97M_D3.xyz
  csv_output_wB97M_D3=${dataset}/predictions_wB97M_D3.csv
  confrankplus --files test_sets/${dataset}.h5 --output_path $xyz_output_wB97M_D3 --precision $precision --fidelity wB97M-D3 --write_additional_energies True --batch_size $batch_size
  python scripts/xyz_to_csv.py --input $xyz_output_wB97M_D3 --output $csv_output_wB97M_D3

  # Compute stats
  python scripts/get_stats.py --input $csv_output_r2scan3c $csv_output_wB97M_D3 --output_dir $dataset --ref_col $ref_col

  echo "Finished processing dataset: $dataset"
done