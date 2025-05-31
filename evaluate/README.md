# Requirements

You might have to install `pandas` and `scikit-learn` for running the scripts. Furthermore, for fast inference, the
installation of `torch_cluster` is highly recommended.

# Usage

The scripts in this folder can be used for reproducing the results from our paper

For example, download the `wiggle150.h5` test dataset from [Zenodo](https://zenodo.org/records/15465665) into this
folder and run:

```bash
dataset="wiggle150"
ref_col=total_energy_dlpno_ccsdt_cbs
precision=32
batch_size=40
mkdir $dataset
# r2SCAN-3C fidelity:
xyz_output_r2scan3c=${dataset}/predictions_r2scan3c.xyz
csv_output_r2scan3c=${dataset}/predictions_r2scan3c.csv
confrankplus --files test_sets/${dataset}.h5 --output_path $xyz_output_r2scan3c --precision $precision --fidelity r2SCAN-3c --write_additional_energies True --batch_size $batch_size
python scripts/xyz_to_csv.py --input $xyz_output_r2scan3c --output $csv_output_r2scan3c
# wB07M-D3 fidelity:
xyz_output_wB97M_D3=${dataset}/predictions_wB97M_D3.xyz
csv_output_wB97M_D3=${dataset}/predictions_wB97M_D3.csv
confrankplus --files test_sets/${dataset}.h5 --output_path $xyz_output_wB97M_D3 --precision $precision --fidelity wB97M-D3 --write_additional_energies True --batch_size $batch_size
python scripts/xyz_to_csv.py --input $xyz_output_wB97M_D3 --output $csv_output_wB97M_D3
# Compute stats 
python scripts/get_stats.py --input $csv_output_r2scan3c $csv_output_wB97M_D3 --output_dir $dataset --ref_col $ref_col
```

`inference_on_all_testsets.sh` is a bash script for running inference on all datasets:

```bash
chmod +x inference_on_all_testsets.sh
./inference_on_all_testsets.sh
```
