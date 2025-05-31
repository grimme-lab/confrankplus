import os
import itertools
import statistics
from typing import List, Optional, Dict, Union
from warnings import warn

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score as sklearn_r2_score
from pandas import DataFrame, Series, concat
from tqdm import tqdm

import pandas as pd
from ase.io import read


def create_path_if_not_exists(path_str):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)


def create_pair_df(df: DataFrame, keys: Optional[List[str]] = None) -> DataFrame:
    """
    Function for transforming single-point energies to pairwise energy differences.
    Note that ensembles with only single samples will be dropped.
    :param df: DataFrame storing energies for different methods. Must have keys ["ensbid", "confid"].
    :param keys: Optional List of keys to specify the columns for which differences should be computed.
    :return: DataFrame with pairwise energy differences over all ensembles
    """
    if keys is not None:
        energy_keys = keys
    else:
        energy_keys = [key for key in df.keys() if "energy" in key]

    pairwise_energies = {key: [] for key in energy_keys}
    confid_1 = []
    confid_2 = []
    ensbids = []
    total_charge = []
    df["total_charge"] = df["total_charge"].astype(dtype=np.int64)
    grouped_df = df.groupby(["ensbid", "total_charge"])
    only_one = []
    for ensbid, ensemble_df in tqdm(grouped_df):
        ensemble_df = ensemble_df.sort_values(by="confid")
        pairs = np.array(list(itertools.combinations(range(len(ensemble_df)), 2)))
        if len(ensemble_df) == 1:
            only_one.append(ensbid)
            continue
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        diff_row = (
                ensemble_df[energy_keys].iloc[idx_i].values
                - ensemble_df[energy_keys].iloc[idx_j].values
        )
        confid_1.extend(ensemble_df["confid"].iloc[idx_i].values.tolist())
        confid_2.extend(ensemble_df["confid"].iloc[idx_j].values.tolist())
        assert np.allclose(
            ensemble_df["total_charge"].iloc[idx_i].values,
            ensemble_df["total_charge"].iloc[idx_j].values,
        ), "Found pair with different total charges."
        total_charge.extend(ensemble_df["total_charge"].iloc[idx_j].values.tolist())
        ensbids.extend(ensemble_df["ensbid"].iloc[idx_j].values.tolist())
        for i, key in enumerate(energy_keys):
            energy_diff = diff_row[:, i]
            pairwise_energies[key].append(energy_diff)
    for key, val in pairwise_energies.items():
        pairwise_energies[key] = np.concatenate(pairwise_energies[key])
    pairwise_energies["confid_1"] = confid_1
    pairwise_energies["confid_2"] = confid_2
    pairwise_energies["ensbid"] = ensbids
    pairwise_energies["total_charge"] = total_charge
    return DataFrame(pairwise_energies)


def calc_stats(df: DataFrame, ref_col: str = "energy_ref") -> DataFrame:
    """
    Calculate statistical deviations for value columns relative to a reference column.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the data.
    ref_col : str, optional
        The reference column name (default is "energy_ref").

    Returns
    -------
    DataFrame
        A DataFrame containing the calculated statistics: median deviation (md),
        mean absolute error (mae), median absolute deviation (mad), root mean
        square deviation (rmsd), and standard deviation (std).
    """

    # do not modify in-place
    df0 = df.copy()

    # automatically infer value columns
    pair_columns = ["ensbid", "confid", "confid_1", "confid_2", "total_charge", ref_col]
    value_columns = [col for col in df.columns if col not in pair_columns]

    # get deviations
    for col in value_columns:
        df0[f"d-{col}"] = df0[col] - df0[ref_col]

    # get ensembles
    df["total_charge"] = df["total_charge"].astype(dtype=np.int64)
    ensembles = df0.groupby(["ensbid", "total_charge"])
    ensembles = {group: dfi for group, dfi in ensembles}

    # calc stats
    stats = {}
    for col in value_columns:
        t1, t3, t5 = get_topX_scores(ensembles, col, ref_col)
        stats[col] = {
            # deviation metrics
            "WTMAD-2": wtmad2(df0, col=col, col_ref=ref_col),
            "md": md(df0[f"d-{col}"]),
            "mae": mae(df0[f"d-{col}"]),
            "mad": mad(df0[f"d-{col}"]),
            "rmsd": rmsd(df0[f"d-{col}"]),
            "std": std(df0[f"d-{col}"]),
            "std_pred": std(df0[f"{col}"]),  # "normal" stddev
            "R2": r2_score(df0, col, ref_col),
            "SF": signflip(df0, col, ref_col),
            "Top1": t1,
            "Top3": t3,
            "Top5": t5,
            "EW0.1": energy_window(ensembles, col, ref_col, 0.1),
            "EW1": energy_window(ensembles, col, ref_col, 1),
            "EW3": energy_window(ensembles, col, ref_col, 3),
            "EW5": energy_window(ensembles, col, ref_col, 5),
            "EW10": energy_window(ensembles, col, ref_col, 10),
            r"$\rho$": calc_avg_corr(ensembles, "pearson", col, ref_col),
            r"$\rho_s$": calc_avg_corr(ensembles, "spearman", col, ref_col),
            r"$\tau$": calc_avg_corr(ensembles, "kendall", col, ref_col),
            "num_pairs": len(df0),
            "num_ensembles": len(ensembles),
            "smallest_ensemble": min([len(df) for ens, df in ensembles.items()]),
            "largest_ensemble": max([len(df) for ens, df in ensembles.items()]),
            "median_ensemble": statistics.median(
                [len(df) for ens, df in ensembles.items()]
            ),
        }
    return DataFrame.from_dict(stats, orient="index")


############### SOME METRICS ###########################

def _convert(values: np.ndarray | list) -> np.ndarray:
    """Convenience function to convert to numpy."""
    if isinstance(values, list):
        return np.array(values)
    return values


def get_deviation(
        predictions: np.ndarray | list, targets: np.ndarray | list
) -> np.ndarray:
    """Get deviation from prediction and targets."""
    predictions = _convert(predictions)
    targets = _convert(targets)
    return predictions - targets


def md(values: np.ndarray):
    """Mean deviation."""
    return np.mean(values)


def mae(values: np.ndarray):
    """Mean absolute error."""
    # NOTE: often referred to as `MAD`
    return np.mean(np.abs(values))


def mad(values: np.ndarray):
    """Mean absolute deviation."""
    return np.mean(np.abs(values - np.mean(values)))


def rmsd(values: np.ndarray):
    """Root mean square deviation."""
    return np.sqrt(np.mean(np.square(values)))


def std(values: np.ndarray):
    """Standard deviation."""
    return np.std(values)


def r2_score(df: DataFrame, col: str, col_ref: str) -> float:
    # NOTE: R2 is not symmetric R2(a,b) != R2(b,a)
    # NOTE: not working with NaNs
    return sklearn_r2_score(df[col_ref], df[col])


def signflip(df: DataFrame, col: str, col_ref: str) -> float:
    """Return fraction of rows where sign of columns does not match."""
    selected_rows = df.loc[(df[col_ref] < 0) & (df[col] > 0)]
    selected_rows2 = df.loc[(df[col_ref] > 0) & (df[col] < 0)]
    ll = len(selected_rows) + len(selected_rows2)
    return ll / len(df)


def topX_score(df: DataFrame, n: int, col: str, col_ref: str) -> bool:
    """Check whether lowest reference value is within `n` lowest target values."""
    # find lowest DFT sample
    min_idx_ref = df[col_ref].idxmin()
    # find n-lowest rows of target column
    min_idx_trg = df[col].nsmallest(n).index
    return min_idx_ref in min_idx_trg


def get_topX_scores(ensembles: dict[DataFrame], col: str, col_ref: str):
    # for every ensemble, check whether DFT lowest is in ML 1,3,5 lowest
    t1, t3, t5 = 0, 0, 0
    for name, ensemble in ensembles.items():
        if topX_score(ensemble, 1, col, col_ref):
            t1 += 1
        if topX_score(ensemble, 3, col, col_ref):
            t3 += 1
        if topX_score(ensemble, 5, col, col_ref):
            t5 += 1
    t1 = t1 / len(ensembles)
    t3 = t3 / len(ensembles)
    t5 = t5 / len(ensembles)
    return t1, t3, t5


def calc_avg_corr(ensembles: dict[DataFrame], method: str, col: str, col_ref: str):
    """Calculate the ensemble-averaged correlation coefficient.
    Supported methods are: `pearson`, `spearman`, `kendall`
    """
    columns_to_keep = [col, col_ref]
    avg = 0.0
    for name, ensemble in ensembles.items():
        if len(ensemble) == 1:
            continue
        ens = ensemble[columns_to_keep]
        corr = ens.corr(method=method)
        avg += corr.loc[col, col_ref].item()
    return avg / len(ensembles)


def wtmad2(df, col: str, col_ref: str):
    if not ("confid_1" in df.columns and "confid_2" in df.columns):
        return np.nan
    delta_e_ref_total = df[col_ref].abs().mean()
    num_pairs_total = len(df)
    ensembles = df.groupby(["ensbid", "total_charge"])
    contribs = []
    for name, ensemble in ensembles:
        delta_e_ref = ensemble[col_ref].abs().mean()
        wt_ad = (
                len(ensemble)
                * delta_e_ref_total
                / delta_e_ref
                * mad(ensemble[col_ref] - ensemble[col])
        )
        contribs.append(wt_ad)
    result = 1.0 / num_pairs_total * np.sum(contribs)
    return result


def energy_window(ensembles: dict[DataFrame], col: str, col_ref: str, window: float):
    """
    Calculate energy window rate, i.e. percentage of how often the method
    includes the 'true' lowest lying conformer in a given energy window.
    """
    cnt = 0
    for name, ensemble in ensembles.items():
        # find lowest-lying FF and lowest-lying DFT
        elow = ensemble[col].min()
        # find idx of minimum value in ref
        min_index = ensemble[col_ref].idxmin()
        eref = ensemble.loc[min_index, col]
        delta = eref - elow
        if delta <= window:
            cnt += 1
    return cnt / len(ensembles)


# Inherit from this class for special plots
class Results:
    """
    A flexible base class for loading results from CSV files that store predictions for conformer energies.

    Assume that each CSV has keys "ensbid", "confid" and "total_charge" in any case.
    Furthermore, it should report model prediction and reference values for each conformer.

    Example of how a typical CSV file with model results will look like:

    | ensbid | confid | confrank_energy | total_energy_gfn2 | total_energy | total_energy_ref |
    |--------|--------|------------------|--------------------|--------------|-------------------|
    | "e23"  | "001"  | -100.0324        | -3.4123            | -40234.4412  | -120.8896         |
    | ...    | ...    | ...              | ...                | ...          | ...               |

    Different models will have different keys, e.g., aimnet2_energy, mace_off23_small_energy, so3lr_energy, etc.,
    but usually the reference energies (total_energy_ref, total_energy_gfn2, ...) will be reported in any case.
    """

    def __init__(
            self,
            result_paths: List[str],
            output_dir: str,
            rename_dict: Optional[dict] = None,
            dropna: bool = False,
    ):
        self.dropna = dropna
        self.result_paths = result_paths
        self.rename_dict = rename_dict
        self.point_wise_df = self.get_pointwise_df()
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    def charges_available(self):
        charges = pd.unique(self.point_wise_df["total_charge"]).tolist()
        return charges

    def get_columns(self):
        return list(self.point_wise_df.columns)

    def get_pointwise_df(self) -> DataFrame:
        df_list = []
        for path in self.result_paths:
            df = pd.read_csv(path, dtype={"ensbid": str, "confid": str})
            if self.rename_dict:
                df.rename(columns=self.rename_dict, inplace=True)
            df.reset_index()
            df.set_index(["ensbid", "confid"], inplace=True)
            df_list.append(df)
        combined_df = pd.concat(df_list, axis=1, join="outer")
        combined_df = combined_df.loc[
                      :, ~combined_df.columns.duplicated()
                      ].reset_index()
        combined_df["total_charge"] = combined_df["total_charge"].astype(np.int64)
        # reorder columns of necessary:
        if self.rename_dict:
            rename_dict_vals = list(self.rename_dict.values())
            remaining_cols = [
                col for col in combined_df.columns if col not in rename_dict_vals
            ]
            new_order = rename_dict_vals + remaining_cols
            combined_df = combined_df[new_order]

        if self.dropna:
            if combined_df.isnull().values.any():
                initial_row_count = combined_df.copy().shape[0]
                print("Found NaN in dataframe: Remove those rows")
                combined_df = combined_df.dropna()
                final_row_count = combined_df.shape[0]
                rows_removed = initial_row_count - final_row_count
                print(f"Rows removed: {rows_removed}")
                print(f"Total rows initially: {initial_row_count}")
        return combined_df

    def get_pairwise_df(self, total_charge: Optional[List[int]] = None) -> DataFrame:
        """
        :param total_charge: Optional list of total_charges for filtering. If None, not filtering is performed.
        :return: Data Frame with energy differences for conformers. The keys of the pointwise dataframes are kept.
        That is, for example, 'total_energy_ref' stores the difference of 'total_energy_ref' for the corresponding pair.
        Always has keys ['ensbid', 'confid_1', 'confid_2', 'total_charge']
        """
        pointwise_df = self.point_wise_df.copy()
        if total_charge is not None:
            pointwise_df = pointwise_df[pointwise_df["total_charge"].isin(total_charge)]
        keys = [
            key
            for key in pointwise_df.keys()
            if key not in ["ensbid", "confid", "total_charge"]
        ]
        pair_df = create_pair_df(pointwise_df, keys=keys)
        first_colums = ["ensbid", "confid_1", "confid_2", "total_charge"]
        pair_df = pair_df[
            first_colums + [col for col in pair_df.columns if col not in first_colums]
            ]
        return pair_df

    def save_raw_data(self):
        pointwise_df_raw = self.get_pointwise_df()
        pairwise_df_raw = self.get_pairwise_df()
        path_pointwise = os.path.join(self.output_dir, "raw_data", "pointwise_raw.csv")
        path_pairwise = os.path.join(self.output_dir, "raw_data", "pairwise_raw.csv")
        create_path_if_not_exists(path_pairwise)
        pointwise_df_raw.to_csv(path_pointwise)
        pairwise_df_raw.to_csv(path_pairwise)


class PairStatsTables(Results):

    def get_pair_stats(
            self,
            ref_col: str,
            total_charge: Optional[List[int]] = None,
            drop_rows: Optional[List[str]] = None,
            drop_cols: Optional[List[str]] = None,
            rename_dict: Optional[Dict[str, str]] = None,
            save_as_csv: bool = False,
    ) -> DataFrame:
        """
        Calculate pair statistics and optionally print the LaTeX representation.

        :param ref_col: Reference column for statistics calculation.
        :param total_charge: List of total charges to filter by.
        :param drop_rows: List of rows to drop from the resulting DataFrame.
        :param drop_cols: List of columns to drop from the resulting DataFrame.
        :param rename_dict: Dictionary for renaming columns or elements in the first column.
        :return: DataFrame containing the calculated pair statistics.
        """

        pointwise_df = self.point_wise_df.copy().dropna(axis=1)
        if total_charge is not None:
            pointwise_df = pointwise_df[pointwise_df["total_charge"].isin(total_charge)]

        # remove ensembles with only a single entry
        # value_counts = pointwise_df['confid'].value_counts()
        # pointwise_df = pointwise_df[pointwise_df['confid'].isin(value_counts[value_counts > 1].index)]

        pair_df = self.get_pairwise_df(total_charge=total_charge).dropna(axis=1)
        _drop_rows = []
        if drop_rows is not None:
            _drop_rows += drop_rows

        pair_df = pair_df.drop(columns=_drop_rows)  # Drop specified rows.
        pointwise_df = pointwise_df.drop(columns=_drop_rows)

        # some metrics are computed on pointwise df and some on the pairwise df:
        full_pointwise_stats = calc_stats(pointwise_df, ref_col=ref_col)
        full_pair_stats = calc_stats(pair_df, ref_col=ref_col)

        pairwise_stats_columns = [
            "WTMAD-2",
            "md",
            "mae",
            "mad",
            "rmsd",
            "std",
            "std_pred",
            "R2",
            "SF",
            "num_pairs",
        ]
        pointwise_stats_columns = [
            "Top1",
            "Top3",
            "Top5",
            "EW0.1",
            "EW1",
            "EW3",
            "EW5",
            "EW10",
            r"$\rho$",
            r"$\rho_s$",
            r"$\tau$",
            "num_ensembles",
            "smallest_ensemble",
            "largest_ensemble",
            "median_ensemble",
        ]

        keys_right_order = [
            "WTMAD-2",
            "md",
            "mae",
            "mad",
            "rmsd",
            "std",
            "std_pred",
            "R2",
            "SF",
            "Top1",
            "Top3",
            "Top5",
            "EW0.1",
            "EW1",
            "EW3",
            "EW5",
            "EW10",
            r"$\rho$",
            r"$\rho_s$",
            r"$\tau$",
            "num_pairs",
            "num_ensembles",
            "smallest_ensemble",
            "largest_ensemble",
            "median_ensemble",
        ]

        pair_stats = pd.concat(
            [
                full_pair_stats[pairwise_stats_columns],
                full_pointwise_stats[pointwise_stats_columns],
            ],
            axis=1,
        )

        pair_stats = pair_stats[keys_right_order]
        if drop_cols is not None:
            pair_stats = pair_stats.drop(columns=drop_cols)
        if rename_dict:
            pair_stats.rename(columns=rename_dict, inplace=True)
            if pair_stats.columns[0] in rename_dict:
                pair_stats.iloc[:, 0] = pair_stats.iloc[:, 0].replace(rename_dict)

        # save results:
        if save_as_csv:
            save_path = os.path.join(self.output_dir, "stats", "pair_stats.csv")
            create_path_if_not_exists(save_path)
            pair_stats.to_csv(save_path)
        return pair_stats

    def get_pair_stats_grouped_by_charge(
            self,
            ref_col: str,
            drop_rows: Optional[List[str]] = None,
            drop_cols: Optional[List[str]] = None,
            rename_dict: Optional[Dict[str, str]] = None,
            print_latex: bool = True,
            float_format="%.2f",
    ) -> DataFrame:
        """
        Calculate pair statistics grouped by total charge and return a single DataFrame.

        :param ref_col: Reference column for statistics calculation.
        :param drop_rows: List of rows to drop from the resulting DataFrames.
        :param drop_cols: List of columns to drop from the resulting DataFrames.
        :param rename_dict: Dictionary for renaming columns or elements in the first column.
        :return: DataFrame containing the calculated pair statistics grouped by total charge.
        """
        all_stats = []

        for c in self.charges_available():
            _stats = self.get_pair_stats(
                ref_col=ref_col,
                total_charge=[c],
                drop_rows=drop_rows,
                drop_cols=drop_cols,
                rename_dict=rename_dict,
            )
            _stats["total_charge"] = c

            all_stats.append(_stats)
        method_col_name = "method"
        grouped_df = (
            pd.concat(all_stats)
                .reset_index()
                .rename(columns={"index": method_col_name})
                .set_index(["total_charge", method_col_name])
        )

        save_path = os.path.join(
            self.output_dir, "stats", "pair_stats_grouped_by_charge.csv"
        )
        create_path_if_not_exists(save_path)
        grouped_df.to_csv(save_path)
        return grouped_df


if __name__ == "__main__":
    import argparse

    # Use argparse to parse input and output paths
    parser = argparse.ArgumentParser(description="Compute statistic based on csv file.")
    parser.add_argument("--input", type=str, nargs="+", help="Path to the input .csv file.")
    parser.add_argument("--ref_col", type=str, help="Column that is used as reference for metric computation.")
    parser.add_argument("--output_dir", type=str, help="Path to the output .csv file.")

    # Parse the arguments
    args = parser.parse_args()
    pairstats = PairStatsTables(result_paths=args.input, output_dir=args.output_dir)
    results = pairstats.get_pair_stats_grouped_by_charge(ref_col=args.ref_col, print_latex=False)
