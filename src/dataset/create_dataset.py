import logging
import operator
from itertools import combinations
from math import ceil, floor
from pathlib import Path
from typing import Iterable

import polars as pl
import typer
from polars import col, element, DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection.iterative_stratification import (
    IterativeStratification,
)
from tqdm import tqdm

import mimic_utils
from utils import log_function_call
from mimic_utils import ICDVersion, MIMICSplit

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)


@app.command()
def dia_groups_3_digits_mimic(
    save_dir: Path = Path("/tmp"),
    psql_connection_string: str = "postgresql://postgres:postgres@localhost:5432/mimiciv",
    icd_version: ICDVersion = ICDVersion.ten,
    split: MIMICSplit = MIMICSplit.hosp,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    group_using_icd_plus: bool = True,
    filter_labels: bool = False,
):
    """
    Extracts information needed for the task from the MIMIC dataset. Mainly, "text" column from `mimiciv_note.discharge`
    and "icd_code" from `mimiciv_hosp.diagnoses_icd`. Shorten all ICD codes' to the first three or four digits and group
    them per admission into column "short_codes".

    Args:
        group_using_icd_plus: Groups per default using an ICD+ like approach, creating a bag of words of all prefixes of the ICD codes for
        a more stratified sampling while splitting.
        filter_labels: Optionally filter labels if they don't appear in all three splits; when working with LLMs that could predict out of
        training-distribution codes this does not make sense and should be handled on the downstream task's end.
    """
    logger.info(locals())

    save_dir.mkdir(parents=True, exist_ok=True)

    mimic_split = read_split_from_db(icd_version, split, psql_connection_string)

    mimic_split = mimic_utils.extract_admission_notes(mimic_split)

    # create column SHORT_CODE including first 3 digits of ICD9 code
    mimic_split = shorten_icd_codes(mimic_split, icd_version)

    # split_dfs = iterative_sampling_3_splits_with_icd_plus(
    #    mimic_split, [train_fraction, val_fraction, test_fraction]
    # )
    # split_dfs = {split: df for split, df in zip(['train', 'val', 'test'], split_dfs)}

    # check_no_overlap(split_dfs, on='hadm_id')
    split_dfs = {"train": mimic_split}
    columns = ["icd_codes", "short_codes"]

    if filter_labels:
        split_dfs = filter_not_shared_labels(split_dfs, columns)
        split_dfs = filter_empty_diagnoses(split_dfs, columns)

    save_to_disk(save_dir, split_dfs, icd_version, split, columns)


@log_function_call(logger=logger)
def read_split_from_db(
    icd_version: ICDVersion, mimic_split: MIMICSplit, postgres_connection_string: str
):
    database_uri = postgres_connection_string

    sql_query = f"""
    SELECT discharge.subject_id, diagnoses.hadm_id, charttime, text, icd_codes 
    FROM (
        SELECT hadm_id, array_agg(icd_code::TEXT) as icd_codes
        FROM mimiciv_hosp.diagnoses_icd
        WHERE icd_version={icd_version}
        AND hadm_id {'NOT IN' if mimic_split == MIMICSplit.hosp else 'IN'} (
            SELECT DISTINCT (hadm_id) FROM mimiciv_icu.icustays)
        GROUP BY hadm_id
        HAVING count(icd_code) > 0)
    AS diagnoses
    INNER JOIN mimiciv_note.discharge ON discharge.hadm_id = diagnoses.hadm_id
    """

    return pl.read_database_uri(sql_query, database_uri)


def shorten_icd_codes(mimic_diagnoses: pl.DataFrame, icd_version: ICDVersion):
    if icd_version is ICDVersion.nine:
        mimic_diagnoses = mimic_diagnoses.with_columns(
            short_codes=col.icd_codes.list.eval(
                pl.when(element().str.starts_with("E"))
                .then(element().str.slice(0, 4))
                .otherwise(element().str.slice(0, 3))
            )
        )
    else:
        mimic_diagnoses = mimic_diagnoses.with_columns(
            short_codes=col.icd_codes.list.eval(element().str.slice(0, 3))
        )

    return mimic_diagnoses


def iterative_sampling_3_splits_with_icd_plus(
    combined_codes: pl.DataFrame, fractions: list[float]
) -> list[pl.DataFrame]:
    max_length = combined_codes["icd_codes"].explode().str.len_chars().max()
    expr = col.icd_codes.alias("icd_plus")
    for i in range(max_length, 0, -1):
        expr = expr.list.concat(col.icd_codes.list.eval(element().str.slice(0, i)))
    mimic_split = combined_codes.with_columns(expr.list.unique())

    return [
        df.drop("icd_plus")
        for df in iterative_sampling_3_splits(mimic_split, "icd_plus", fractions)
    ]


@log_function_call(logger=logger)
def iterative_sampling_3_splits(
    combined_codes: pl.DataFrame, on_column: str, fractions: list[float]
) -> list[DataFrame]:
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(combined_codes[on_column])

    rest_split = combined_codes
    splits = []
    for i in tqdm(range(len(fractions) - 1), desc=f"Splitting into {len(fractions)}"):
        if i != 0:
            remaining_fraction = 1 - fractions[0]
            fractions = [f / remaining_fraction for f in fractions[1:]]
        first_split, rest_split = _iterative_sampling(
            rest_split, mlb, on_column, fractions
        )
        splits.append(first_split)

    splits.append(rest_split)

    return splits


def _iterative_sampling(
    combined_codes: pl.DataFrame,
    mlb: MultiLabelBinarizer,
    on_column: str,
    fractions: list[float],
):
    stratifier = IterativeStratification(
        n_splits=len(fractions), sample_distribution_per_fold=fractions
    )
    y = mlb.transform(combined_codes[on_column])
    remaining_fractions_indices, first_fraction_indices = next(
        stratifier.split(combined_codes["hadm_id"], y)
    )
    return combined_codes[first_fraction_indices], combined_codes[
        remaining_fractions_indices
    ]


# alternative implementation using `iterative-stratification` package. Is faster, but can't work with sparse arrays
# and thus more memory intensive.
# def _iterative_sampling(combined_codes: pl.DataFrame, mlb: MultiLabelBinarizer, on_column: str, fractions: list[float]):
#     stratifier = MultilabelStratifiedShuffleSplit(n_splits=3, test_size=fractions[0])
#     y = mlb.transform(combined_codes[on_column])
#     remaining_fractions_indices, first_fraction_indices = next(stratifier.split(combined_codes['hadm_id'], y))
#     return combined_codes[first_fraction_indices], combined_codes[remaining_fractions_indices]


def filter_not_shared_labels(
    dfs: dict[str, pl.DataFrame], columns: Iterable[str]
) -> dict[str, DataFrame]:
    for column in columns:
        uniques = {
            split: set(df[column].explode().unique()) for split, df in dfs.items()
        }
        shared_labels = set.intersection(*uniques.values())

        _log_all_combinations(column, uniques)

        for split, df in dfs.items():
            dfs[split] = df.with_columns(
                col(column).list.filter(element().is_in(shared_labels))
            )

    return dfs


def _log_all_combinations(name, uniques):
    logging.info(_fill_up_with_dashes(f" {name} "))
    total_symmetric_diff = set()
    for split1, split2 in combinations(uniques.keys(), 2):
        set1 = uniques[split1]
        set2 = uniques[split2]

        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        symmetric_diff = only_in_1 | only_in_2  # all labels not shared
        total_symmetric_diff |= symmetric_diff

        logging.info(
            f"Comparing '{split1}' (len={len(set1)}) vs '{split2}' (len={len(set1)}):"
        )
        logging.info(f"    Labels only in {split1}: {len(only_in_1)}")
        logging.info(f"    Labels only in {split2}: {len(only_in_2)}")
        logging.info(f"    Total non-overlapping: {len(symmetric_diff)}")
    logging.info(
        _fill_up_with_dashes(f" All non-overlapping: {len(total_symmetric_diff)} ")
    )


def _fill_up_with_dashes(string: str, target_length=50) -> str:
    dashes = (target_length - len(string)) / 2
    return "-" * floor(dashes) + string + "-" * ceil(dashes)


def filter_empty_diagnoses(
    dfs: dict[str, pl.DataFrame], columns: Iterable[str]
) -> dict[str, pl.DataFrame]:
    conditions = [col(column).list.len() > 0 for column in columns]

    dfs = {
        split: df.filter(
            pl.reduce(operator.and_, conditions)  # all label columns must be non-empty
        )
        for split, df in dfs.items()
    }

    return dfs


def check_no_overlap(dfs: dict[str, DataFrame], on="hadm_id"):
    for (name1, df1), (name2, df2) in combinations(dfs.items(), 2):
        overlap = df1.join(df2, on=on, how="inner")
        if not overlap.is_empty():
            raise RuntimeError(
                f"Overlap detected between splits {name1} and {name2}!\n\n{len(overlap)}"
            )


def save_to_disk(
    save_dir: Path,
    dfs: dict[str, DataFrame],
    icd_version: ICDVersion,
    mimic_split: MIMICSplit,
    label_columns: Iterable[str],
):
    output_folder = save_dir / f"icd-{icd_version}/{mimic_split}"
    output_folder.mkdir(parents=True, exist_ok=True)
    for split, df in tqdm(dfs.items(), "Writing data splits"):
        df.write_parquet(output_folder / (split + ".parquet"))

    for column in tqdm(label_columns, desc="Writing label files"):
        exploded_labels = pl.concat([df[column].explode() for df in dfs.values()])
        frequencies = exploded_labels.value_counts(parallel=True, sort=True)
        frequencies.write_csv(output_folder / (column + ".csv"))


if __name__ == "__main__":
    app()
