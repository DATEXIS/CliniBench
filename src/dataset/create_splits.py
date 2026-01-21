from argparse import ArgumentParser
import pandas as pd
from glob import glob
import os


def main():
    departments = ["hosp", "icu"]
    icd_versions = ["9", "10"]
    os.makedirs(os.path.abspath(args.save_dir), exist_ok=True)
    for icd_version in icd_versions:
        os.makedirs(
            os.path.abspath(args.save_dir) + "/" + f"icd-{icd_version}", exist_ok=True
        )
        for department in departments:
            os.makedirs(
                os.path.abspath(
                    args.save_dir) + "/" + f"icd-{icd_version}/{department}"
                ,
                exist_ok=True,
            )

    for department in departments:
        for icd_version in icd_versions:
            admission_note_dataset_filename = list(
                glob(
                    os.path.abspath(args.path_to_admission_notes)
                    + f"/icd-{icd_version}"
                    + f"/{department}"
                    + "/*.parquet",
                    recursive=True,
                )
            )[0]

            splits_filenames = list(
                glob(
                    os.path.abspath(args.path_to_split_ids)
                    + f"/icd-{icd_version}"
                    + f"/{department}"
                    + "/*.parquet",
                    recursive=True,
                )
            )

            admission_notes = pd.read_parquet(
                admission_note_dataset_filename
            ).set_index("hadm_id")

            for splits_filename in splits_filenames:
                split_ids = pd.read_parquet(splits_filename).set_index("hadm_id")
                split_ids = split_ids.join(admission_notes)
                split_ids = split_ids.rename(
                    columns={"icd_codes":"LONG_CODES", "short_codes": "SHORT_CODES", "text": "TEXT"})
                split_ids.to_parquet(os.path.abspath(
                    args.save_dir) + "/" + f"icd-{icd_version}/{department}" + "/" + os.path.basename(splits_filename))

    # create joined dataset from all training splits which will be use (hosp-icd9, hosp-icd10, icu-icd9, icu-icd10)
    # which will be used for the few-shot experiments
    all_datasets = []
    for filename in glob(os.path.abspath(
                    args.save_dir) + "/**/train.parquet", recursive=True):
        all_datasets.append(pd.read_parquet(filename))

    pd.concat(all_datasets).to_parquet(args.few_shot_note_save_path)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_admission_notes",
        required=True,
        help="Path to parquet files with the admission notes (directory above the icd versions)",
    )
    parser.add_argument(
        "--path_to_split_ids",
        required=True,
        help="Path to parquet files that contain the hadm_ids for the train/val/test splits",
    )
    parser.add_argument(
        "--save_dir", required=True, help="Directory to save the splits"
    )
    parser.add_argument(
        "--few_shot_note_save_path", required=True, help="Directory to save the few-shot notes consisting of all train splits"
    )
    args = parser.parse_args()
    main()
