import json
import logging
import os
from argparse import ArgumentParser
from typing import List

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import pandas as pd
import txtai
from tqdm import tqdm
from tqdm.asyncio import tqdm

from metrics import evaluate, short_code_from_code

logging.getLogger().setLevel(logging.DEBUG)


def get_icd_codes_from_vector_index(
    predicted_diag_names: List,
    long_desc_index: txtai.Embeddings,
):
    logger.info("Map predictions to ICD codes using vector index")
    predicted_codes = []

    flattened_predicted_diag_names = [x for xs in predicted_diag_names for x in xs]
    num_predictions_per_note = len(predicted_diag_names[0])

    results_long = long_desc_index.batchsearch(flattened_predicted_diag_names, limit=1)

    partitioned_results_long = [
        results_long[x : x + num_predictions_per_note]
        for x in range(0, len(results_long), num_predictions_per_note)
    ]

    for long_results in partitioned_results_long:
        better_results = []
        for long_res in long_results:
            better_results.append(long_res[0][0])
        predicted_codes.append(better_results)
    logger.info("Queries successfully mapped")

    return predicted_codes


def build_icd_vector_indices(
    embedding_model_name, limit_label_space_codes: List[str] = None, icd_version=9
):
    logger.info("Build vector index based on: " + embedding_model_name)

    # build or load index over short and long names based on icd version
    embeddings_long = txtai.Embeddings(path=embedding_model_name, content=False)

    if icd_version == 9:
        icd_9_codes = pd.read_csv(args.icd9_code_file, dtype=str).to_dict()
        codes = icd_9_codes
    elif icd_version == 10:
        icd_10_codes = pd.read_csv(args.icd10_code_file, dtype=str).to_dict()
        codes = icd_10_codes

    limit_label_space_codes = [x for xs in limit_label_space_codes for x in xs]
    limit_label_space_codes = set(limit_label_space_codes)
    logger.info("Created set of all codes in the testset")
    diag_codes = [x.strip() for x in list(codes["icd_code"].values())]
    keep_idx = []

    for i, diag_code in tqdm(
        enumerate(diag_codes),
        desc="Create filter to filter out all ICD codes that are not present in the test set",
    ):
        if (
            short_code_from_code(diag_code, args.icd_version)
            not in limit_label_space_codes
        ):
            keep_idx.append(False)
        else:
            keep_idx.append(True)
    logger.info("Created filter")

    filtered_diag_codes = []
    filtered_short_descs = []
    filtered_long_descs = []
    all_diag_codes = [x.strip() for x in list(codes["icd_code"].values())]
    all_long_descs = list(codes["long_title"].values())
    for i in tqdm(range(0, len(diag_codes)), desc="Remove non present codes by filter"):
        if keep_idx[i]:
            filtered_diag_codes.append(all_diag_codes[i])
            filtered_long_descs.append(all_long_descs[i])
    codes = {
        "DIAGNOSIS CODE": filtered_diag_codes,
        "LONG DESCRIPTION": filtered_long_descs,
    }
    logger.info("Filtering complete")

    assert len(codes["DIAGNOSIS CODE"]) == len(codes["LONG DESCRIPTION"])
    embeddings_long.index(zip(codes["DIAGNOSIS CODE"], codes["LONG DESCRIPTION"]))

    logger.info("Vector indices initialized")

    return embeddings_long


def evaluate_cases():
    results = pd.read_parquet(args.predictions_file_path)

    labels = [x.tolist() for x in results["labels"].tolist()]
    prompts = results["prompts"].tolist()
    results = [x.tolist() for x in results["results"].tolist()]

    logger.info("Evaluate using sentence embeddings")
    icd_long_index = build_icd_vector_indices(
        args.embedding_model_name,
        limit_label_space_codes=labels,
        icd_version=args.icd_version,
    )

    # evaluate using embedding model
    predicted_codes = get_icd_codes_from_vector_index(results, icd_long_index)

    result_data = {
        "predicted_codes": [],
        "annotated_codes": [],
        "predicted_diagnoses": [],
        "prompts": [],
        "unique_labels": list(set(x for xs in labels for x in xs)),
    }
    for diag_list, pred_codes, ground_truth, prompt in zip(
        results, predicted_codes, labels, prompts
    ):
        if len(ground_truth) > 0:
            result_data["predicted_codes"].append(pred_codes)
            result_data["annotated_codes"].append(ground_truth)
            result_data["predicted_diagnoses"].append(diag_list)
            result_data["prompts"].append(prompt)

    result_data["icd_version"] = args.icd_version

    metrics = evaluate(result_data)
    logger.info("Successfully computed metrics")

    if args.save_mapped_results:
        metrics["mappings"] = result_data
    if os.path.dirname(args.results_save_path) != "":
        os.makedirs(os.path.dirname(args.results_save_path), exist_ok=True)

    with open(args.results_save_path, "w+") as f:
        json.dump(metrics, f)


def trim_unique_labels(data, icd_version=9):
    unique_codes = set()
    for code in data["unique_labels"]:
        unique_codes.add(short_code_from_code(code, icd_version=icd_version))
    annotated = []
    for example in data["annotated_codes"]:
        t_list = []
        for code in example:
            t_list.append(short_code_from_code(code, icd_version=icd_version))
        annotated.append(t_list)

    data["unique_labels"] = list(unique_codes)
    data["annotated_codes"] = annotated
    return data


def main():
    evaluate_cases()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--embedding_model_name",
        default="neuml/pubmedbert-base-embeddings",
        help="Model to use for the ICD retrieval index",
    )

    parser.add_argument(
        "--predictions_file_path",
        required=True,
        type=str,
        help="Path of the predictions file produced by the predict_diagnoses.py script",
    )

    parser.add_argument(
        "--test_data",
        default="/pvc/data/mimic-iv/admission_mimiciv_dia_fold_1_test.csv",
        help="Notes from the test dataset used for the evaluation",
    )

    parser.add_argument(
        "--icd9_code_file",
        default="data/icd9_codes.csv",
        help="Excel sheet that contains all ICD-9 codes and their respective short and long descriptions",
    )

    parser.add_argument(
        "--icd10_code_file",
        default="data/icd10_codes.csv",
        help="Excel sheet that contains all ICD-10 codes and their respective short and long descriptions",
    )

    parser.add_argument(
        "--logfile",
        default="clinibench_logs.log",
        help="Logfile. Make sure to log to different files if multiple processes run on the same distributed filesystem. Logging from multiple processes to the same file could lead to undefined behaviour.",
    )

    parser.add_argument(
        "--icd_version",
        type=int,
        choices=[9, 10],
        default=10,
        help="Which icd version to use to map against",
    )

    parser.add_argument(
        "--save_mapped_results",
        action="store_true",
        default=False,
        help="Store the mapped diagnosis codes together with the metrics",
    )

    parser.add_argument(
        "--results_save_path",
        default="/tmp/results/results.json",
        help="Path for the metrics and optionally the mapped ICD codes",
    )

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=args.logfile, encoding="utf-8", level=logging.DEBUG)

    main()
