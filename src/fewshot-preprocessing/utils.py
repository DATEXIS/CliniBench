import itertools
import json
import logging
import os
from pathlib import Path

import pandas as pd

from src.exp_args import ExpArgs

# Define fine-tuned model mappings
FINE_TUNED_MODELS = {
    'CORe-clinibench': 'CORe-clinibench',
    'm2-bert-80M-32k-retrieval': 'm2-bert-80M-32k-retrieval',
    'diagnose_tuned': 'clinibench_dt',
    'triplet_tuned': 'clinibench_tt',
    'GatorTronS': 'GatorTronS',
}

PRED_TYPES = ['shot', 'bm25', 'pubmed-clinibench', 'random', 'gold']
TEST_LEN = 8


def init_notebook(depth=0):
    """Initialize working directory and logging for Jupyter notebooks."""
    top_level = os.getenv("REPO_ROOT", str(Path(os.getcwd()).parents[depth]))
    os.environ["REPO_ROOT"] = top_level
    os.chdir(top_level)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Working directory set to: {top_level}")


def get_model_name(model: str) -> str:
    """Extracts model name from path."""
    return os.path.basename(model)


def create_dir(directory, from_path=False):
    """Create directory if it doesn't exist."""
    if from_path:
        directory = Path(directory).parents[0]
    logging.info('Create directory %s', directory)
    os.makedirs(directory, exist_ok=True)


def load_parquet(file_path: Path, index_col=None, test_mode=False) -> pd.DataFrame:
    """Load a parquet file with optional test mode limiting results."""
    df = pd.read_parquet(file_path)
    if index_col:
        df.set_index(index_col, inplace=True)
    return df.head(TEST_LEN) if test_mode else df


def load_notes(args: ExpArgs, search_notes=False) -> pd.DataFrame:
    """Load patient notes based on retrieval type."""
    file_path = args.search_note_path if search_notes else args.query_note_path
    logging.info(f'Load notes from {file_path}')
    # Always load full set of search notes
    test_mode = args.test_mode and not search_notes
    return load_parquet(file_path, index_col='hadm_id', test_mode=test_mode)


def load_embeddings(args: ExpArgs, search_notes=False) -> pd.DataFrame:
    """Load embeddings based on retrieval type."""
    file_path = args.search_embedding_path if search_notes else args.embedding_path
    return load_parquet(file_path, test_mode=args.test_mode and not search_notes)


def extract_from_json(json_path: str, field='predicted_codes') -> pd.Series:
    """Extracts a specific field from a JSON file."""
    with open(json_path, 'r') as json_data:
        data = json.load(json_data)
        return pd.Series(data.get(field, []))


def load_json_results(json_path: Path, test_mode: bool) -> tuple[pd.DataFrame, list]:
    """Load JSON results as DataFrame and extract unique labels."""
    with open(json_path, 'r') as json_data:
        data = json.load(json_data)

    unique_codes = data.pop('unique_labels', [])
    data.pop('answer', None)
    data.pop('prompts', None)

    df = pd.DataFrame(data)
    return (df.head(TEST_LEN) if test_mode else df), unique_codes


def convert_code_to_short_code(icd_version: int, code: str) -> str:
    """Converts ICD codes to short format."""
    return code[:4] if icd_version == 9 and code.lower().startswith('e') else code[:3]


def convert_codes_to_short_codes(codes: list, icd_version: int) -> list:
    """Batch conversion of codes to short format."""
    return list(set([convert_code_to_short_code(icd_version, code) for code in codes]))


def convert_to_short_codes(codes: pd.Series, icd_version: int) -> pd.Series:
    """Apply short code conversion to a Pandas Series."""
    return codes.apply(lambda pred: convert_codes_to_short_codes(pred, icd_version))


def load_few_shots_for_experiment(args: ExpArgs) -> pd.Series:
    """Load few-shot ids for a given experiment."""
    few_shots = pd.read_parquet(args.few_shot_path)
    column_name = f"icd{args.icd}_{args.task}_few_shots"
    return few_shots[column_name].dropna().apply(lambda x: x[:args.num_few_shots])


def get_few_shots_labels(shot_lists: pd.Series, labels_dict: dict) -> pd.DataFrame:
    """Retrieve label sets for few-shot examples."""
    shots_df = shot_lists.apply(
        lambda shots: pd.Series(shots[:5]).apply(lambda hadm_id: set(labels_dict.get(hadm_id, [])))
    )
    shots_df.columns = [f'label_shot_{i + 1}' for i in range(5)]
    shots_df['label_shot_union'] = shots_df.apply(lambda row: set.union(*row), axis=1)
    return shots_df


def get_label_df(args: ExpArgs, label_type='SHORT_CODES') -> pd.DataFrame:
    """Load ICD labels and few-shot samples."""
    few_shots = load_few_shots_for_experiment(args)
    label_dict = load_notes(args, search_notes=True)['SHORT_CODES'].to_dict()
    label_df = pd.DataFrame({'labels': load_notes(args).reset_index()['SHORT_CODES']})
    label_df = pd.concat([label_df, get_few_shots_labels(few_shots, label_dict)], axis=1)

    if label_type == 'SHORT_CODES':
        label_df = label_df.apply(lambda col: convert_to_short_codes(col, args.icd), axis=1)
    else:
        label_df = label_df.map(list)

    if args.task == 'hosp' and args.icd == 9:
        label_df = label_df[label_df['labels'].str.len() > 0].reset_index()

    return label_df.head(TEST_LEN) if args.test_mode else label_df


def get_json_file_paths(args: ExpArgs) -> dict:
    """Generate JSON file paths for various prediction types and few-shot configurations."""
    paths = {'zeroshot': args.get_zeroshot_results_path(), 'cot': args.get_zeroshot_results_path('cot')}
    paths.update({
        f'{fs_type}-{shot}': args.get_fewshot_results_path(shot, fs_type)
        for fs_type, shot in itertools.product(PRED_TYPES, [1, 3, 5])
    })
    return paths
