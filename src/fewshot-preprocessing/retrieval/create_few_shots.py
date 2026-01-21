import logging
import random
from math import sqrt

import pandas as pd
from tqdm import tqdm

from ..exp_args import ExpArgs
from bm25_shots import create_bm25_few_shots, filter_future_subject_notes
from sbert_shots import create_all_sbert_few_shots, store_few_shots
from ..utils import load_notes

LABEL_TYPE = 'LONG_CODES'

logger = logging.getLogger(__name__)


def calculate_label_cos_sim(query_label, search_label):
    """Calculate the cosine similarity between two lists of strings."""
    try:
        set1 = set(query_label)
    except TypeError:
        return 0  # return 0 if list 1 is np.nan
    set2 = set(search_label)

    intersection = set1.intersection(set2)
    d = sqrt(len(set1) * len(set2))
    similarity = (len(intersection) / d) if d else 0
    similarity = int(similarity * 10000)
    return round(similarity, 4)


def get_cos_sims_1_n(query_label: list, search_labels: list) -> list:
    return [
        calculate_label_cos_sim(query_label, train_label)
        for train_label
        in search_labels
    ]


def get_subject_cos_sims(query_note: pd.Series, search_df: pd.DataFrame) -> pd.Series:
    """Filter future subject notes and calculate cosine similarity for the remaining ones."""
    cos_sims = pd.Series(get_cos_sims_1_n(query_note[LABEL_TYPE], search_df[LABEL_TYPE]))
    search_df = filter_future_subject_notes(query_note, search_df)
    return cos_sims.loc[search_df.index]


def get_few_shots(cos_sims: pd.Series, search_df: pd.DataFrame):
    """Calculate golden few shots (hadm_ids for 10 notes with the highest cosine similarity)."""
    return search_df.loc[cos_sims.nlargest(10).index]['hadm_id'].values


def get_randoms_df(args: ExpArgs, query_ids: pd.Series, search_ids: pd.Series):
    random_label_df = pd.DataFrame()
    random_ids = [random.sample(list(search_ids), 10) for _ in range(len(query_ids))]
    random_label_df[f'icd{args.icd}_{args.task}_hadm_id'] = query_ids
    random_label_df[f'icd{args.icd}_{args.task}_few_shots'] = random_ids
    return random_label_df


def get_gold_shots(
        args: ExpArgs,
        query_notes: pd.DataFrame,
        search_notes: pd.DataFrame
) -> pd.DataFrame:
    """Calculate golden few shots and recall for each model."""
    gold_label_df = pd.DataFrame()
    gold_shots = []
    for i, query_note in tqdm(query_notes.iterrows(), args.tqdm_str, len(query_notes)):
        cos_sims = get_subject_cos_sims(query_note, search_notes)
        gold_shots.append(get_few_shots(cos_sims, search_notes))

    gold_label_df[f'icd{args.icd}_{args.task}_hadm_id'] = query_notes['hadm_id']
    gold_label_df[f'icd{args.icd}_{args.task}_few_shots'] = gold_shots

    return gold_label_df


def create_gold_or_random_shots(args: ExpArgs, splits=('dev', 'test')):
    logger.info(f'--- Create {args.encoder_name} for {splits} ---')
    for split in splits:
        args.query_split = split
        shot_df = pd.DataFrame()

        for icd, task in args.get_experiment_iterator():
            args.set_task(task, icd)
            query_notes = load_notes(args).reset_index()
            search_notes = load_notes(args, True).reset_index()

            if args.encoder_name == 'gold_shots':
                shot_experiment_df = get_gold_shots(args, query_notes, search_notes)
            elif args.encoder_name == 'random_shots':
                shot_experiment_df = get_randoms_df(args, query_notes['hadm_id'],
                                                    search_notes['hadm_id'])
            shot_df = pd.concat([shot_df, shot_experiment_df], axis=1)

        store_few_shots(args, shot_df)


def create_few_shots(args: ExpArgs, splits=('dev', 'test')):
    create_all_sbert_few_shots(args, splits)
    if 'BM_25' in args.all_retrievers:
        create_bm25_few_shots(args, splits)
    if 'gold_shots' in args.all_retrievers:
        args.encoder_name = 'gold_shots'
        create_gold_or_random_shots(args, splits)
    if 'random_shots' in args.all_retrievers:
        args.encoder_name = 'random_shots'
        create_gold_or_random_shots(args, splits)
