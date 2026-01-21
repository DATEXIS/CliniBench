import logging

import pandas as pd
from fastbm25 import fastbm25
from tqdm import tqdm

from sbert_shots import store_few_shots
from ..exp_args import ExpArgs
from ..utils import load_notes

logger = logging.getLogger(__name__)


def filter_future_subject_notes(query_note: pd.Series, search_df: pd.DataFrame) -> pd.DataFrame:
    return search_df[(search_df['subject_id'] != query_note['subject_id']) |
                     (search_df['charttime'] < query_note['charttime'])]


def get_bm25_predictions(bm25: fastbm25, query_note: pd.Series, search_df: pd.DataFrame, top_k=10):
    subject_search_df = filter_future_subject_notes(query_note, search_df)
    search_ids = subject_search_df.index.to_list()
    top_notes = bm25.top_k_sentence(query_note['TEXT'].lower().split(" "), 70)
    top_ids = [subject_search_df.loc[x[1]]['hadm_id'] for x in top_notes if x[1] in search_ids]
    return top_ids[:top_k]


def create_bm25_few_shots(
    args: ExpArgs,
    splits=('dev', 'test')
):
    logger.info(f'--- Create BM25 Few Shots for {splits} ---')
    args.encoder_name = 'BM_25'  # BM_25 ain't no encoder
    for split in splits:
        few_shot_df = pd.DataFrame()
        args.query_split = split
        for icd, task in args.get_experiment_iterator():
            args.set_task(task, icd)
            query_notes = load_notes(args).reset_index()
            search_notes = load_notes(args, True).reset_index()
            tokenized_corpus = [doc.lower().split(" ") for doc in search_notes["TEXT"]]
            bm25 = fastbm25(tokenized_corpus)
            all_predictions = []
            for i, query_note in tqdm(query_notes.iterrows(), args.tqdm_str, len(query_notes)):
                predictions = get_bm25_predictions(bm25, query_note, search_notes)
                all_predictions.append(predictions)

            few_shot_df[f'icd{icd}_{task}_hadm_id'] = query_notes['hadm_id']
            few_shot_df[f'icd{icd}_{task}_few_shots'] = pd.Series(all_predictions)
        store_few_shots(args, few_shot_df)
