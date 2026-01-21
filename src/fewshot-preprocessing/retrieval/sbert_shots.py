import logging

import numpy as np
import pandas as pd
import torch
from sentence_transformers.util import semantic_search

from ..exp_args import ExpArgs
from ..utils import get_model_name, create_dir, load_embeddings

logger = logging.getLogger(__name__)


def get_top_semantic_search_ids(
        query_embeddings: pd.DataFrame,
        search_embeddings: pd.DataFrame,
        args: ExpArgs
):
    hadm_ids = list(search_embeddings.index)
    search_tensor = torch.from_numpy(np.vstack(search_embeddings['embeddings']).astype(np.float32))
    query_tensor = torch.from_numpy(np.vstack(query_embeddings['embeddings']).astype(np.float32))
    search_tensor = search_tensor.to(device=args.device)
    query_tensor = query_tensor.to(device=args.device)

    with torch.no_grad():
        results_3d = semantic_search(query_tensor, search_tensor, 100, 500000, 70)

    return [[hadm_ids[result['corpus_id']] for result in results] for results in results_3d]


def get_future_subject_notes(query_note: pd.Series, search_df: pd.DataFrame) -> set:
    """Notes from same patient with same or bigger charttime."""
    future_notes = search_df[
        (search_df['subject_id'] == query_note['subject_id']) &
        (search_df['charttime'] >= query_note['charttime'])]
    return set(future_notes.index)


def store_few_shots(args: ExpArgs, few_shot_df: pd.DataFrame):
    directory = args.data_dir / 'few_shots' / args.encoder_name
    create_dir(directory)
    few_shot_df.to_parquet(args.few_shot_path)


def filter_for_top_k(query_note: pd.Series, search_df: pd.DataFrame) -> list:
    future_hadm_ids = get_future_subject_notes(query_note, search_df)
    no_future_shots = list(np.setdiff1d(query_note['few_shots'], future_hadm_ids))
    return no_future_shots[:10]


def create_sbert_few_shots(args: ExpArgs):
    query_embeddings = load_embeddings(args)
    search_embeddings = load_embeddings(args, True)
    query_embeddings['few_shots'] = get_top_semantic_search_ids(
        query_embeddings, search_embeddings, args
    )
    query_embeddings['few_shots'] = query_embeddings.apply(
        lambda query_note: filter_for_top_k(query_note, search_embeddings),
        axis=1
    )
    task_name = f"icd{args.icd}_{args.task}"
    return pd.DataFrame({
        f'{task_name}_hadm_id': query_embeddings.index.to_list(),
        f'{task_name}_few_shots': query_embeddings['few_shots']
    }).reset_index(drop=True)


def create_all_sbert_few_shots(
    args: ExpArgs,
    splits=('dev', 'test')
):
    torch.device(args.device)
    logger.info(f'--- Creating Sentence Transformers Few-Shots for {splits} ---')

    for model in args.all_retrievers:
        if model in ['BM_25', 'gold_shots', 'random_shots']:
            continue

        args.encoder_name = get_model_name(model)
        logger.info(f'SBert {args.encoder_name}')

        for split in splits:
            args.query_split = split
            few_shot_df = pd.DataFrame()
            for icd, task in args.get_experiment_iterator():
                args.set_task(task, icd)
                few_shots = create_sbert_few_shots(args)
                few_shot_df = pd.concat([few_shot_df, few_shots], axis=1)
            store_few_shots(args, few_shot_df)



