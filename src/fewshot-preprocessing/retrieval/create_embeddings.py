import logging
import warnings

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from ..utils import FINE_TUNED_MODELS, create_dir, get_model_name, load_notes
from ..exp_args import ExpArgs

logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_sentence_transformer(args: ExpArgs, model_path: str) -> SentenceTransformer:
    if model_path[-4:] == '2048':
        model_path = model_path[:-5]
    elif model_path in FINE_TUNED_MODELS.keys():
        model_path = str(args.ft_model_path)

    model = SentenceTransformer(model_path, device=args.device, trust_remote_code=True)

    if args.encoder_name[-4:] == '2048':
        model.max_seq_length = 2048
    if '32k' in args.encoder_name:
        model.max_seq_length = 32768

    return model


def create_embeddings(model: SentenceTransformer, texts: pd.Series, args: ExpArgs) -> np.array:
    with torch.no_grad():
        embeddings = model.encode(
            texts.values,
            batch_size=args.batch_size,
            device=args.device,
            show_progress_bar=True
        )
    return embeddings.tolist()


def create_embedding_file(args: ExpArgs, model: SentenceTransformer):
    """Creates and stores embeddings for specified task / icd / split """
    admission_notes = load_notes(args)
    admission_notes['embeddings'] = create_embeddings(model, admission_notes["TEXT"], args)
    create_dir(args.embedding_dir)
    admission_notes.drop(columns=['TEXT'], axis=1, inplace=True)
    admission_notes.to_parquet(args.embedding_path)


def create_all_embeddings(
        args: ExpArgs,
        splits=('train', 'dev', 'test'),
):
    logger.info(f'--- Creating Sentence Transformers Embeddings for {splits} ---')
    for model in args.all_retrievers:
        if model in ['BM_25', 'gold_shots', 'random_shots']:
            continue  # Skip non-SBERT models

        args.encoder_name = get_model_name(model)
        if model not in FINE_TUNED_MODELS.keys():
            model = load_sentence_transformer(args, model)

        for icd, task in args.get_experiment_iterator():
            args.set_task(task, icd)
            # logger.info(f'Model: {args.encoder_name}, task: {task}, ICD: {icd}')
            if model in FINE_TUNED_MODELS.keys():
                # Load fine-tuned models for specific task / icd
                model = load_sentence_transformer(args, model)

            for split in splits:
                args.query_split = split
                create_embedding_file(args, model)
