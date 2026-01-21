import argparse

from retrieval.create_few_shots import create_few_shots
from exp_args import ExpArgs

if __name__ == "__main__":

    RETRIEVER_MODELS = [
        # 'sentence-transformers/all-mpnet-base-v2',
        'neuml/pubmedbert-base-embeddings',
        # 'bvanaken/CORe-clinical-diagnosis-prediction',
        # 'jinaai/jina-embeddings-v2-base-en',
        # 'jinaai/jina-embeddings-v2-base-en-2048',
        # 'dmlls/all-mpnet-base-v2-negation',
        # 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
        # 'pritamdeka/S-PubMedBert-MS-MARCO',
        # 'CORe-clinibench',
        # 'GatorTronS',
        # 'm2-bert-80M-32k-retrieval'
        'BM_25',
        'gold_shots',
        'random_shots'
    ]
    ALL_ICDS = [9, 10]
    ALL_TASKS = ['icu', 'hosp']
    ALL_SPLITS = ['test']

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--data_dir", type=str, default='\data', help="path to pvc storage")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--model', action='append', default=RETRIEVER_MODELS,
                        help='Model, appends multi calls')
    parser.add_argument("--split", action='append', default=ALL_SPLITS)
    parser.add_argument("--icds", action='append', default=ALL_ICDS)
    parser.add_argument("--tasks", action='append', default=ALL_TASKS)
    args = parser.parse_args()

    splits = args.split
    args = ExpArgs(
        data_dir=args.data_dir,
        all_retrievers=args.model,
        all_icds=args.icds,
        all_tasks=args.tasks,
        batch_size=args.batch_size,
        model_dir='',
        device=args.device
    )
    create_few_shots(args, splits)
