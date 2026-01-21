import itertools
import logging
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExpArgs:
    results_path: Path = Path("/Users/janfrick/Documents/data/results")
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    config_path: Path = Path("scripts/tuning_config.yaml")

    all_llms: list[str] = ("Llama3p1-8B", "Llama3p1-8B-Instruct")
    all_retrievers: list[str] = (
        "sentence-transformers/all-mpnet-base-v2",
        "neuml/pubmedbert-base-embeddings"
    )
    all_encoders: list[str] = ("BiomedBERT_tuned", "BiomedBERT_untuned")
    all_icds: list[int] = (9, 10)
    all_tasks: list[str] = ("icu", "hosp")

    icd: int = 9
    task: str = "icu"
    encoder_name: str = "pubmedbert-base-embeddings"
    llm_name: str = "Qwen2.5-3B-Instruct"
    query_split: str = "test"
    num_few_shots: int = 0
    device: str = "mps"
    batch_size: int = 32
    max_concurrency: int = 8
    test_mode: bool = False
    limit_label_space: bool = True  # True = labels are limited to those appearing in the test set

    def __post_init__(self):
        """Post-initialization: Configure logging and warn if test mode is active."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.logger.info(
            f"Experiment initialized with ICDs: {self.all_icds} and Tasks: {self.all_tasks}")
        self.logger.info(f"Mimic file path: {self.data_dir / 'mimic-iv'}")
        self.logger.info(f"Results file path: {self.results_path}")

        if self.test_mode:
            self.logger.warning("Test mode is active. Only loading 5 query samples.")

    def get_envs_from_os(self):
        """Overwrite(!) values with values of os.env if they exist. Does not affect all_XXX vals."""
        self.icd = int(os.getenv("ICD", self.icd))
        self.task = os.getenv("TASK", self.task)
        self.encoder_name = os.getenv("ENCODER", self.encoder_name)
        self.llm_name = os.path.basename(os.getenv("MODEL", self.llm_name))
        self.num_few_shots = int(os.getenv("NUM_FEW_SHOTS", self.num_few_shots))
        self.query_split = os.getenv("QUERY_SPLIT", self.query_split)
        self.device = os.getenv("DEVICE", self.device)
        self.batch_size = int(os.getenv("BATCH_SIZE", self.batch_size))
        self.max_concurrency = int(os.getenv("CONCURRENCY", self.max_concurrency))
        self.test_mode = os.getenv("TEST_MODE") == 'TRUE'  # defaults False
        self.limit_label_space = os.getenv("LIMIT_LABEL_SPACE", "TRUE") != 'FALSE'  # defaults True

        logging.info('Envs loaded ' + self.tqdm_str)
        if self.test_mode:
            self.logger.warning("Test mode is active. Only loading 5 query samples.")

    def set_task(self, task: str, icd: int) -> None:
        """Set the current experiment task and ICD version."""
        self.task = task
        self.icd = icd

    @property
    def llm_metrics_dir(self) -> Path:
        return self.data_dir / "metrics" / self.llm_name

    @property
    def aggregate_dir(self) -> Path:
        return self.data_dir / "metrics" / "aggregate"

    @property
    def query_note_path(self) -> Path:
        return (self.data_dir / "mimic-iv" / f"icd-{self.icd}" / self.task /
                f"{self.query_split}_{self.icd}_{self.task}.pq")

    @property
    def search_note_path(self) -> Path:
        return (self.data_dir / "mimic-iv" / f"icd-{self.icd}" / self.task /
                f"train_{self.icd}_{self.task}.pq")

    @property
    def embedding_dir(self) -> Path:
        return self.data_dir / "embeddings" / self.encoder_name / f"icd-{self.icd}" / self.task

    @property
    def embedding_path(self) -> Path:
        return self.embedding_dir / f"{self.query_split}_embeddings.pq"

    @property
    def search_embedding_path(self) -> Path:
        return self.embedding_dir / "train_embeddings.pq"

    @property
    def few_shot_dir(self) -> Path:
        return self.data_dir / "few_shots" / self.encoder_name

    @property
    def few_shot_path(self) -> Path:
        return self.few_shot_dir / f"{self.query_split}_few_shots.pq"

    @property
    def _llm_results_path(self) -> Path:
        return self.results_path / f"{self.llm_name}-{self.task}-icd-{self.icd}"

    @property
    def ft_model_path(self) -> Path:
        return self.model_dir / self.encoder_name / f"{self.icd}_{self.task}"

    @property
    def encoder_pred_dir(self) -> Path:
        return self.results_path / self.llm_name.lower()

    @property
    def encoder_pred_path(self) -> Path:
        return (self.encoder_pred_dir /
                f"{self.llm_name}_{self.encoder_name}_{self.task}-icd-{self.icd}.json")

    @property
    def tqdm_str(self) -> str:
        return f'ICD: {self.icd}, Task: {self.task}\t'

    def get_zeroshot_results_path(self, pred_type: str = None, file_format: str = "json") -> Path:
        """Get path for zero-shot results."""
        filename = f"zeroshot-{pred_type}.{file_format}" if pred_type else f"zeroshot.{file_format}"
        return self._llm_results_path.with_name(f"{self._llm_results_path.name}-{filename}")

    def get_fewshot_results_path(self, shot: int, pred_type=None, file_format='json') -> str:
        """Path to results from LLM few shot experiment"""
        if pred_type in ['bm25', 'pubmed-clinibench', 'random', 'gold']:
            return f'{self._llm_results_path}-fewshot-{shot}-{pred_type}.{file_format}'
        return f'{self._llm_results_path}-fewshot-{shot}.{file_format}'

    def get_icd_description_path(self, icd=None) -> Path:
        """Get path for ICD code description mapping."""
        icd = icd if icd is not None else self.icd
        return self.data_dir / "mimic-iv" / f"icd_codes_{icd}.csv"

    def get_metrics_df_path(self, macro: bool = False) -> Path:
        """Get path for LLM metrics CSV."""
        filename = "macro.csv" if macro else f"icd-{self.icd}_{self.task}.csv"
        return self.llm_metrics_dir / filename

    def get_experiment_iterator(self, splits: list[str] = None):
        """Generate combinations of ICDs, tasks, and optional splits."""
        if splits:
            return list(itertools.product(self.all_icds, self.all_tasks, splits))
        else:
            return list(itertools.product(self.all_icds, self.all_tasks))

    def __str__(self) -> str:
        return (f"Results Path: {self.results_path}\n"
                f"ICD + Task: {self.icd} {self.task}\n"
                f"LLM: {self.llm_name}")

    def __repr__(self) -> str:
        return self.__str__()
