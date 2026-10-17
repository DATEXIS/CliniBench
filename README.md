# CliniBench: Clinical Outcome Prediction Benchmark

**CliniBench** is the first comprehensive benchmark for comparing encoder-based classifiers and generative large language models (LLMs) for discharge diagnosis prediction from admission notes in the MIMIC-IV dataset.

> **Corrected results.** The paper's result tables mixed two metric implementations for encoders and LLMs. See [Results (corrected evaluation)](#results-corrected-evaluation) for the recomputed numbers and what changed.

## Installation

### Prerequisites
- Python 3.11
- CUDA-capable GPU (recommended for encoder training and LLM inference)
- Access to MIMIC-IV dataset (requires PhysioNet credentialing)

### Setup with uv (recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/clinibench.git
cd clinibench

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Docker Setup

```bash
# Build the Docker image
docker build -t clinibench:latest .

# Run container with GPU support
docker run --gpus all -it -v $(pwd):/clinibench clinibench:latest {script_name.py}
```

## Dataset Preparation

### Accessing MIMIC-IV

1. Complete the CITI "Data or Specimens Only Research" course
2. Request access to MIMIC-IV on [PhysioNet](https://physionet.org/content/mimiciv/2.2/)
3. Download MIMIC-IV dataset (version 2.2)
4. Download MIMIC-IV-Note dataset (version 2.2)
5. Follow https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres for the general mimic-iv content
6. Follow https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv-note/buildmimic/postgres/README.md for mimic-iv notes

### Generating the admission notes and dataset splits
Adjust the PostgreSQL connection string (username, password, database name, hostname and port) in the `create_admission_note_dataset.sh` script and execute it.
You may adjust the paths where the dataset is stored within the script. All following steps and default parameters assume the paths defined in the script. 

This script will 1. create the admission from discharge notes dataset, 2. create the train/dev/test splits from the full 
dataset and 3. create a file that contains the notes from all train splits which is used for few-shot experiments. 

## Usage

**Step 1: Start the vLLM server**

Launch a vLLM server that provides access to your model. By default, the script assumes it's running on `localhost`. If your server is on a different host, you'll need to adjust the `--vllm_ip` parameter in Step 2.

**Step 2a: Generate diagnosis predictions (zero-shot)**

Run the prediction script with your model and data:
```bash
python src/generative_models/predict_diagnoses.py \
   --vllm_ip=http://localhost:8000 \
   --test_data=data/mimic-iv/icd-10/icu/test_10_icu.parquet \
   --icd10_code_file=data/icd10_codes.csv \
   --icd9_code_file=data/icd9_codes.csv \
   --results_file=qwen2.5-3B-mimic-iv-icu-icd10.parquet
```

**Step 2b: Generate diagnosis predictions (few-shot)**

Run the prediction script including few-shot examples:
```bash
python src/generative_models/predict_diagnoses.py \
    --vllm_ip=http://localhost:8000 \
    --test_data=data/mimic-iv/icd-10/icu/test_10_icu.parquet \
    --icd10_code_file=data/icd10_codes.csv \
    --icd9_code_file=data/icd9_codes.csv \
    --results_file=qwen2.5-3B-mimic-iv-icu-icd10.parquet \
    --few_shot \
    --few_shot_note_data=./data/fewshot-candidates/notes.parquet \
    --few_shot_ids=./data/fewshot-candidates/gold_shots/test_few_shots.pq \
    --few_shot_column=icd10_icu \
    --num_few_shot_candidates=5
```

**Step 3: Evaluate the predictions**

Map the generated diagnosis descriptions to ICD codes and calculate evaluation metrics:
```bash
python src/generative_models/map_and_evaluate.py \
    --predictions_file_path=qwen2.5-3B-mimic-iv-icu-icd10.parquet \
    --results_save_path=results.json \
    --icd_version=10 \
    --icd10_code_file=data/icd10_codes.csv \
    --icd9_code_file=data/icd9_codes.csv
```

**Step 4: View results**

All evaluation metrics will be saved to the file specified in `--results_save_path` (e.g., `results.json`).

To compare runs with each other and with the encoders, add `--save_mapped_results` in Step 3 and score the saved mappings with `src/evaluation/evaluate_runs.py` (see [Reproducing](#reproducing)). The metrics written by `map_and_evaluate.py` use the legacy definitions of MAP and code ranking and are not comparable to the corrected tables.

## Results (corrected evaluation)

The numbers below replace the result tables of the paper (Tables 2, 3, 10 and 11 in arXiv 2509.26136v2, EACL 2026). They are recomputed from the released per-note predictions of all 840 runs (24 encoder runs, 816 generative runs) with a single evaluation protocol for both model families.

### Why the numbers changed

Encoders and generative LLMs were scored by two different code paths: the encoder test loop in `src/encoder_models/model/lightning_model.py`, and `src/generative_models/metrics.py` via `map_and_evaluate.py`. Both paths compute macro recall, macro precision and main diagnosis accuracy the same way, and those columns reproduce from the released predictions. The other columns do not compare like with like:

| Issue | Paper | Corrected |
|:---|:---|:---|
| MAP | Encoders: torchmetrics `RetrievalMAP` with the main-diagnosis class as query index (`query_idces` in `model/dataset.py`), which pools all notes sharing a main diagnosis into one ranking over full score vectors. LLMs: per-note `RetrievalMAP` over the predicted codes in `set()` order, i.e. a random ranking. | Per-note AP@20 over the ranked top-20 list for every system. LLM codes are ranked in generation order. |
| F1 (encoders) | HOSP Table 2 values are not computed on the top-20 predictions and cannot be reproduced from the released predictions. The ICU Table 11 encoder F1 column contains the HOSP top-20 values. | Micro and macro F1 on the top-20 list, as for the LLMs. |
| GatorTronS (untuned) | The printed rows do not match the released predictions (HOSP recall 30.45, ICU recall 28.61). | Recomputed from the released predictions (HOSP 35.36, ICU 23.71). The released dumps may come from a different checkpoint than the printed rows; the other five encoder configurations match their dumps exactly. |
| Misprinted LLM cells (HOSP, Table 2) | Llama 3.1-8B-Instruct zero-shot precision 17.23; Llama 3.1-8B 5-shot F1 12.41; Qwen2-7B 5-shot F1 15.08 | 10.88; 17.22; 15.31 |

All other LLM cells of Tables 2, 3, 10 and 11 reproduce to within 0.05 on recall, precision, MD accuracy and F1. MAP changes for every row because of the new definition.

### Evaluation protocol

The same rules apply to every system and data partition (`src/evaluation/unified_metrics.py`):

* **Label space.** The short codes of the test gold. CliniBench keeps only codes that occur in the train, validation and test split, and this filtered label space is used for encoders and LLMs alike: 917 / 1,177 classes (HOSP ICD-9 / ICD-10) and 777 / 849 classes (ICU). The LLM text-to-code index is restricted to codes in this label space. The evaluator checks that every run of a partition uses the identical label space.
* **Notes.** All test notes with at least one gold code (HOSP 36,003 / 19,779, ICU 8,157 / 5,493).
* **Predictions.** The top 20 codes, truncated to short codes and de-duplicated by first occurrence. Encoders rank by score, LLMs by generation order. LLM lists can be shorter than 20 after de-duplication.
* **Metrics.** The main tables report micro F1, macro F1, MAP@20 and MD Acc.; `results/all_runs.md` adds micro and macro recall and precision.
  * Micro scores pool true positives, false positives and false negatives over all (note, code) pairs.
  * Macro scores are computed per class over the label space (0 when undefined) and averaged with equal class weights. Macro F1 is the mean of per-class F1.
  * MAP@20 is the per-note average precision normalised by min(|gold|, 20).
  * MD Acc. is the share of notes whose first gold code is in the top 20.
* **Aggregation.** Unweighted mean of the ICD-9 and ICD-10 partitions of each care unit, as in the paper.

### What changes in the conclusions

* **Encoders still lead on every main metric.** GatorTronS (untuned) is best on all four. The best LLM without oracle retrieval is Qwen2-72B-Instruct with BM25 demonstrations.
  * Micro F1: HOSP 48.03 vs. 35.64, ICU 50.06 vs. 32.21.
  * Macro F1: HOSP 25.77 vs. 17.58, ICU 21.49 vs. 15.08.
  * In ICU the macro F1 gap shrinks from about 11 points in the paper (encoder F1 printed as 25.77) to about 6.
* **The MAP gap is larger than reported.** It grows once both families are ranked the same way (HOSP 58.84 vs. 31.26, ICU 52.54 vs. 25.30).
* **ICU macro recall:** the best zero-shot LLM (Qwen2-72B-Instruct, 24.11) is now slightly above the best encoder (GatorTronS, 23.71; printed as 28.61).
* **Threshold tuning** lowers micro F1, macro F1, MAP, recall and MD accuracy for all three encoders in both units, and raises only macro precision. In the paper's HOSP table the tuned encoders appeared to have the best F1.
* **Prompting:** 5 BM25 demonstrations raise micro F1 and MAP for all 12 LLMs in both units. CoT improves macro F1 for only 1 and micro F1 for only 2 of 12 models in each unit.

### Reproducing

```bash
# score every run in a directory of encoder dumps and mapped LLM predictions
python src/evaluation/evaluate_runs.py score \
    --results_dir <dir with Encoder_*.json and <model>-<unit>-icd-<v>-<setting>.{json,parquet}> \
    --out_csv results/unified_metrics.csv
# aggregate into the tables below and results/all_runs.md
python src/evaluation/evaluate_runs.py tables --metrics_csv results/unified_metrics.csv
```

Per-partition scores for all 840 runs are in [`results/unified_metrics.csv`](results/unified_metrics.csv). Aggregated tables for every setting (all retrievers and shot counts for all 12 LLMs) are in [`results/all_runs.md`](results/all_runs.md). File-name suffixes map to the paper's retrievers as follows: no suffix = PubMedBERT-NeuML, `pubmed-clinibench` = BioMedBERT-ft, `bm25`, `gold`, `random`.

### Main results

Top-20 predictions; unweighted mean of the ICD-9 and ICD-10 partitions of each care unit, as in the paper. Micro F1 pools all (note, code) decisions and is dominated by frequent codes; macro F1 weights every class equally and is dominated by the long tail. Encoder rows marked *untuned* rank by raw scores, *threshold-tuned* rows by the calibrated scores. Micro/macro recall and precision for every run are in [`results/all_runs.md`](results/all_runs.md).

**HOSP**

| Model | Setting | Micro F1 | Macro F1 | MAP@20 | MD Acc. |
|:---|:---|---:|---:|---:|---:|
| BioMedBERT-110M | untuned | 45.83 | 22.00 | 54.99 | 78.21 |
| BioMedBERT-110M | threshold-tuned | 30.88 | 20.71 | 32.83 | 59.16 |
| GatorTronS-345M | untuned | 48.03 | 25.77 | 58.84 | 81.64 |
| GatorTronS-345M | threshold-tuned | 33.55 | 25.34 | 37.23 | 65.22 |
| M2-BERT-80M | untuned | 38.72 | 9.49 | 41.67 | 60.83 |
| M2-BERT-80M | threshold-tuned | 24.32 | 8.34 | 24.87 | 40.47 |
| Llama 3.1-8B | zero-shot | 22.10 | 11.06 | 14.69 | 35.22 |
| Llama 3.1-8B-Instruct | zero-shot | 24.71 | 11.61 | 17.89 | 47.53 |
| Llama 3.1-70B | zero-shot | 19.44 | 9.66 | 14.99 | 35.21 |
| Llama 3.1-70B-Instruct | zero-shot | 22.04 | 11.33 | 15.15 | 52.20 |
| Mistral-7B | zero-shot | 15.60 | 8.50 | 9.67 | 27.25 |
| Mistral-7B-Instruct | zero-shot | 25.06 | 12.16 | 19.91 | 54.06 |
| Mistral-Nemo-13B | zero-shot | 21.46 | 10.16 | 14.10 | 32.83 |
| Mistral-Nemo-13B-Instruct | zero-shot | 20.84 | 10.37 | 16.47 | 50.35 |
| Qwen2-7B | zero-shot | 26.09 | 12.26 | 20.54 | 43.37 |
| Qwen2-7B-Instruct | zero-shot | 27.28 | 12.90 | 22.30 | 48.95 |
| Qwen2-72B | zero-shot | 26.96 | 12.48 | 21.77 | 46.17 |
| Qwen2-72B-Instruct | zero-shot | 28.53 | 13.13 | 24.10 | 51.58 |
| Llama 3.1-8B | zero-shot CoT | 19.12 | 9.68 | 13.06 | 37.96 |
| Llama 3.1-8B-Instruct | zero-shot CoT | 20.80 | 10.11 | 14.82 | 49.61 |
| Llama 3.1-70B | zero-shot CoT | 22.20 | 10.19 | 18.01 | 43.63 |
| Llama 3.1-70B-Instruct | zero-shot CoT | 22.46 | 10.59 | 18.07 | 51.39 |
| Mistral-7B | zero-shot CoT | 0.37 | 0.16 | 0.20 | 0.69 |
| Mistral-7B-Instruct | zero-shot CoT | 22.39 | 10.85 | 17.30 | 49.76 |
| Mistral-Nemo-13B | zero-shot CoT | 15.59 | 8.00 | 11.29 | 38.32 |
| Mistral-Nemo-13B-Instruct | zero-shot CoT | 18.68 | 9.57 | 14.61 | 51.61 |
| Qwen2-7B | zero-shot CoT | 25.42 | 12.04 | 18.80 | 42.01 |
| Qwen2-7B-Instruct | zero-shot CoT | 25.82 | 12.15 | 19.88 | 49.38 |
| Qwen2-72B | zero-shot CoT | 25.40 | 11.49 | 20.45 | 44.05 |
| Qwen2-72B-Instruct | zero-shot CoT | 27.45 | 12.74 | 23.03 | 50.24 |
| Llama 3.1-8B | 5-shot BM25 | 35.01 | 17.22 | 27.15 | 47.23 |
| Llama 3.1-8B-Instruct | 5-shot BM25 | 32.18 | 15.44 | 23.92 | 44.34 |
| Llama 3.1-70B | 5-shot BM25 | 32.14 | 15.09 | 26.18 | 49.05 |
| Llama 3.1-70B-Instruct | 5-shot BM25 | 32.76 | 15.81 | 26.05 | 45.49 |
| Mistral-7B | 5-shot BM25 | 31.14 | 15.29 | 23.64 | 44.13 |
| Mistral-7B-Instruct | 5-shot BM25 | 32.88 | 15.93 | 29.07 | 48.89 |
| Mistral-Nemo-13B | 5-shot BM25 | 31.06 | 15.10 | 23.60 | 45.97 |
| Mistral-Nemo-13B-Instruct | 5-shot BM25 | 32.52 | 16.05 | 26.02 | 48.41 |
| Qwen2-7B | 5-shot BM25 | 31.01 | 15.31 | 23.15 | 43.40 |
| Qwen2-7B-Instruct | 5-shot BM25 | 32.04 | 15.55 | 26.46 | 51.65 |
| Qwen2-72B | 5-shot BM25 | 33.84 | 16.46 | 27.74 | 49.44 |
| Qwen2-72B-Instruct | 5-shot BM25 | 35.64 | 17.58 | 31.26 | 53.00 |

**ICU**

| Model | Setting | Micro F1 | Macro F1 | MAP@20 | MD Acc. |
|:---|:---|---:|---:|---:|---:|
| BioMedBERT-110M | untuned | 47.33 | 17.11 | 48.25 | 72.83 |
| BioMedBERT-110M | threshold-tuned | 31.52 | 14.09 | 29.85 | 55.91 |
| GatorTronS-345M | untuned | 50.06 | 21.49 | 52.54 | 77.74 |
| GatorTronS-345M | threshold-tuned | 34.58 | 18.89 | 33.10 | 63.66 |
| M2-BERT-80M | untuned | 38.54 | 8.14 | 33.65 | 56.05 |
| M2-BERT-80M | threshold-tuned | 18.22 | 4.60 | 13.68 | 35.05 |
| Llama 3.1-8B | zero-shot | 20.87 | 11.20 | 12.69 | 33.43 |
| Llama 3.1-8B-Instruct | zero-shot | 25.81 | 13.03 | 16.49 | 47.34 |
| Llama 3.1-70B | zero-shot | 19.69 | 10.48 | 13.22 | 33.60 |
| Llama 3.1-70B-Instruct | zero-shot | 24.59 | 12.55 | 14.95 | 50.37 |
| Mistral-7B | zero-shot | 13.99 | 8.07 | 7.68 | 25.66 |
| Mistral-7B-Instruct | zero-shot | 26.55 | 13.69 | 18.36 | 51.37 |
| Mistral-Nemo-13B | zero-shot | 20.22 | 10.86 | 12.25 | 31.63 |
| Mistral-Nemo-13B-Instruct | zero-shot | 23.08 | 11.58 | 15.95 | 48.39 |
| Qwen2-7B | zero-shot | 25.33 | 13.11 | 17.78 | 40.97 |
| Qwen2-7B-Instruct | zero-shot | 27.73 | 14.10 | 20.11 | 45.86 |
| Qwen2-72B | zero-shot | 26.84 | 13.70 | 18.46 | 43.28 |
| Qwen2-72B-Instruct | zero-shot | 29.50 | 14.42 | 21.29 | 48.11 |
| Llama 3.1-8B | zero-shot CoT | 18.68 | 9.76 | 11.24 | 35.57 |
| Llama 3.1-8B-Instruct | zero-shot CoT | 22.65 | 11.49 | 13.71 | 47.65 |
| Llama 3.1-70B | zero-shot CoT | 22.52 | 11.22 | 15.73 | 40.81 |
| Llama 3.1-70B-Instruct | zero-shot CoT | 25.53 | 12.08 | 18.00 | 50.18 |
| Mistral-7B | zero-shot CoT | 1.48 | 0.13 | 0.72 | 1.38 |
| Mistral-7B-Instruct | zero-shot CoT | 24.41 | 12.30 | 16.28 | 46.10 |
| Mistral-Nemo-13B | zero-shot CoT | 15.66 | 8.23 | 9.67 | 37.51 |
| Mistral-Nemo-13B-Instruct | zero-shot CoT | 20.97 | 10.78 | 14.03 | 48.54 |
| Qwen2-7B | zero-shot CoT | 24.64 | 12.55 | 16.68 | 40.19 |
| Qwen2-7B-Instruct | zero-shot CoT | 26.99 | 13.64 | 18.59 | 46.30 |
| Qwen2-72B | zero-shot CoT | 25.46 | 12.56 | 17.80 | 40.93 |
| Qwen2-72B-Instruct | zero-shot CoT | 28.76 | 14.12 | 21.00 | 47.16 |
| Llama 3.1-8B | 5-shot BM25 | 29.91 | 13.20 | 21.16 | 48.33 |
| Llama 3.1-8B-Instruct | 5-shot BM25 | 29.27 | 13.33 | 19.75 | 44.27 |
| Llama 3.1-70B | 5-shot BM25 | 28.64 | 12.13 | 20.08 | 50.85 |
| Llama 3.1-70B-Instruct | 5-shot BM25 | 30.08 | 14.47 | 21.36 | 44.90 |
| Mistral-7B | 5-shot BM25 | 26.07 | 10.78 | 17.21 | 43.16 |
| Mistral-7B-Instruct | 5-shot BM25 | 31.08 | 14.95 | 24.09 | 47.72 |
| Mistral-Nemo-13B | 5-shot BM25 | 26.18 | 11.01 | 17.66 | 44.95 |
| Mistral-Nemo-13B-Instruct | 5-shot BM25 | 26.44 | 11.37 | 17.82 | 43.28 |
| Qwen2-7B | 5-shot BM25 | 26.43 | 11.35 | 17.80 | 43.35 |
| Qwen2-7B-Instruct | 5-shot BM25 | 30.22 | 14.15 | 21.98 | 50.52 |
| Qwen2-72B | 5-shot BM25 | 30.07 | 13.32 | 21.96 | 49.40 |
| Qwen2-72B-Instruct | 5-shot BM25 | 32.21 | 14.77 | 24.99 | 52.06 |

### Few-shot retrievers (Qwen2-72B-Instruct)

Counterpart of Tables 3 and 10. "Gold heuristic" uses the gold labels to select demonstrations and is an upper bound, not a deployable retriever.

**HOSP: Qwen2-72B-Instruct, few-shot retrievers**

| Retriever | #FS | Micro F1 | Macro F1 | MAP@20 | MD Acc. |
|:---|:---|---:|---:|---:|---:|
| Zero-shot | 0 | 28.53 | 13.13 | 24.10 | 51.58 |
| Random | 1 | 27.10 | 12.70 | 21.44 | 42.66 |
| Random | 3 | 28.12 | 12.66 | 22.17 | 42.86 |
| Random | 5 | 28.30 | 12.48 | 22.16 | 41.59 |
| BM25 | 1 | 31.97 | 15.56 | 26.72 | 49.62 |
| BM25 | 3 | 34.50 | 16.92 | 29.83 | 52.32 |
| BM25 | 5 | 35.64 | 17.58 | 31.26 | 53.00 |
| PubMedBERT-NeuML | 1 | 28.24 | 13.32 | 23.33 | 46.86 |
| PubMedBERT-NeuML | 3 | 30.48 | 14.46 | 26.42 | 50.97 |
| PubMedBERT-NeuML | 5 | 31.75 | 14.90 | 28.19 | 52.63 |
| BioMedBERT-ft | 1 | 28.84 | 13.78 | 23.89 | 46.59 |
| BioMedBERT-ft | 3 | 32.29 | 15.34 | 28.15 | 51.24 |
| BioMedBERT-ft | 5 | 34.08 | 16.30 | 30.49 | 53.04 |
| Gold heuristic | 1 | 34.16 | 16.68 | 31.16 | 51.36 |
| Gold heuristic | 3 | 42.42 | 20.71 | 42.26 | 60.04 |
| Gold heuristic | 5 | 45.88 | 22.43 | 46.89 | 62.62 |

**ICU: Qwen2-72B-Instruct, few-shot retrievers**

| Retriever | #FS | Micro F1 | Macro F1 | MAP@20 | MD Acc. |
|:---|:---|---:|---:|---:|---:|
| Zero-shot | 0 | 29.50 | 14.42 | 21.29 | 48.11 |
| Random | 1 | 26.19 | 12.81 | 17.74 | 40.80 |
| Random | 3 | 26.33 | 12.04 | 18.05 | 41.22 |
| Random | 5 | 26.14 | 11.53 | 17.84 | 40.66 |
| BM25 | 1 | 29.59 | 14.62 | 21.41 | 47.27 |
| BM25 | 3 | 31.44 | 15.08 | 23.90 | 51.18 |
| BM25 | 5 | 32.21 | 14.77 | 24.99 | 52.06 |
| PubMedBERT-NeuML | 1 | 27.84 | 13.17 | 19.76 | 45.59 |
| PubMedBERT-NeuML | 3 | 29.74 | 13.38 | 22.50 | 50.50 |
| PubMedBERT-NeuML | 5 | 30.64 | 13.46 | 23.79 | 52.34 |
| BioMedBERT-ft | 1 | 28.14 | 13.73 | 19.96 | 44.80 |
| BioMedBERT-ft | 3 | 30.74 | 14.13 | 23.50 | 50.92 |
| BioMedBERT-ft | 5 | 32.19 | 14.81 | 25.30 | 52.59 |
| Gold heuristic | 1 | 32.64 | 16.00 | 25.05 | 48.46 |
| Gold heuristic | 3 | 41.39 | 18.63 | 35.75 | 59.15 |
| Gold heuristic | 5 | 44.75 | 19.70 | 40.06 | 61.66 |

## Citation

If you use CliniBench in your research, please cite:

```bibtex
@misc{grundmann2025clinibenchclinicaloutcomeprediction,
      title={CliniBench: A Clinical Outcome Prediction Benchmark for Generative and Encoder-Based Language Models}, 
      author={Paul Grundmann and Dennis Fast and Jan Frick and Thomas Steffek and Felix Gers and Wolfgang Nejdl and Alexander Löser},
      year={2025},
      eprint={2509.26136},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26136}, 
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
