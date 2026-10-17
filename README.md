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
| F1 (encoders) | HOSP Table 2 values are not computed on the top-20 predictions and cannot be reproduced from the released predictions. The ICU Table 11 encoder F1 column contains the HOSP top-20 values. | Macro F1 on the top-20 list, as for the LLMs. |
| GatorTronS (untuned) | The printed rows do not match the released predictions (HOSP recall 30.45, ICU recall 28.61). | Recomputed from the released predictions (HOSP 35.36, ICU 23.71). The released dumps may come from a different checkpoint than the printed rows; the other five encoder configurations match their dumps exactly. |
| Misprinted LLM cells (HOSP, Table 2) | Llama 3.1-8B-Instruct zero-shot precision 17.23; Llama 3.1-8B 5-shot F1 12.41; Qwen2-7B 5-shot F1 15.08 | 10.88; 17.22; 15.31 |

All other LLM cells of Tables 2, 3, 10 and 11 reproduce to within 0.05 on recall, precision, MD accuracy and F1. MAP changes for every row because of the new definition.

### Evaluation protocol

The same rules apply to every system and data partition (`src/evaluation/unified_metrics.py`):

* **Label space.** The short codes of the test gold. CliniBench keeps only codes that occur in the train, validation and test split, and this filtered label space is used for encoders and LLMs alike: 917 / 1,177 classes (HOSP ICD-9 / ICD-10) and 777 / 849 classes (ICU). The LLM text-to-code index is restricted to codes in this label space. The evaluator checks that every run of a partition uses the identical label space.
* **Notes.** All test notes with at least one gold code (HOSP 36,003 / 19,779, ICU 8,157 / 5,493).
* **Predictions.** The top 20 codes, truncated to short codes and de-duplicated by first occurrence. Encoders rank by score, LLMs by generation order. LLM lists can be shorter than 20 after de-duplication.
* **Metrics.** Macro recall, precision and F1 per class over the label space (0 when undefined), then averaged with equal class weights. F1 is the mean of per-class F1. MAP@20 is the per-note average precision normalised by min(|gold|, 20). MD Acc. is the share of notes whose first gold code is in the top 20. The CSV also contains micro-averaged scores.
* **Aggregation.** Unweighted mean of the ICD-9 and ICD-10 partitions of each care unit, as in the paper.

### What changes in the conclusions

* **Encoders still lead overall.** GatorTronS (untuned) has the best macro F1: HOSP 25.77 vs. 17.58 for the best LLM (Qwen2-72B-Instruct, 5-shot BM25), ICU 21.49 vs. 15.08 (Qwen2-72B-Instruct, 3-shot BM25). In ICU the gap shrinks from about 11 points in the paper (encoder F1 printed as 25.77) to about 6.
* **The MAP gap is larger than reported.** It grows once both families are ranked the same way (HOSP 58.84 vs. 31.26, ICU 52.54 vs. 25.30).
* **ICU recall:** the best zero-shot LLM (Qwen2-72B-Instruct, 24.11) is now slightly above the best encoder (GatorTronS, 23.71; printed as 28.61).
* **Threshold tuning** lowers macro F1, MAP, recall and MD accuracy for all three encoders in both units, and only raises precision. In the paper's HOSP table the tuned encoders appeared to have the best F1.
* **Prompting:** 5 BM25 demonstrations raise MAP for all 12 LLMs in both units. CoT improves F1 for only 1 of 12 models in each unit.

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

Macro-averaged, top-20 predictions, mean of ICD-9 and ICD-10. Encoders without a suffix use untuned scores; threshold-tuned rows use the calibrated scores.

**HOSP: encoders**

| Model | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|---:|---:|---:|---:|---:|
| BioMedBERT-110M | 30.74 | 19.91 | 54.99 | 78.21 | 22.00 |
| BioMedBERT-110M (threshold-tuned) | 22.91 | 29.55 | 32.83 | 59.16 | 20.71 |
| GatorTronS-345M | 35.36 | 23.12 | 58.84 | 81.64 | 25.77 |
| GatorTronS-345M (threshold-tuned) | 28.61 | 33.44 | 37.23 | 65.22 | 25.34 |
| M2-BERT-80M | 14.90 | 8.51 | 41.67 | 60.83 | 9.49 |
| M2-BERT-80M (threshold-tuned) | 11.72 | 12.05 | 24.87 | 40.47 | 8.34 |

**HOSP: generative LLMs, zero-shot**

| Model | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|---:|---:|---:|---:|---:|
| Llama 3.1-8B | 16.61 | 12.16 | 14.69 | 35.22 | 11.06 |
| Llama 3.1-8B-Instruct | 22.31 | 10.88 | 17.89 | 47.53 | 11.61 |
| Llama 3.1-70B | 17.29 | 9.78 | 14.99 | 35.21 | 9.66 |
| Llama 3.1-70B-Instruct | 23.04 | 10.40 | 15.15 | 52.20 | 11.33 |
| Mistral-7B | 10.45 | 12.02 | 9.67 | 27.25 | 8.50 |
| Mistral-7B-Instruct | 26.45 | 10.52 | 19.91 | 54.06 | 12.16 |
| Mistral-Nemo-13B | 14.87 | 11.30 | 14.10 | 32.83 | 10.16 |
| Mistral-Nemo-13B-Instruct | 22.23 | 9.55 | 16.47 | 50.35 | 10.37 |
| Qwen2-7B | 22.86 | 11.32 | 20.54 | 43.37 | 12.26 |
| Qwen2-7B-Instruct | 25.80 | 11.27 | 22.30 | 48.95 | 12.90 |
| Qwen2-72B | 24.63 | 11.06 | 21.77 | 46.17 | 12.48 |
| Qwen2-72B-Instruct | 27.24 | 11.09 | 24.10 | 51.58 | 13.13 |

**HOSP: generative LLMs, zero-shot CoT**

| Model | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|---:|---:|---:|---:|---:|
| Llama 3.1-8B | 14.85 | 10.89 | 13.06 | 37.96 | 9.68 |
| Llama 3.1-8B-Instruct | 20.84 | 9.93 | 14.82 | 49.61 | 10.11 |
| Llama 3.1-70B | 20.72 | 9.42 | 18.01 | 43.63 | 10.19 |
| Llama 3.1-70B-Instruct | 23.70 | 10.00 | 18.07 | 51.39 | 10.59 |
| Mistral-7B | 0.40 | 3.86 | 0.20 | 0.69 | 0.16 |
| Mistral-7B-Instruct | 23.96 | 9.46 | 17.30 | 49.76 | 10.85 |
| Mistral-Nemo-13B | 13.66 | 9.37 | 11.29 | 38.32 | 8.00 |
| Mistral-Nemo-13B-Instruct | 21.19 | 9.22 | 14.61 | 51.61 | 9.57 |
| Qwen2-7B | 21.04 | 11.62 | 18.80 | 42.01 | 12.04 |
| Qwen2-7B-Instruct | 24.63 | 10.76 | 19.88 | 49.38 | 12.15 |
| Qwen2-72B | 22.18 | 10.42 | 20.45 | 44.05 | 11.49 |
| Qwen2-72B-Instruct | 25.66 | 10.91 | 23.03 | 50.24 | 12.74 |

**HOSP: generative LLMs, 5-shot BM25**

| Model | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|---:|---:|---:|---:|---:|
| Llama 3.1-8B | 21.76 | 15.99 | 27.15 | 47.23 | 17.22 |
| Llama 3.1-8B-Instruct | 20.99 | 14.24 | 23.92 | 44.34 | 15.44 |
| Llama 3.1-70B | 20.55 | 12.67 | 26.18 | 49.05 | 15.09 |
| Llama 3.1-70B-Instruct | 21.89 | 14.43 | 26.05 | 45.49 | 15.81 |
| Mistral-7B | 19.15 | 13.78 | 23.64 | 44.13 | 15.29 |
| Mistral-7B-Instruct | 25.33 | 14.29 | 29.07 | 48.89 | 15.93 |
| Mistral-Nemo-13B | 19.18 | 13.77 | 23.60 | 45.97 | 15.10 |
| Mistral-Nemo-13B-Instruct | 21.15 | 15.27 | 26.02 | 48.41 | 16.05 |
| Qwen2-7B | 17.71 | 14.94 | 23.15 | 43.40 | 15.31 |
| Qwen2-7B-Instruct | 24.32 | 13.68 | 26.46 | 51.65 | 15.55 |
| Qwen2-72B | 22.57 | 14.35 | 27.74 | 49.44 | 16.46 |
| Qwen2-72B-Instruct | 25.29 | 15.83 | 31.26 | 53.00 | 17.58 |

**ICU: encoders**

| Model | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|---:|---:|---:|---:|---:|
| BioMedBERT-110M | 19.60 | 17.99 | 48.25 | 72.83 | 17.11 |
| BioMedBERT-110M (threshold-tuned) | 15.56 | 23.13 | 29.85 | 55.91 | 14.09 |
| GatorTronS-345M | 23.71 | 23.54 | 52.54 | 77.74 | 21.49 |
| GatorTronS-345M (threshold-tuned) | 20.00 | 29.62 | 33.10 | 63.66 | 18.89 |
| M2-BERT-80M | 10.16 | 8.27 | 33.65 | 56.05 | 8.14 |
| M2-BERT-80M (threshold-tuned) | 8.13 | 8.98 | 13.68 | 35.05 | 4.60 |

**ICU: generative LLMs, zero-shot**

| Model | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|---:|---:|---:|---:|---:|
| Llama 3.1-8B | 13.97 | 14.81 | 12.69 | 33.43 | 11.20 |
| Llama 3.1-8B-Instruct | 19.59 | 13.93 | 16.49 | 47.34 | 13.03 |
| Llama 3.1-70B | 15.40 | 12.44 | 13.22 | 33.60 | 10.48 |
| Llama 3.1-70B-Instruct | 20.31 | 13.36 | 14.95 | 50.37 | 12.55 |
| Mistral-7B | 8.84 | 13.49 | 7.68 | 25.66 | 8.07 |
| Mistral-7B-Instruct | 23.65 | 13.67 | 18.36 | 51.37 | 13.69 |
| Mistral-Nemo-13B | 13.49 | 14.31 | 12.25 | 31.63 | 10.86 |
| Mistral-Nemo-13B-Instruct | 19.64 | 11.97 | 15.95 | 48.39 | 11.58 |
| Qwen2-7B | 20.95 | 13.92 | 17.78 | 40.97 | 13.11 |
| Qwen2-7B-Instruct | 23.06 | 14.14 | 20.11 | 45.86 | 14.10 |
| Qwen2-72B | 22.55 | 14.06 | 18.46 | 43.28 | 13.70 |
| Qwen2-72B-Instruct | 24.11 | 14.20 | 21.29 | 48.11 | 14.42 |

**ICU: generative LLMs, zero-shot CoT**

| Model | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|---:|---:|---:|---:|---:|
| Llama 3.1-8B | 12.25 | 13.41 | 11.24 | 35.57 | 9.76 |
| Llama 3.1-8B-Instruct | 17.88 | 13.00 | 13.71 | 47.65 | 11.49 |
| Llama 3.1-70B | 18.73 | 11.87 | 15.73 | 40.81 | 11.22 |
| Llama 3.1-70B-Instruct | 21.24 | 12.66 | 18.00 | 50.18 | 12.08 |
| Mistral-7B | 0.34 | 2.63 | 0.72 | 1.38 | 0.13 |
| Mistral-7B-Instruct | 21.59 | 12.12 | 16.28 | 46.10 | 12.30 |
| Mistral-Nemo-13B | 11.11 | 11.63 | 9.67 | 37.51 | 8.23 |
| Mistral-Nemo-13B-Instruct | 19.03 | 11.96 | 14.03 | 48.54 | 10.78 |
| Qwen2-7B | 19.05 | 13.96 | 16.68 | 40.19 | 12.55 |
| Qwen2-7B-Instruct | 22.13 | 13.89 | 18.59 | 46.30 | 13.64 |
| Qwen2-72B | 20.32 | 13.14 | 17.80 | 40.93 | 12.56 |
| Qwen2-72B-Instruct | 22.78 | 14.10 | 21.00 | 47.16 | 14.12 |

**ICU: generative LLMs, 5-shot BM25**

| Model | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|---:|---:|---:|---:|---:|
| Llama 3.1-8B | 14.12 | 14.60 | 21.16 | 48.33 | 13.20 |
| Llama 3.1-8B-Instruct | 15.52 | 14.35 | 19.75 | 44.27 | 13.33 |
| Llama 3.1-70B | 13.74 | 12.11 | 20.08 | 50.85 | 12.13 |
| Llama 3.1-70B-Instruct | 17.06 | 15.38 | 21.36 | 44.90 | 14.47 |
| Mistral-7B | 11.08 | 11.91 | 17.21 | 43.16 | 10.78 |
| Mistral-7B-Instruct | 20.96 | 15.08 | 24.09 | 47.72 | 14.95 |
| Mistral-Nemo-13B | 11.31 | 12.73 | 17.66 | 44.95 | 11.01 |
| Mistral-Nemo-13B-Instruct | 11.24 | 13.35 | 17.82 | 43.28 | 11.37 |
| Qwen2-7B | 11.23 | 13.35 | 17.80 | 43.35 | 11.35 |
| Qwen2-7B-Instruct | 18.23 | 14.40 | 21.98 | 50.52 | 14.15 |
| Qwen2-72B | 15.27 | 13.79 | 21.96 | 49.40 | 13.32 |
| Qwen2-72B-Instruct | 17.42 | 15.73 | 24.99 | 52.06 | 14.77 |

### Few-shot retrievers (Qwen2-72B-Instruct)

Counterpart of Tables 3 and 10. "Gold heuristic" uses the gold labels to select demonstrations and is an upper bound, not a deployable retriever.

**HOSP: Qwen2-72B-Instruct, few-shot retrievers**

| Retriever | #FS | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|:---|---:|---:|---:|---:|---:|
| Zero-shot | 0 | 27.24 | 11.09 | 24.10 | 51.58 | 13.13 |
| Random | 1 | 21.53 | 11.77 | 21.44 | 42.66 | 12.70 |
| Random | 3 | 20.14 | 12.25 | 22.17 | 42.86 | 12.66 |
| Random | 5 | 18.90 | 12.32 | 22.16 | 41.59 | 12.48 |
| BM25 | 1 | 25.12 | 13.97 | 26.72 | 49.62 | 15.56 |
| BM25 | 3 | 25.48 | 15.17 | 29.83 | 52.32 | 16.92 |
| BM25 | 5 | 25.29 | 15.83 | 31.26 | 53.00 | 17.58 |
| PubMedBERT-NeuML | 1 | 22.38 | 11.90 | 23.33 | 46.86 | 13.32 |
| PubMedBERT-NeuML | 3 | 22.20 | 13.35 | 26.42 | 50.97 | 14.46 |
| PubMedBERT-NeuML | 5 | 21.87 | 13.87 | 28.19 | 52.63 | 14.90 |
| BioMedBERT-ft | 1 | 23.05 | 12.59 | 23.89 | 46.59 | 13.78 |
| BioMedBERT-ft | 3 | 23.52 | 13.89 | 28.15 | 51.24 | 15.34 |
| BioMedBERT-ft | 5 | 24.25 | 14.63 | 30.49 | 53.04 | 16.30 |
| Gold heuristic | 1 | 27.04 | 15.80 | 31.16 | 51.36 | 16.68 |
| Gold heuristic | 3 | 30.42 | 19.50 | 42.26 | 60.04 | 20.71 |
| Gold heuristic | 5 | 32.02 | 20.78 | 46.89 | 62.62 | 22.43 |

**ICU: Qwen2-72B-Instruct, few-shot retrievers**

| Retriever | #FS | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|:---|---:|---:|---:|---:|---:|
| Zero-shot | 0 | 24.11 | 14.20 | 21.29 | 48.11 | 14.42 |
| Random | 1 | 17.67 | 13.95 | 17.74 | 40.80 | 12.81 |
| Random | 3 | 15.90 | 13.27 | 18.05 | 41.22 | 12.04 |
| Random | 5 | 14.24 | 13.45 | 17.84 | 40.66 | 11.53 |
| BM25 | 1 | 19.97 | 15.29 | 21.41 | 47.27 | 14.62 |
| BM25 | 3 | 18.89 | 15.89 | 23.90 | 51.18 | 15.08 |
| BM25 | 5 | 17.42 | 15.73 | 24.99 | 52.06 | 14.77 |
| PubMedBERT-NeuML | 1 | 18.25 | 13.83 | 19.76 | 45.59 | 13.17 |
| PubMedBERT-NeuML | 3 | 17.13 | 13.97 | 22.50 | 50.50 | 13.38 |
| PubMedBERT-NeuML | 5 | 16.10 | 14.48 | 23.79 | 52.34 | 13.46 |
| BioMedBERT-ft | 1 | 18.97 | 14.60 | 19.96 | 44.80 | 13.73 |
| BioMedBERT-ft | 3 | 18.04 | 14.86 | 23.50 | 50.92 | 14.13 |
| BioMedBERT-ft | 5 | 18.12 | 15.44 | 25.30 | 52.59 | 14.81 |
| Gold heuristic | 1 | 21.32 | 17.84 | 25.05 | 48.46 | 16.00 |
| Gold heuristic | 3 | 22.72 | 20.24 | 35.75 | 59.15 | 18.63 |
| Gold heuristic | 5 | 22.83 | 21.21 | 40.06 | 61.66 | 19.70 |

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
