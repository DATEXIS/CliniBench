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
* **Predictions.** The top 20 codes, truncated to short codes and de-duplicated by first occurrence. Encoders rank by score, LLMs by generation order. LLM lists can be shorter than 20 after de-duplication. Encoders are additionally scored at their own decision thresholds, without the top-20 cut ([see below](#encoders-at-their-own-operating-point)).
* **Metrics.** The main tables report micro and macro F1. `results/all_runs.md` adds MAP@20, MD Acc. and micro/macro recall and precision for every run.
  * Micro scores pool true positives, false positives and false negatives over all (note, code) pairs.
  * Macro scores are computed per class over the label space (0 when undefined) and averaged with equal class weights. Macro F1 is the mean of per-class F1.
  * MAP@20 is the per-note average precision normalised by min(|gold|, 20).
  * MD Acc. is the share of notes whose first gold code is in the top 20.
* **Aggregation.** Unweighted mean of the ICD-9 and ICD-10 partitions of each care unit, as in the paper.

### What changes in the conclusions

* **Encoders still lead.** GatorTronS (top 20 by raw score) is best on micro F1, macro F1, MAP@20 and MD Acc. in both care units. The best LLM without oracle retrieval is Qwen2-72B-Instruct with BM25 demonstrations.
  * Micro F1: HOSP 48.03 vs. 35.64, ICU 50.06 vs. 32.21.
  * Macro F1: HOSP 25.77 vs. 17.58, ICU 21.49 vs. 15.08.
  * In ICU the macro F1 gap shrinks from about 11 points in the paper (encoder F1 printed as 25.77) to about 6.
* **The MAP gap is larger than reported.** It grows once both families are ranked the same way (HOSP 58.84 vs. 31.26, ICU 52.54 vs. 25.30; see `results/all_runs.md`). Only 2–3 points of it come from the LLMs' shorter lists ([see above](#main-results)).
* **ICU macro recall:** the best zero-shot LLM (Qwen2-72B-Instruct, 24.11) is now slightly above the best encoder (GatorTronS, 23.71; printed as 28.61).
* **Threshold tuning depends on the protocol.** Within the top-20 budget, tuned scores lower micro F1, macro F1, MAP, recall and MD accuracy for all three encoders and raise only macro precision. Used as a classifier, the tuned thresholds give every encoder its best macro F1 (GatorTronS: HOSP 28.28, ICU 23.62). They do so by predicting many more codes (32–90 per note), which halves micro F1. The 0.5 threshold does the opposite. It predicts few codes (5–9 per note) and has low macro F1, but it gives BioMedBERT and GatorTronS their highest micro F1 (GatorTronS: HOSP 56.09, ICU 51.45).
* **Prompting:** 5 BM25 demonstrations raise micro F1 and MAP for all 12 LLMs in both units. Zero-shot CoT (in `results/all_runs.md`) improves macro F1 for only 1 and micro F1 for only 2 of 12 models in each unit.

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

Counterpart of Tables 2 and 11. All values use the top-20 predictions and are the unweighted mean of the ICD-9 and ICD-10 partitions of each care unit, as in the paper. Encoders are fine-tuned, rank labels by raw score and use no demonstrations, so they appear in the zero-shot columns. Micro F1 pools all (note, code) decisions and is dominated by frequent codes. Macro F1 weights every class equally and is dominated by the long tail. **Bold**: best per column; <ins>underlined</ins>: best generative LLM where it is not the best overall. Zero-shot CoT, MAP@20, MD Acc. and micro/macro recall and precision for every run are in [`results/all_runs.md`](results/all_runs.md).

MAP@20 is not in the main tables because it favours longer prediction lists: appending a code never lowers average precision, and wrong codes at the end cost nothing. Encoders always return 20 codes, while LLM lists average about 13 unique codes after pruning to short codes (5–6 for zero-shot Mistral-7B). The bias is small compared with the encoder–LLM gap. In spot checks on ICU runs, cutting each GatorTronS list to the length of the LLM's list for the same note lowered its MAP@20 by only 2–3 of roughly 30 points. Between LLMs with very different list lengths, such as zero-shot Mistral-7B versus the instruct models, MAP@20 can largely reflect list length.

**HOSP**

| Model | Zero-shot Micro F1 | Zero-shot Macro F1 | 5-shot BM25 Micro F1 | 5-shot BM25 Macro F1 |
|:---|---:|---:|---:|---:|
| *Encoders* | | | | |
| BioMedBERT-110M | 45.83 | 22.00 | – | – |
| GatorTronS-345M | **48.03** | **25.77** | – | – |
| M2-BERT-80M | 38.72 | 9.49 | – | – |
| *Generative LLMs* | | | | |
| Llama 3.1-8B | 22.10 | 11.06 | 35.01 | 17.22 |
| Llama 3.1-8B-Instruct | 24.71 | 11.61 | 32.18 | 15.44 |
| Llama 3.1-70B | 19.44 | 9.66 | 32.14 | 15.09 |
| Llama 3.1-70B-Instruct | 22.04 | 11.33 | 32.76 | 15.81 |
| Mistral-7B | 15.60 | 8.50 | 31.14 | 15.29 |
| Mistral-7B-Instruct | 25.06 | 12.16 | 32.88 | 15.93 |
| Mistral-Nemo-13B | 21.46 | 10.16 | 31.06 | 15.10 |
| Mistral-Nemo-13B-Instruct | 20.84 | 10.37 | 32.52 | 16.05 |
| Qwen2-7B | 26.09 | 12.26 | 31.01 | 15.31 |
| Qwen2-7B-Instruct | 27.28 | 12.90 | 32.04 | 15.55 |
| Qwen2-72B | 26.96 | 12.48 | 33.84 | 16.46 |
| Qwen2-72B-Instruct | <ins>28.53</ins> | <ins>13.13</ins> | **35.64** | **17.58** |

**ICU**

| Model | Zero-shot Micro F1 | Zero-shot Macro F1 | 5-shot BM25 Micro F1 | 5-shot BM25 Macro F1 |
|:---|---:|---:|---:|---:|
| *Encoders* | | | | |
| BioMedBERT-110M | 47.33 | 17.11 | – | – |
| GatorTronS-345M | **50.06** | **21.49** | – | – |
| M2-BERT-80M | 38.54 | 8.14 | – | – |
| *Generative LLMs* | | | | |
| Llama 3.1-8B | 20.87 | 11.20 | 29.91 | 13.20 |
| Llama 3.1-8B-Instruct | 25.81 | 13.03 | 29.27 | 13.33 |
| Llama 3.1-70B | 19.69 | 10.48 | 28.64 | 12.13 |
| Llama 3.1-70B-Instruct | 24.59 | 12.55 | 30.08 | 14.47 |
| Mistral-7B | 13.99 | 8.07 | 26.07 | 10.78 |
| Mistral-7B-Instruct | 26.55 | 13.69 | 31.08 | **14.95** |
| Mistral-Nemo-13B | 20.22 | 10.86 | 26.18 | 11.01 |
| Mistral-Nemo-13B-Instruct | 23.08 | 11.58 | 26.44 | 11.37 |
| Qwen2-7B | 25.33 | 13.11 | 26.43 | 11.35 |
| Qwen2-7B-Instruct | 27.73 | 14.10 | 30.22 | 14.15 |
| Qwen2-72B | 26.84 | 13.70 | 30.07 | 13.32 |
| Qwen2-72B-Instruct | <ins>29.50</ins> | <ins>14.42</ins> | **32.21** | 14.77 |

### Encoders at their own operating point

The top-20 budget is the paper's shared protocol, but it is not how an encoder classifier is normally used. Each encoder dump also contains the labels the model predicts on its own, and the table below scores both:

* **top-20, raw scores:** the encoder rows of the main table.
* **top-20, tuned scores:** the top 20 by score after dividing by the per-class thresholds. This is what the paper reported as "-tuned". Tuning only reorders the labels here, which moves rare labels with low thresholds ahead of frequent correct ones.
* **threshold 0.5:** every label with probability above 0.5 (`logits > 0` in `bert_model.py`).
* **tuned thresholds:** every label above its own threshold tuned for F1 on the validation split (`tune_thresholds` in `bert_model.py`).

These sets are unranked, so MAP@20 is not defined for them. *Codes/note* is the average number of unique predicted codes.

| Encoder, setting | HOSP Codes/note | HOSP Micro F1 | HOSP Macro F1 | ICU Codes/note | ICU Micro F1 | ICU Macro F1 |
|:---|---:|---:|---:|---:|---:|---:|
| BioMedBERT-110M, top-20, raw scores | 20.00 | 45.83 | 22.00 | 20.00 | 47.33 | 17.11 |
| BioMedBERT-110M, top-20, tuned scores | 20.00 | 30.88 | 20.71 | 20.00 | 31.52 | 14.09 |
| BioMedBERT-110M, threshold 0.5 | 6.35 | 53.06 | 16.83 | 8.61 | 47.97 | 12.42 |
| BioMedBERT-110M, tuned thresholds | 37.31 | 25.72 | 24.32 | 63.21 | 21.78 | 19.17 |
| GatorTronS-345M, top-20, raw scores | 20.00 | 48.03 | 25.77 | 20.00 | 50.06 | 21.49 |
| GatorTronS-345M, top-20, tuned scores | 20.00 | 33.55 | 25.34 | 20.00 | 34.58 | 18.89 |
| GatorTronS-345M, threshold 0.5 | 6.62 | 56.09 | 20.94 | 8.94 | 51.45 | 15.97 |
| GatorTronS-345M, tuned thresholds | 31.80 | 30.05 | 28.28 | 56.38 | 25.25 | 23.62 |
| M2-BERT-80M, top-20, raw scores | 20.00 | 38.72 | 9.49 | 20.00 | 38.54 | 8.14 |
| M2-BERT-80M, top-20, tuned scores | 20.00 | 24.32 | 8.34 | 20.00 | 18.22 | 4.60 |
| M2-BERT-80M, threshold 0.5 | 4.59 | 40.04 | 4.78 | 6.01 | 33.34 | 4.30 |
| M2-BERT-80M, tuned thresholds | 59.02 | 16.43 | 12.03 | 89.86 | 15.44 | 10.28 |

### Few-shot retrievers (Qwen2-72B-Instruct)

Counterpart of Tables 3 and 10. "Gold heuristic" uses the gold labels to select demonstrations and is an upper bound, not a deployable retriever.

| Demonstrations | HOSP Micro F1 | HOSP Macro F1 | ICU Micro F1 | ICU Macro F1 |
|:---|---:|---:|---:|---:|
| none (zero-shot) | 28.53 | 13.13 | 29.50 | 14.42 |
| Random, 1-shot | 27.10 | 12.70 | 26.19 | 12.81 |
| Random, 3-shot | 28.12 | 12.66 | 26.33 | 12.04 |
| Random, 5-shot | 28.30 | 12.48 | 26.14 | 11.53 |
| BM25, 1-shot | 31.97 | 15.56 | 29.59 | 14.62 |
| BM25, 3-shot | 34.50 | 16.92 | 31.44 | 15.08 |
| BM25, 5-shot | 35.64 | 17.58 | 32.21 | 14.77 |
| PubMedBERT-NeuML, 1-shot | 28.24 | 13.32 | 27.84 | 13.17 |
| PubMedBERT-NeuML, 3-shot | 30.48 | 14.46 | 29.74 | 13.38 |
| PubMedBERT-NeuML, 5-shot | 31.75 | 14.90 | 30.64 | 13.46 |
| BioMedBERT-ft, 1-shot | 28.84 | 13.78 | 28.14 | 13.73 |
| BioMedBERT-ft, 3-shot | 32.29 | 15.34 | 30.74 | 14.13 |
| BioMedBERT-ft, 5-shot | 34.08 | 16.30 | 32.19 | 14.81 |
| Gold heuristic, 1-shot | 34.16 | 16.68 | 32.64 | 16.00 |
| Gold heuristic, 3-shot | 42.42 | 20.71 | 41.39 | 18.63 |
| Gold heuristic, 5-shot | 45.88 | 22.43 | 44.75 | 19.70 |

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
