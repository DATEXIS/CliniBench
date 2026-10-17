# CliniBench: all runs (unified evaluation)

Macro-averaged over the shared label space, top-20 predictions, unweighted mean of ICD-9 and ICD-10. Produced by `src/evaluation/evaluate_runs.py tables`; per-partition values in `unified_metrics.csv`.

## HOSP

| Model | Setting | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|:---|---:|---:|---:|---:|---:|
| BioMedBERT-110M | threshold-tuned | 22.91 | 29.55 | 32.83 | 59.16 | 20.71 |
| BioMedBERT-110M | untuned | 30.74 | 19.91 | 54.99 | 78.21 | 22.00 |
| GatorTronS-345M | threshold-tuned | 28.61 | 33.44 | 37.23 | 65.22 | 25.34 |
| GatorTronS-345M | untuned | 35.36 | 23.12 | 58.84 | 81.64 | 25.77 |
| M2-BERT-80M | threshold-tuned | 11.72 | 12.05 | 24.87 | 40.47 | 8.34 |
| M2-BERT-80M | untuned | 14.90 | 8.51 | 41.67 | 60.83 | 9.49 |
| Llama 3.1-8B | zero-shot | 16.61 | 12.16 | 14.69 | 35.22 | 11.06 |
| Llama 3.1-8B | zero-shot CoT | 14.85 | 10.89 | 13.06 | 37.96 | 9.68 |
| Llama 3.1-8B | 1-shot Random | 15.41 | 10.18 | 15.25 | 33.98 | 10.76 |
| Llama 3.1-8B | 3-shot Random | 15.85 | 12.19 | 17.61 | 35.09 | 11.72 |
| Llama 3.1-8B | 5-shot Random | 15.46 | 12.43 | 18.22 | 35.00 | 11.75 |
| Llama 3.1-8B | 1-shot BM25 | 20.88 | 14.17 | 22.04 | 45.06 | 15.76 |
| Llama 3.1-8B | 3-shot BM25 | 21.47 | 15.48 | 25.75 | 46.59 | 16.80 |
| Llama 3.1-8B | 5-shot BM25 | 21.76 | 15.99 | 27.15 | 47.23 | 17.22 |
| Llama 3.1-8B | 1-shot PubMedBERT-NeuML | 15.53 | 10.51 | 16.75 | 42.61 | 11.37 |
| Llama 3.1-8B | 3-shot PubMedBERT-NeuML | 16.12 | 12.21 | 21.09 | 46.00 | 12.81 |
| Llama 3.1-8B | 5-shot PubMedBERT-NeuML | 17.12 | 13.23 | 22.99 | 48.26 | 13.74 |
| Llama 3.1-8B | 1-shot BioMedBERT-ft | 17.56 | 12.83 | 19.33 | 43.63 | 13.47 |
| Llama 3.1-8B | 3-shot BioMedBERT-ft | 18.62 | 14.06 | 23.63 | 46.56 | 14.71 |
| Llama 3.1-8B | 5-shot BioMedBERT-ft | 19.22 | 14.85 | 25.77 | 48.25 | 15.43 |
| Llama 3.1-8B | 1-shot Gold heuristic | 27.08 | 22.85 | 36.91 | 54.58 | 22.24 |
| Llama 3.1-8B | 3-shot Gold heuristic | 28.72 | 25.52 | 44.22 | 58.25 | 24.93 |
| Llama 3.1-8B | 5-shot Gold heuristic | 29.86 | 26.08 | 47.19 | 60.76 | 25.88 |
| Llama 3.1-8B-Instruct | zero-shot | 22.31 | 10.88 | 17.89 | 47.53 | 11.61 |
| Llama 3.1-8B-Instruct | zero-shot CoT | 20.84 | 9.93 | 14.82 | 49.61 | 10.11 |
| Llama 3.1-8B-Instruct | 1-shot Random | 13.20 | 9.40 | 10.95 | 27.44 | 9.38 |
| Llama 3.1-8B-Instruct | 3-shot Random | 17.22 | 11.08 | 16.37 | 35.16 | 11.44 |
| Llama 3.1-8B-Instruct | 5-shot Random | 18.80 | 11.55 | 17.96 | 38.01 | 12.12 |
| Llama 3.1-8B-Instruct | 1-shot BM25 | 17.27 | 12.39 | 16.47 | 36.05 | 12.81 |
| Llama 3.1-8B-Instruct | 3-shot BM25 | 19.34 | 13.72 | 20.98 | 40.35 | 14.45 |
| Llama 3.1-8B-Instruct | 5-shot BM25 | 20.99 | 14.24 | 23.92 | 44.34 | 15.44 |
| Llama 3.1-8B-Instruct | 1-shot PubMedBERT-NeuML | 13.17 | 9.94 | 12.12 | 31.32 | 9.91 |
| Llama 3.1-8B-Instruct | 3-shot PubMedBERT-NeuML | 15.29 | 11.14 | 16.76 | 37.80 | 11.38 |
| Llama 3.1-8B-Instruct | 5-shot PubMedBERT-NeuML | 17.00 | 11.92 | 19.87 | 42.88 | 12.53 |
| Llama 3.1-8B-Instruct | 1-shot BioMedBERT-ft | 13.53 | 11.10 | 12.72 | 29.57 | 10.60 |
| Llama 3.1-8B-Instruct | 3-shot BioMedBERT-ft | 15.60 | 12.68 | 17.61 | 35.82 | 12.43 |
| Llama 3.1-8B-Instruct | 5-shot BioMedBERT-ft | 17.54 | 13.21 | 20.92 | 40.37 | 13.54 |
| Llama 3.1-8B-Instruct | 1-shot Gold heuristic | 17.62 | 17.84 | 20.96 | 34.29 | 14.96 |
| Llama 3.1-8B-Instruct | 3-shot Gold heuristic | 21.89 | 19.72 | 31.93 | 44.90 | 18.23 |
| Llama 3.1-8B-Instruct | 5-shot Gold heuristic | 24.67 | 20.99 | 38.66 | 52.16 | 20.19 |
| Llama 3.1-70B | zero-shot | 17.29 | 9.78 | 14.99 | 35.21 | 9.66 |
| Llama 3.1-70B | zero-shot CoT | 20.72 | 9.42 | 18.01 | 43.63 | 10.19 |
| Llama 3.1-70B | 1-shot Random | 9.38 | 6.22 | 10.27 | 23.15 | 6.83 |
| Llama 3.1-70B | 3-shot Random | 10.23 | 6.79 | 13.81 | 27.14 | 7.48 |
| Llama 3.1-70B | 5-shot Random | 11.95 | 8.27 | 16.25 | 31.01 | 8.88 |
| Llama 3.1-70B | 1-shot BM25 | 17.58 | 13.10 | 19.83 | 40.27 | 14.24 |
| Llama 3.1-70B | 3-shot BM25 | 20.09 | 12.31 | 24.45 | 47.78 | 14.70 |
| Llama 3.1-70B | 5-shot BM25 | 20.55 | 12.67 | 26.18 | 49.05 | 15.09 |
| Llama 3.1-70B | 1-shot PubMedBERT-NeuML | 10.66 | 8.80 | 13.93 | 37.78 | 8.88 |
| Llama 3.1-70B | 3-shot PubMedBERT-NeuML | 13.04 | 8.74 | 18.82 | 47.83 | 9.91 |
| Llama 3.1-70B | 5-shot PubMedBERT-NeuML | 14.39 | 10.05 | 21.39 | 49.67 | 11.10 |
| Llama 3.1-70B | 1-shot BioMedBERT-ft | 13.46 | 11.45 | 16.77 | 38.41 | 11.44 |
| Llama 3.1-70B | 3-shot BioMedBERT-ft | 17.10 | 11.77 | 22.38 | 47.74 | 13.00 |
| Llama 3.1-70B | 5-shot BioMedBERT-ft | 18.74 | 12.42 | 25.28 | 50.54 | 13.96 |
| Llama 3.1-70B | 1-shot Gold heuristic | 24.23 | 23.34 | 36.97 | 51.01 | 21.79 |
| Llama 3.1-70B | 3-shot Gold heuristic | 28.34 | 24.69 | 46.19 | 58.57 | 25.06 |
| Llama 3.1-70B | 5-shot Gold heuristic | 30.54 | 24.55 | 50.09 | 61.80 | 25.83 |
| Llama 3.1-70B-Instruct | zero-shot | 23.04 | 10.40 | 15.15 | 52.20 | 11.33 |
| Llama 3.1-70B-Instruct | zero-shot CoT | 23.70 | 10.00 | 18.07 | 51.39 | 10.59 |
| Llama 3.1-70B-Instruct | 1-shot Random | 17.37 | 11.63 | 16.09 | 32.20 | 11.63 |
| Llama 3.1-70B-Instruct | 3-shot Random | 21.99 | 12.30 | 22.20 | 42.99 | 13.37 |
| Llama 3.1-70B-Instruct | 5-shot Random | 20.13 | 12.51 | 21.18 | 40.32 | 13.15 |
| Llama 3.1-70B-Instruct | 1-shot BM25 | 18.57 | 13.56 | 18.81 | 35.09 | 13.49 |
| Llama 3.1-70B-Instruct | 3-shot BM25 | 22.78 | 14.00 | 26.01 | 46.80 | 15.63 |
| Llama 3.1-70B-Instruct | 5-shot BM25 | 21.89 | 14.43 | 26.05 | 45.49 | 15.81 |
| Llama 3.1-70B-Instruct | 1-shot PubMedBERT-NeuML | 15.38 | 11.45 | 15.34 | 30.83 | 11.15 |
| Llama 3.1-70B-Instruct | 3-shot PubMedBERT-NeuML | 19.76 | 11.96 | 22.72 | 44.88 | 13.21 |
| Llama 3.1-70B-Instruct | 5-shot PubMedBERT-NeuML | 18.43 | 12.38 | 22.90 | 44.29 | 13.24 |
| Llama 3.1-70B-Instruct | 1-shot BioMedBERT-ft | 14.82 | 11.73 | 14.82 | 29.31 | 11.03 |
| Llama 3.1-70B-Instruct | 3-shot BioMedBERT-ft | 19.72 | 12.85 | 22.87 | 42.86 | 13.88 |
| Llama 3.1-70B-Instruct | 5-shot BioMedBERT-ft | 18.97 | 13.25 | 23.85 | 43.02 | 14.15 |
| Llama 3.1-70B-Instruct | 1-shot Gold heuristic | 16.98 | 15.66 | 19.43 | 32.14 | 13.41 |
| Llama 3.1-70B-Instruct | 3-shot Gold heuristic | 25.89 | 18.21 | 36.12 | 51.28 | 18.83 |
| Llama 3.1-70B-Instruct | 5-shot Gold heuristic | 26.42 | 19.36 | 40.11 | 53.82 | 20.08 |
| Mistral-7B | zero-shot | 10.45 | 12.02 | 9.67 | 27.25 | 8.50 |
| Mistral-7B | zero-shot CoT | 0.40 | 3.86 | 0.20 | 0.69 | 0.16 |
| Mistral-7B | 1-shot Random | 5.67 | 4.26 | 5.83 | 14.89 | 4.58 |
| Mistral-7B | 3-shot Random | 10.53 | 8.31 | 12.61 | 26.34 | 8.39 |
| Mistral-7B | 5-shot Random | 10.78 | 8.85 | 13.34 | 27.11 | 8.75 |
| Mistral-7B | 1-shot BM25 | 16.28 | 13.02 | 17.94 | 36.83 | 13.89 |
| Mistral-7B | 3-shot BM25 | 18.56 | 13.24 | 22.30 | 43.34 | 14.83 |
| Mistral-7B | 5-shot BM25 | 19.15 | 13.78 | 23.64 | 44.13 | 15.29 |
| Mistral-7B | 1-shot PubMedBERT-NeuML | 9.45 | 8.88 | 12.65 | 35.40 | 8.40 |
| Mistral-7B | 3-shot PubMedBERT-NeuML | 12.76 | 10.39 | 17.74 | 43.52 | 10.60 |
| Mistral-7B | 5-shot PubMedBERT-NeuML | 13.65 | 10.94 | 19.65 | 45.23 | 11.38 |
| Mistral-7B | 1-shot BioMedBERT-ft | 11.84 | 11.19 | 15.29 | 35.55 | 10.78 |
| Mistral-7B | 3-shot BioMedBERT-ft | 15.55 | 12.40 | 20.52 | 43.83 | 12.92 |
| Mistral-7B | 5-shot BioMedBERT-ft | 16.60 | 12.73 | 22.80 | 46.10 | 13.53 |
| Mistral-7B | 1-shot Gold heuristic | 23.56 | 26.68 | 37.54 | 48.83 | 23.44 |
| Mistral-7B | 3-shot Gold heuristic | 27.31 | 26.27 | 44.48 | 56.52 | 25.27 |
| Mistral-7B | 5-shot Gold heuristic | 28.68 | 25.36 | 47.51 | 58.75 | 25.46 |
| Mistral-7B-Instruct | zero-shot | 26.45 | 10.52 | 19.91 | 54.06 | 12.16 |
| Mistral-7B-Instruct | zero-shot CoT | 23.96 | 9.46 | 17.30 | 49.76 | 10.85 |
| Mistral-7B-Instruct | 1-shot Random | 24.83 | 10.47 | 23.37 | 46.39 | 12.35 |
| Mistral-7B-Instruct | 3-shot Random | 23.52 | 10.84 | 22.97 | 42.97 | 12.30 |
| Mistral-7B-Instruct | 5-shot Random | 22.20 | 10.96 | 22.66 | 41.20 | 12.23 |
| Mistral-7B-Instruct | 1-shot BM25 | 27.36 | 12.36 | 26.61 | 50.67 | 14.39 |
| Mistral-7B-Instruct | 3-shot BM25 | 25.73 | 13.61 | 28.05 | 49.12 | 15.31 |
| Mistral-7B-Instruct | 5-shot BM25 | 25.33 | 14.29 | 29.07 | 48.89 | 15.93 |
| Mistral-7B-Instruct | 1-shot PubMedBERT-NeuML | 25.50 | 10.91 | 24.35 | 48.18 | 12.86 |
| Mistral-7B-Instruct | 3-shot PubMedBERT-NeuML | 23.98 | 11.85 | 25.54 | 47.49 | 13.32 |
| Mistral-7B-Instruct | 5-shot PubMedBERT-NeuML | 22.76 | 12.55 | 26.50 | 47.85 | 13.84 |
| Mistral-7B-Instruct | 1-shot BioMedBERT-ft | 25.90 | 11.18 | 24.70 | 48.87 | 13.10 |
| Mistral-7B-Instruct | 3-shot BioMedBERT-ft | 25.07 | 12.42 | 26.75 | 48.43 | 14.12 |
| Mistral-7B-Instruct | 5-shot BioMedBERT-ft | 24.45 | 13.29 | 28.23 | 48.72 | 14.84 |
| Mistral-7B-Instruct | 1-shot Gold heuristic | 28.38 | 13.44 | 29.96 | 52.42 | 15.02 |
| Mistral-7B-Instruct | 3-shot Gold heuristic | 29.93 | 16.60 | 38.61 | 55.83 | 18.06 |
| Mistral-7B-Instruct | 5-shot Gold heuristic | 30.71 | 18.38 | 43.00 | 57.35 | 19.76 |
| Mistral-Nemo-13B | zero-shot | 14.87 | 11.30 | 14.10 | 32.83 | 10.16 |
| Mistral-Nemo-13B | zero-shot CoT | 13.66 | 9.37 | 11.29 | 38.32 | 8.00 |
| Mistral-Nemo-13B | 1-shot Random | 9.33 | 7.17 | 7.58 | 22.06 | 7.18 |
| Mistral-Nemo-13B | 3-shot Random | 10.15 | 8.48 | 12.01 | 27.88 | 8.02 |
| Mistral-Nemo-13B | 5-shot Random | 10.68 | 9.14 | 13.28 | 30.18 | 8.42 |
| Mistral-Nemo-13B | 1-shot BM25 | 18.13 | 12.90 | 17.86 | 39.15 | 13.93 |
| Mistral-Nemo-13B | 3-shot BM25 | 19.01 | 13.08 | 22.21 | 45.09 | 14.69 |
| Mistral-Nemo-13B | 5-shot BM25 | 19.18 | 13.77 | 23.60 | 45.97 | 15.10 |
| Mistral-Nemo-13B | 1-shot PubMedBERT-NeuML | 11.63 | 9.51 | 12.20 | 34.17 | 9.33 |
| Mistral-Nemo-13B | 3-shot PubMedBERT-NeuML | 12.68 | 9.70 | 17.05 | 43.58 | 10.11 |
| Mistral-Nemo-13B | 5-shot PubMedBERT-NeuML | 13.59 | 10.41 | 19.20 | 46.52 | 10.86 |
| Mistral-Nemo-13B | 1-shot BioMedBERT-ft | 13.03 | 11.50 | 14.01 | 34.31 | 10.86 |
| Mistral-Nemo-13B | 3-shot BioMedBERT-ft | 15.24 | 11.92 | 19.24 | 42.75 | 12.26 |
| Mistral-Nemo-13B | 5-shot BioMedBERT-ft | 16.80 | 12.55 | 22.22 | 46.69 | 13.15 |
| Mistral-Nemo-13B | 1-shot Gold heuristic | 21.43 | 23.19 | 30.59 | 44.12 | 19.91 |
| Mistral-Nemo-13B | 3-shot Gold heuristic | 26.45 | 25.02 | 41.54 | 55.04 | 23.88 |
| Mistral-Nemo-13B | 5-shot Gold heuristic | 28.46 | 24.88 | 46.24 | 59.18 | 24.77 |
| Mistral-Nemo-13B-Instruct | zero-shot | 22.23 | 9.55 | 16.47 | 50.35 | 10.37 |
| Mistral-Nemo-13B-Instruct | zero-shot CoT | 21.19 | 9.22 | 14.61 | 51.61 | 9.57 |
| Mistral-Nemo-13B-Instruct | 1-shot Random | 14.55 | 8.97 | 13.92 | 33.07 | 9.78 |
| Mistral-Nemo-13B-Instruct | 3-shot Random | 13.02 | 10.70 | 14.57 | 32.39 | 10.20 |
| Mistral-Nemo-13B-Instruct | 5-shot Random | 11.76 | 11.12 | 14.39 | 30.61 | 10.02 |
| Mistral-Nemo-13B-Instruct | 1-shot BM25 | 19.84 | 13.32 | 20.63 | 43.33 | 14.70 |
| Mistral-Nemo-13B-Instruct | 3-shot BM25 | 18.17 | 14.45 | 22.44 | 43.96 | 15.21 |
| Mistral-Nemo-13B-Instruct | 5-shot BM25 | 21.15 | 15.27 | 26.02 | 48.41 | 16.05 |
| Mistral-Nemo-13B-Instruct | 1-shot PubMedBERT-NeuML | 14.90 | 9.70 | 15.96 | 41.27 | 10.63 |
| Mistral-Nemo-13B-Instruct | 3-shot PubMedBERT-NeuML | 13.43 | 10.82 | 18.33 | 43.90 | 11.11 |
| Mistral-Nemo-13B-Instruct | 5-shot PubMedBERT-NeuML | 13.14 | 11.99 | 19.64 | 44.32 | 11.71 |
| Mistral-Nemo-13B-Instruct | 1-shot BioMedBERT-ft | 16.89 | 11.22 | 14.75 | 34.49 | 11.21 |
| Mistral-Nemo-13B-Instruct | 3-shot BioMedBERT-ft | 25.57 | 12.99 | 26.84 | 53.94 | 14.95 |
| Mistral-Nemo-13B-Instruct | 5-shot BioMedBERT-ft | 25.34 | 14.05 | 28.64 | 54.83 | 15.78 |
| Mistral-Nemo-13B-Instruct | 1-shot Gold heuristic | 25.19 | 19.94 | 33.60 | 52.36 | 19.64 |
| Mistral-Nemo-13B-Instruct | 3-shot Gold heuristic | 24.81 | 25.07 | 41.29 | 54.18 | 23.10 |
| Mistral-Nemo-13B-Instruct | 5-shot Gold heuristic | 25.03 | 27.00 | 44.74 | 55.01 | 24.32 |
| Qwen2-7B | zero-shot | 22.86 | 11.32 | 20.54 | 43.37 | 12.26 |
| Qwen2-7B | zero-shot CoT | 21.04 | 11.62 | 18.80 | 42.01 | 12.04 |
| Qwen2-7B | 1-shot Random | 14.56 | 9.00 | 13.89 | 32.97 | 9.79 |
| Qwen2-7B | 3-shot Random | 13.03 | 10.70 | 14.57 | 32.36 | 10.20 |
| Qwen2-7B | 5-shot Random | 11.75 | 11.14 | 14.39 | 30.56 | 10.03 |
| Qwen2-7B | 1-shot BM25 | 19.83 | 13.31 | 20.62 | 43.35 | 14.69 |
| Qwen2-7B | 3-shot BM25 | 18.15 | 14.43 | 22.43 | 43.93 | 15.19 |
| Qwen2-7B | 5-shot BM25 | 17.71 | 14.94 | 23.15 | 43.40 | 15.31 |
| Qwen2-7B | 1-shot PubMedBERT-NeuML | 14.89 | 9.72 | 15.96 | 41.23 | 10.63 |
| Qwen2-7B | 3-shot PubMedBERT-NeuML | 13.41 | 10.81 | 18.33 | 43.88 | 11.10 |
| Qwen2-7B | 5-shot PubMedBERT-NeuML | 13.12 | 11.99 | 19.64 | 44.32 | 11.70 |
| Qwen2-7B | 1-shot BioMedBERT-ft | 17.31 | 11.74 | 18.44 | 43.04 | 12.44 |
| Qwen2-7B | 3-shot BioMedBERT-ft | 15.96 | 13.24 | 20.94 | 43.72 | 13.42 |
| Qwen2-7B | 5-shot BioMedBERT-ft | 16.02 | 14.03 | 22.84 | 44.44 | 13.96 |
| Qwen2-7B | 1-shot Gold heuristic | 25.26 | 20.01 | 33.58 | 52.26 | 19.71 |
| Qwen2-7B | 3-shot Gold heuristic | 24.77 | 24.98 | 41.24 | 54.11 | 23.08 |
| Qwen2-7B | 5-shot Gold heuristic | 25.14 | 27.19 | 44.73 | 54.98 | 24.41 |
| Qwen2-7B-Instruct | zero-shot | 25.80 | 11.27 | 22.30 | 48.95 | 12.90 |
| Qwen2-7B-Instruct | zero-shot CoT | 24.63 | 10.76 | 19.88 | 49.38 | 12.15 |
| Qwen2-7B-Instruct | 1-shot Random | 23.40 | 10.98 | 21.04 | 47.30 | 12.55 |
| Qwen2-7B-Instruct | 3-shot Random | 22.97 | 10.90 | 20.77 | 47.09 | 12.40 |
| Qwen2-7B-Instruct | 5-shot Random | 21.96 | 10.95 | 20.68 | 46.18 | 12.33 |
| Qwen2-7B-Instruct | 1-shot BM25 | 24.63 | 12.60 | 23.60 | 49.77 | 14.19 |
| Qwen2-7B-Instruct | 3-shot BM25 | 24.48 | 13.19 | 25.26 | 50.97 | 15.03 |
| Qwen2-7B-Instruct | 5-shot BM25 | 24.32 | 13.68 | 26.46 | 51.65 | 15.55 |
| Qwen2-7B-Instruct | 1-shot PubMedBERT-NeuML | 23.49 | 11.23 | 21.78 | 48.32 | 12.83 |
| Qwen2-7B-Instruct | 3-shot PubMedBERT-NeuML | 22.76 | 11.68 | 23.08 | 50.52 | 13.30 |
| Qwen2-7B-Instruct | 5-shot PubMedBERT-NeuML | 21.85 | 12.10 | 24.15 | 51.54 | 13.60 |
| Qwen2-7B-Instruct | 1-shot BioMedBERT-ft | 24.04 | 11.59 | 22.30 | 48.76 | 13.12 |
| Qwen2-7B-Instruct | 3-shot BioMedBERT-ft | 23.61 | 12.29 | 24.30 | 50.89 | 13.94 |
| Qwen2-7B-Instruct | 5-shot BioMedBERT-ft | 23.47 | 13.04 | 25.87 | 51.66 | 14.73 |
| Qwen2-7B-Instruct | 1-shot Gold heuristic | 26.64 | 14.33 | 27.66 | 52.47 | 15.21 |
| Qwen2-7B-Instruct | 3-shot Gold heuristic | 28.76 | 16.92 | 35.96 | 57.31 | 18.16 |
| Qwen2-7B-Instruct | 5-shot Gold heuristic | 29.59 | 18.51 | 40.96 | 59.79 | 20.00 |
| Qwen2-72B | zero-shot | 24.63 | 11.06 | 21.77 | 46.17 | 12.48 |
| Qwen2-72B | zero-shot CoT | 22.18 | 10.42 | 20.45 | 44.05 | 11.49 |
| Qwen2-72B | 1-shot Random | 17.42 | 10.14 | 17.86 | 37.58 | 11.08 |
| Qwen2-72B | 3-shot Random | 16.95 | 11.45 | 19.59 | 38.72 | 11.77 |
| Qwen2-72B | 5-shot Random | 16.15 | 11.50 | 19.66 | 37.75 | 11.61 |
| Qwen2-72B | 1-shot BM25 | 22.44 | 13.63 | 23.82 | 46.68 | 15.42 |
| Qwen2-72B | 3-shot BM25 | 22.35 | 13.71 | 26.74 | 49.33 | 15.90 |
| Qwen2-72B | 5-shot BM25 | 22.57 | 14.35 | 27.74 | 49.44 | 16.46 |
| Qwen2-72B | 1-shot PubMedBERT-NeuML | 18.05 | 10.54 | 19.16 | 44.57 | 11.80 |
| Qwen2-72B | 3-shot PubMedBERT-NeuML | 17.95 | 11.17 | 22.32 | 49.01 | 12.53 |
| Qwen2-72B | 5-shot PubMedBERT-NeuML | 17.82 | 11.64 | 23.70 | 49.47 | 12.99 |
| Qwen2-72B | 1-shot BioMedBERT-ft | 19.34 | 11.83 | 21.19 | 45.39 | 12.99 |
| Qwen2-72B | 3-shot BioMedBERT-ft | 20.76 | 13.11 | 24.92 | 49.22 | 14.61 |
| Qwen2-72B | 5-shot BioMedBERT-ft | 21.07 | 13.39 | 26.77 | 50.30 | 15.05 |
| Qwen2-72B | 1-shot Gold heuristic | 26.98 | 18.99 | 35.49 | 54.39 | 19.58 |
| Qwen2-72B | 3-shot Gold heuristic | 28.90 | 22.43 | 43.07 | 58.80 | 23.25 |
| Qwen2-72B | 5-shot Gold heuristic | 29.83 | 22.68 | 46.40 | 60.37 | 23.99 |
| Qwen2-72B-Instruct | zero-shot | 27.24 | 11.09 | 24.10 | 51.58 | 13.13 |
| Qwen2-72B-Instruct | zero-shot CoT | 25.66 | 10.91 | 23.03 | 50.24 | 12.74 |
| Qwen2-72B-Instruct | 1-shot Random | 21.53 | 11.77 | 21.44 | 42.66 | 12.70 |
| Qwen2-72B-Instruct | 3-shot Random | 20.14 | 12.25 | 22.17 | 42.86 | 12.66 |
| Qwen2-72B-Instruct | 5-shot Random | 18.90 | 12.32 | 22.16 | 41.59 | 12.48 |
| Qwen2-72B-Instruct | 1-shot BM25 | 25.12 | 13.97 | 26.72 | 49.62 | 15.56 |
| Qwen2-72B-Instruct | 3-shot BM25 | 25.48 | 15.17 | 29.83 | 52.32 | 16.92 |
| Qwen2-72B-Instruct | 5-shot BM25 | 25.29 | 15.83 | 31.26 | 53.00 | 17.58 |
| Qwen2-72B-Instruct | 1-shot PubMedBERT-NeuML | 22.38 | 11.90 | 23.33 | 46.86 | 13.32 |
| Qwen2-72B-Instruct | 3-shot PubMedBERT-NeuML | 22.20 | 13.35 | 26.42 | 50.97 | 14.46 |
| Qwen2-72B-Instruct | 5-shot PubMedBERT-NeuML | 21.87 | 13.87 | 28.19 | 52.63 | 14.90 |
| Qwen2-72B-Instruct | 1-shot BioMedBERT-ft | 23.05 | 12.59 | 23.89 | 46.59 | 13.78 |
| Qwen2-72B-Instruct | 3-shot BioMedBERT-ft | 23.52 | 13.89 | 28.15 | 51.24 | 15.34 |
| Qwen2-72B-Instruct | 5-shot BioMedBERT-ft | 24.25 | 14.63 | 30.49 | 53.04 | 16.30 |
| Qwen2-72B-Instruct | 1-shot Gold heuristic | 27.04 | 15.80 | 31.16 | 51.36 | 16.68 |
| Qwen2-72B-Instruct | 3-shot Gold heuristic | 30.42 | 19.50 | 42.26 | 60.04 | 20.71 |
| Qwen2-72B-Instruct | 5-shot Gold heuristic | 32.02 | 20.78 | 46.89 | 62.62 | 22.43 |

## ICU

| Model | Setting | Rec. | Prec. | MAP@20 | MD Acc. | F1 |
|:---|:---|---:|---:|---:|---:|---:|
| BioMedBERT-110M | threshold-tuned | 15.56 | 23.13 | 29.85 | 55.91 | 14.09 |
| BioMedBERT-110M | untuned | 19.60 | 17.99 | 48.25 | 72.83 | 17.11 |
| GatorTronS-345M | threshold-tuned | 20.00 | 29.62 | 33.10 | 63.66 | 18.89 |
| GatorTronS-345M | untuned | 23.71 | 23.54 | 52.54 | 77.74 | 21.49 |
| M2-BERT-80M | threshold-tuned | 8.13 | 8.98 | 13.68 | 35.05 | 4.60 |
| M2-BERT-80M | untuned | 10.16 | 8.27 | 33.65 | 56.05 | 8.14 |
| Llama 3.1-8B | zero-shot | 13.97 | 14.81 | 12.69 | 33.43 | 11.20 |
| Llama 3.1-8B | zero-shot CoT | 12.25 | 13.41 | 11.24 | 35.57 | 9.76 |
| Llama 3.1-8B | 1-shot Random | 11.69 | 10.65 | 12.66 | 33.08 | 9.79 |
| Llama 3.1-8B | 3-shot Random | 11.23 | 12.11 | 13.39 | 33.03 | 9.99 |
| Llama 3.1-8B | 5-shot Random | 10.82 | 12.20 | 13.90 | 33.24 | 9.83 |
| Llama 3.1-8B | 1-shot BM25 | 15.13 | 13.14 | 17.46 | 46.05 | 12.93 |
| Llama 3.1-8B | 3-shot BM25 | 14.56 | 14.36 | 19.92 | 47.14 | 13.30 |
| Llama 3.1-8B | 5-shot BM25 | 14.12 | 14.60 | 21.16 | 48.33 | 13.20 |
| Llama 3.1-8B | 1-shot PubMedBERT-NeuML | 11.78 | 10.73 | 15.10 | 43.75 | 10.17 |
| Llama 3.1-8B | 3-shot PubMedBERT-NeuML | 11.14 | 11.78 | 17.69 | 46.20 | 10.43 |
| Llama 3.1-8B | 5-shot PubMedBERT-NeuML | 11.69 | 12.73 | 19.23 | 48.01 | 11.08 |
| Llama 3.1-8B | 1-shot BioMedBERT-ft | 13.52 | 12.37 | 16.49 | 44.64 | 11.61 |
| Llama 3.1-8B | 3-shot BioMedBERT-ft | 12.67 | 13.48 | 18.86 | 46.37 | 11.68 |
| Llama 3.1-8B | 5-shot BioMedBERT-ft | 12.49 | 13.69 | 20.38 | 48.43 | 11.80 |
| Llama 3.1-8B | 1-shot Gold heuristic | 20.05 | 21.41 | 31.01 | 55.44 | 18.64 |
| Llama 3.1-8B | 3-shot Gold heuristic | 19.15 | 22.36 | 35.94 | 58.28 | 19.05 |
| Llama 3.1-8B | 5-shot Gold heuristic | 19.47 | 23.41 | 38.19 | 60.42 | 19.69 |
| Llama 3.1-8B-Instruct | zero-shot | 19.59 | 13.93 | 16.49 | 47.34 | 13.03 |
| Llama 3.1-8B-Instruct | zero-shot CoT | 17.88 | 13.00 | 13.71 | 47.65 | 11.49 |
| Llama 3.1-8B-Instruct | 1-shot Random | 9.84 | 10.01 | 9.09 | 25.73 | 8.63 |
| Llama 3.1-8B-Instruct | 3-shot Random | 12.93 | 12.02 | 12.68 | 32.52 | 10.49 |
| Llama 3.1-8B-Instruct | 5-shot Random | 13.94 | 12.59 | 14.12 | 35.41 | 11.17 |
| Llama 3.1-8B-Instruct | 1-shot BM25 | 12.58 | 12.45 | 13.47 | 35.28 | 11.33 |
| Llama 3.1-8B-Instruct | 3-shot BM25 | 14.61 | 13.82 | 17.35 | 40.40 | 12.63 |
| Llama 3.1-8B-Instruct | 5-shot BM25 | 15.52 | 14.35 | 19.75 | 44.27 | 13.33 |
| Llama 3.1-8B-Instruct | 1-shot PubMedBERT-NeuML | 9.42 | 11.11 | 10.58 | 30.58 | 8.96 |
| Llama 3.1-8B-Instruct | 3-shot PubMedBERT-NeuML | 11.04 | 12.38 | 14.03 | 36.25 | 10.35 |
| Llama 3.1-8B-Instruct | 5-shot PubMedBERT-NeuML | 12.32 | 12.69 | 16.63 | 41.37 | 11.20 |
| Llama 3.1-8B-Instruct | 1-shot BioMedBERT-ft | 10.20 | 12.36 | 10.89 | 28.98 | 9.84 |
| Llama 3.1-8B-Instruct | 3-shot BioMedBERT-ft | 11.46 | 13.29 | 13.88 | 33.41 | 10.95 |
| Llama 3.1-8B-Instruct | 5-shot BioMedBERT-ft | 12.56 | 13.51 | 16.66 | 37.99 | 11.68 |
| Llama 3.1-8B-Instruct | 1-shot Gold heuristic | 13.67 | 19.37 | 18.01 | 34.14 | 14.01 |
| Llama 3.1-8B-Instruct | 3-shot Gold heuristic | 16.15 | 20.09 | 25.94 | 43.14 | 15.90 |
| Llama 3.1-8B-Instruct | 5-shot Gold heuristic | 18.38 | 20.92 | 31.70 | 50.76 | 17.58 |
| Llama 3.1-70B | zero-shot | 15.40 | 12.44 | 13.22 | 33.60 | 10.48 |
| Llama 3.1-70B | zero-shot CoT | 18.73 | 11.87 | 15.73 | 40.81 | 11.22 |
| Llama 3.1-70B | 1-shot Random | 7.46 | 6.05 | 8.79 | 23.41 | 6.11 |
| Llama 3.1-70B | 3-shot Random | 7.78 | 6.70 | 10.53 | 26.68 | 6.48 |
| Llama 3.1-70B | 5-shot Random | 8.43 | 7.89 | 12.27 | 30.97 | 7.38 |
| Llama 3.1-70B | 1-shot BM25 | 12.07 | 10.70 | 15.81 | 42.56 | 10.69 |
| Llama 3.1-70B | 3-shot BM25 | 13.10 | 11.02 | 18.79 | 49.65 | 11.30 |
| Llama 3.1-70B | 5-shot BM25 | 13.74 | 12.11 | 20.08 | 50.85 | 12.13 |
| Llama 3.1-70B | 1-shot PubMedBERT-NeuML | 8.02 | 8.54 | 13.10 | 40.50 | 7.54 |
| Llama 3.1-70B | 3-shot PubMedBERT-NeuML | 8.91 | 8.57 | 16.01 | 48.18 | 8.07 |
| Llama 3.1-70B | 5-shot PubMedBERT-NeuML | 9.68 | 9.59 | 17.64 | 50.06 | 8.91 |
| Llama 3.1-70B | 1-shot BioMedBERT-ft | 10.00 | 10.33 | 14.63 | 40.96 | 9.19 |
| Llama 3.1-70B | 3-shot BioMedBERT-ft | 11.47 | 10.57 | 17.86 | 49.63 | 9.91 |
| Llama 3.1-70B | 5-shot BioMedBERT-ft | 11.94 | 11.24 | 19.69 | 51.28 | 10.55 |
| Llama 3.1-70B | 1-shot Gold heuristic | 17.12 | 20.86 | 31.16 | 53.45 | 17.39 |
| Llama 3.1-70B | 3-shot Gold heuristic | 18.92 | 20.59 | 37.35 | 61.04 | 18.58 |
| Llama 3.1-70B | 5-shot Gold heuristic | 19.76 | 21.00 | 40.09 | 62.37 | 19.18 |
| Llama 3.1-70B-Instruct | zero-shot | 20.31 | 13.36 | 14.95 | 50.37 | 12.55 |
| Llama 3.1-70B-Instruct | zero-shot CoT | 21.24 | 12.66 | 18.00 | 50.18 | 12.08 |
| Llama 3.1-70B-Instruct | 1-shot Random | 14.85 | 14.22 | 14.06 | 30.84 | 11.98 |
| Llama 3.1-70B-Instruct | 3-shot Random | 18.94 | 14.54 | 19.13 | 41.78 | 13.90 |
| Llama 3.1-70B-Instruct | 5-shot Random | 15.97 | 14.34 | 17.12 | 37.63 | 12.73 |
| Llama 3.1-70B-Instruct | 1-shot BM25 | 16.46 | 15.16 | 16.10 | 34.13 | 13.43 |
| Llama 3.1-70B-Instruct | 3-shot BM25 | 19.03 | 15.26 | 22.07 | 47.33 | 14.96 |
| Llama 3.1-70B-Instruct | 5-shot BM25 | 17.06 | 15.38 | 21.36 | 44.90 | 14.47 |
| Llama 3.1-70B-Instruct | 1-shot PubMedBERT-NeuML | 13.68 | 14.28 | 13.63 | 29.55 | 11.80 |
| Llama 3.1-70B-Instruct | 3-shot PubMedBERT-NeuML | 16.65 | 13.99 | 20.34 | 46.01 | 13.37 |
| Llama 3.1-70B-Instruct | 5-shot PubMedBERT-NeuML | 14.84 | 14.06 | 19.87 | 44.36 | 12.76 |
| Llama 3.1-70B-Instruct | 1-shot BioMedBERT-ft | 13.10 | 14.43 | 12.90 | 27.11 | 11.64 |
| Llama 3.1-70B-Instruct | 3-shot BioMedBERT-ft | 16.73 | 14.71 | 20.17 | 43.16 | 13.83 |
| Llama 3.1-70B-Instruct | 5-shot BioMedBERT-ft | 15.43 | 14.86 | 20.13 | 42.42 | 13.53 |
| Llama 3.1-70B-Instruct | 1-shot Gold heuristic | 14.56 | 17.93 | 15.78 | 29.21 | 13.40 |
| Llama 3.1-70B-Instruct | 3-shot Gold heuristic | 21.09 | 19.73 | 31.34 | 52.15 | 17.87 |
| Llama 3.1-70B-Instruct | 5-shot Gold heuristic | 19.91 | 20.03 | 33.22 | 52.28 | 18.03 |
| Mistral-7B | zero-shot | 8.84 | 13.49 | 7.68 | 25.66 | 8.07 |
| Mistral-7B | zero-shot CoT | 0.34 | 2.63 | 0.72 | 1.38 | 0.13 |
| Mistral-7B | 1-shot Random | 4.19 | 4.10 | 5.90 | 15.64 | 3.95 |
| Mistral-7B | 3-shot Random | 6.78 | 8.05 | 9.31 | 23.57 | 6.63 |
| Mistral-7B | 5-shot Random | 5.99 | 7.77 | 9.15 | 22.50 | 6.06 |
| Mistral-7B | 1-shot BM25 | 10.74 | 10.60 | 14.48 | 39.60 | 10.11 |
| Mistral-7B | 3-shot BM25 | 11.36 | 11.90 | 16.55 | 42.97 | 10.87 |
| Mistral-7B | 5-shot BM25 | 11.08 | 11.91 | 17.21 | 43.16 | 10.78 |
| Mistral-7B | 1-shot PubMedBERT-NeuML | 7.16 | 8.16 | 12.31 | 37.82 | 6.96 |
| Mistral-7B | 3-shot PubMedBERT-NeuML | 8.35 | 9.86 | 14.69 | 42.53 | 8.28 |
| Mistral-7B | 5-shot PubMedBERT-NeuML | 8.40 | 9.84 | 15.78 | 43.58 | 8.37 |
| Mistral-7B | 1-shot BioMedBERT-ft | 8.90 | 10.00 | 13.70 | 38.22 | 8.51 |
| Mistral-7B | 3-shot BioMedBERT-ft | 9.85 | 11.38 | 16.01 | 43.04 | 9.54 |
| Mistral-7B | 5-shot BioMedBERT-ft | 9.75 | 11.44 | 17.27 | 44.74 | 9.65 |
| Mistral-7B | 1-shot Gold heuristic | 16.10 | 22.19 | 31.13 | 50.90 | 17.55 |
| Mistral-7B | 3-shot Gold heuristic | 17.65 | 21.68 | 35.15 | 55.77 | 18.18 |
| Mistral-7B | 5-shot Gold heuristic | 17.72 | 21.74 | 36.75 | 57.13 | 18.36 |
| Mistral-7B-Instruct | zero-shot | 23.65 | 13.67 | 18.36 | 51.37 | 13.69 |
| Mistral-7B-Instruct | zero-shot CoT | 21.59 | 12.12 | 16.28 | 46.10 | 12.30 |
| Mistral-7B-Instruct | 1-shot Random | 22.67 | 12.91 | 20.14 | 44.62 | 13.32 |
| Mistral-7B-Instruct | 3-shot Random | 21.23 | 12.95 | 19.79 | 41.57 | 13.12 |
| Mistral-7B-Instruct | 5-shot Random | 20.31 | 13.09 | 19.46 | 39.99 | 13.01 |
| Mistral-7B-Instruct | 1-shot BM25 | 23.58 | 13.89 | 21.87 | 48.11 | 14.30 |
| Mistral-7B-Instruct | 3-shot BM25 | 22.10 | 14.77 | 23.22 | 47.75 | 14.79 |
| Mistral-7B-Instruct | 5-shot BM25 | 20.96 | 15.08 | 24.09 | 47.72 | 14.95 |
| Mistral-7B-Instruct | 1-shot PubMedBERT-NeuML | 22.73 | 13.23 | 21.27 | 47.23 | 13.74 |
| Mistral-7B-Instruct | 3-shot PubMedBERT-NeuML | 21.31 | 13.74 | 22.17 | 47.17 | 13.77 |
| Mistral-7B-Instruct | 5-shot PubMedBERT-NeuML | 20.17 | 14.18 | 23.33 | 47.87 | 13.99 |
| Mistral-7B-Instruct | 1-shot BioMedBERT-ft | 23.40 | 13.42 | 21.36 | 47.00 | 13.92 |
| Mistral-7B-Instruct | 3-shot BioMedBERT-ft | 22.03 | 14.22 | 22.86 | 47.50 | 14.36 |
| Mistral-7B-Instruct | 5-shot BioMedBERT-ft | 20.54 | 14.75 | 24.18 | 48.06 | 14.53 |
| Mistral-7B-Instruct | 1-shot Gold heuristic | 24.45 | 15.47 | 24.96 | 50.32 | 15.31 |
| Mistral-7B-Instruct | 3-shot Gold heuristic | 24.74 | 18.21 | 32.28 | 53.39 | 17.48 |
| Mistral-7B-Instruct | 5-shot Gold heuristic | 24.75 | 19.58 | 36.27 | 55.11 | 18.69 |
| Mistral-Nemo-13B | zero-shot | 13.49 | 14.31 | 12.25 | 31.63 | 10.86 |
| Mistral-Nemo-13B | zero-shot CoT | 11.11 | 11.63 | 9.67 | 37.51 | 8.23 |
| Mistral-Nemo-13B | 1-shot Random | 7.00 | 7.45 | 6.62 | 21.43 | 6.37 |
| Mistral-Nemo-13B | 3-shot Random | 6.81 | 8.30 | 9.14 | 27.04 | 6.41 |
| Mistral-Nemo-13B | 5-shot Random | 6.16 | 8.21 | 9.53 | 27.53 | 5.96 |
| Mistral-Nemo-13B | 1-shot BM25 | 11.98 | 11.32 | 14.09 | 39.70 | 10.75 |
| Mistral-Nemo-13B | 3-shot BM25 | 12.07 | 12.05 | 17.05 | 44.53 | 11.06 |
| Mistral-Nemo-13B | 5-shot BM25 | 11.31 | 12.73 | 17.66 | 44.95 | 11.01 |
| Mistral-Nemo-13B | 1-shot PubMedBERT-NeuML | 8.26 | 9.92 | 10.99 | 34.41 | 8.04 |
| Mistral-Nemo-13B | 3-shot PubMedBERT-NeuML | 8.62 | 9.60 | 14.93 | 42.94 | 8.16 |
| Mistral-Nemo-13B | 5-shot PubMedBERT-NeuML | 8.61 | 10.48 | 16.14 | 44.68 | 8.47 |
| Mistral-Nemo-13B | 1-shot BioMedBERT-ft | 9.65 | 10.93 | 12.33 | 35.33 | 9.05 |
| Mistral-Nemo-13B | 3-shot BioMedBERT-ft | 9.65 | 10.93 | 15.91 | 42.67 | 9.14 |
| Mistral-Nemo-13B | 5-shot BioMedBERT-ft | 10.09 | 11.54 | 17.32 | 44.98 | 9.58 |
| Mistral-Nemo-13B | 1-shot Gold heuristic | 15.03 | 20.73 | 24.97 | 44.03 | 15.71 |
| Mistral-Nemo-13B | 3-shot Gold heuristic | 17.02 | 21.25 | 33.33 | 54.59 | 17.47 |
| Mistral-Nemo-13B | 5-shot Gold heuristic | 17.73 | 21.96 | 36.04 | 56.91 | 18.12 |
| Mistral-Nemo-13B-Instruct | zero-shot | 19.64 | 11.97 | 15.95 | 48.39 | 11.58 |
| Mistral-Nemo-13B-Instruct | zero-shot CoT | 19.03 | 11.96 | 14.03 | 48.54 | 10.78 |
| Mistral-Nemo-13B-Instruct | 1-shot Random | 11.65 | 9.49 | 11.88 | 31.12 | 9.17 |
| Mistral-Nemo-13B-Instruct | 3-shot Random | 9.60 | 10.76 | 11.83 | 29.86 | 8.86 |
| Mistral-Nemo-13B-Instruct | 5-shot Random | 7.95 | 10.80 | 11.21 | 27.59 | 7.94 |
| Mistral-Nemo-13B-Instruct | 1-shot BM25 | 14.58 | 12.40 | 16.49 | 43.53 | 12.22 |
| Mistral-Nemo-13B-Instruct | 3-shot BM25 | 11.90 | 12.55 | 17.21 | 43.55 | 11.32 |
| Mistral-Nemo-13B-Instruct | 5-shot BM25 | 11.24 | 13.35 | 17.82 | 43.28 | 11.37 |
| Mistral-Nemo-13B-Instruct | 1-shot PubMedBERT-NeuML | 11.42 | 10.13 | 14.56 | 41.77 | 9.57 |
| Mistral-Nemo-13B-Instruct | 3-shot PubMedBERT-NeuML | 9.61 | 10.77 | 15.59 | 42.50 | 9.25 |
| Mistral-Nemo-13B-Instruct | 5-shot PubMedBERT-NeuML | 8.87 | 11.08 | 16.58 | 43.45 | 9.10 |
| Mistral-Nemo-13B-Instruct | 1-shot BioMedBERT-ft | 13.04 | 13.18 | 12.82 | 32.62 | 10.96 |
| Mistral-Nemo-13B-Instruct | 3-shot BioMedBERT-ft | 20.14 | 14.87 | 22.76 | 53.77 | 14.71 |
| Mistral-Nemo-13B-Instruct | 5-shot BioMedBERT-ft | 19.64 | 15.57 | 24.06 | 55.45 | 14.94 |
| Mistral-Nemo-13B-Instruct | 1-shot Gold heuristic | 18.89 | 19.22 | 27.82 | 52.17 | 16.90 |
| Mistral-Nemo-13B-Instruct | 3-shot Gold heuristic | 16.28 | 21.25 | 33.10 | 52.65 | 16.93 |
| Mistral-Nemo-13B-Instruct | 5-shot Gold heuristic | 15.54 | 22.13 | 35.83 | 53.29 | 16.98 |
| Qwen2-7B | zero-shot | 20.95 | 13.92 | 17.78 | 40.97 | 13.11 |
| Qwen2-7B | zero-shot CoT | 19.05 | 13.96 | 16.68 | 40.19 | 12.55 |
| Qwen2-7B | 1-shot Random | 11.63 | 9.48 | 11.89 | 30.99 | 9.16 |
| Qwen2-7B | 3-shot Random | 9.62 | 10.79 | 11.83 | 29.86 | 8.87 |
| Qwen2-7B | 5-shot Random | 7.92 | 10.78 | 11.20 | 27.52 | 7.90 |
| Qwen2-7B | 1-shot BM25 | 14.50 | 12.30 | 16.48 | 43.55 | 12.13 |
| Qwen2-7B | 3-shot BM25 | 11.93 | 12.54 | 17.21 | 43.50 | 11.33 |
| Qwen2-7B | 5-shot BM25 | 11.23 | 13.35 | 17.80 | 43.35 | 11.35 |
| Qwen2-7B | 1-shot PubMedBERT-NeuML | 11.41 | 10.14 | 14.56 | 41.72 | 9.56 |
| Qwen2-7B | 3-shot PubMedBERT-NeuML | 9.65 | 10.80 | 15.60 | 42.50 | 9.28 |
| Qwen2-7B | 5-shot PubMedBERT-NeuML | 8.83 | 11.11 | 16.59 | 43.49 | 9.10 |
| Qwen2-7B | 1-shot BioMedBERT-ft | 13.45 | 12.05 | 15.96 | 43.33 | 11.26 |
| Qwen2-7B | 3-shot BioMedBERT-ft | 11.05 | 12.86 | 17.06 | 43.43 | 10.71 |
| Qwen2-7B | 5-shot BioMedBERT-ft | 10.25 | 12.69 | 18.31 | 43.98 | 10.41 |
| Qwen2-7B | 1-shot Gold heuristic | 18.90 | 19.21 | 27.87 | 52.23 | 16.91 |
| Qwen2-7B | 3-shot Gold heuristic | 16.30 | 21.23 | 33.10 | 52.54 | 16.97 |
| Qwen2-7B | 5-shot Gold heuristic | 15.49 | 22.03 | 35.78 | 53.47 | 16.93 |
| Qwen2-7B-Instruct | zero-shot | 23.06 | 14.14 | 20.11 | 45.86 | 14.10 |
| Qwen2-7B-Instruct | zero-shot CoT | 22.13 | 13.89 | 18.59 | 46.30 | 13.64 |
| Qwen2-7B-Instruct | 1-shot Random | 20.22 | 13.33 | 18.27 | 44.70 | 13.30 |
| Qwen2-7B-Instruct | 3-shot Random | 18.85 | 13.14 | 17.89 | 44.80 | 12.87 |
| Qwen2-7B-Instruct | 5-shot Random | 17.86 | 13.04 | 17.46 | 43.81 | 12.67 |
| Qwen2-7B-Instruct | 1-shot BM25 | 20.47 | 14.04 | 20.02 | 47.76 | 13.94 |
| Qwen2-7B-Instruct | 3-shot BM25 | 18.77 | 14.19 | 20.98 | 50.07 | 13.95 |
| Qwen2-7B-Instruct | 5-shot BM25 | 18.23 | 14.40 | 21.98 | 50.52 | 14.15 |
| Qwen2-7B-Instruct | 1-shot PubMedBERT-NeuML | 20.34 | 13.80 | 19.34 | 47.24 | 13.63 |
| Qwen2-7B-Instruct | 3-shot PubMedBERT-NeuML | 18.11 | 13.00 | 20.24 | 50.52 | 13.08 |
| Qwen2-7B-Instruct | 5-shot PubMedBERT-NeuML | 16.69 | 13.25 | 21.00 | 51.38 | 12.96 |
| Qwen2-7B-Instruct | 1-shot BioMedBERT-ft | 20.27 | 13.69 | 19.54 | 47.23 | 13.59 |
| Qwen2-7B-Instruct | 3-shot BioMedBERT-ft | 18.87 | 13.81 | 20.80 | 50.02 | 13.64 |
| Qwen2-7B-Instruct | 5-shot BioMedBERT-ft | 18.33 | 14.01 | 22.10 | 51.87 | 13.88 |
| Qwen2-7B-Instruct | 1-shot Gold heuristic | 22.25 | 16.26 | 23.64 | 50.99 | 15.32 |
| Qwen2-7B-Instruct | 3-shot Gold heuristic | 22.19 | 17.80 | 30.62 | 56.69 | 16.86 |
| Qwen2-7B-Instruct | 5-shot Gold heuristic | 21.37 | 18.90 | 34.63 | 58.92 | 17.55 |
| Qwen2-72B | zero-shot | 22.55 | 14.06 | 18.46 | 43.28 | 13.70 |
| Qwen2-72B | zero-shot CoT | 20.32 | 13.14 | 17.80 | 40.93 | 12.56 |
| Qwen2-72B | 1-shot Random | 15.24 | 11.66 | 15.12 | 37.11 | 11.32 |
| Qwen2-72B | 3-shot Random | 13.64 | 12.05 | 15.73 | 37.47 | 10.90 |
| Qwen2-72B | 5-shot Random | 12.60 | 11.98 | 15.82 | 36.24 | 10.52 |
| Qwen2-72B | 1-shot BM25 | 17.36 | 13.15 | 19.14 | 46.55 | 13.37 |
| Qwen2-72B | 3-shot BM25 | 16.21 | 13.61 | 21.01 | 49.26 | 13.49 |
| Qwen2-72B | 5-shot BM25 | 15.27 | 13.79 | 21.96 | 49.40 | 13.32 |
| Qwen2-72B | 1-shot PubMedBERT-NeuML | 14.68 | 11.46 | 17.08 | 44.63 | 11.35 |
| Qwen2-72B | 3-shot PubMedBERT-NeuML | 13.24 | 11.30 | 19.02 | 48.00 | 11.07 |
| Qwen2-72B | 5-shot PubMedBERT-NeuML | 13.18 | 11.84 | 19.99 | 49.01 | 11.28 |
| Qwen2-72B | 1-shot BioMedBERT-ft | 16.79 | 13.31 | 18.14 | 44.90 | 12.89 |
| Qwen2-72B | 3-shot BioMedBERT-ft | 14.82 | 12.83 | 20.34 | 48.69 | 12.43 |
| Qwen2-72B | 5-shot BioMedBERT-ft | 15.08 | 13.50 | 21.52 | 49.70 | 12.95 |
| Qwen2-72B | 1-shot Gold heuristic | 21.79 | 19.03 | 29.50 | 54.07 | 17.55 |
| Qwen2-72B | 3-shot Gold heuristic | 20.67 | 20.41 | 35.77 | 59.32 | 18.71 |
| Qwen2-72B | 5-shot Gold heuristic | 20.30 | 20.76 | 38.41 | 59.97 | 18.97 |
| Qwen2-72B-Instruct | zero-shot | 24.11 | 14.20 | 21.29 | 48.11 | 14.42 |
| Qwen2-72B-Instruct | zero-shot CoT | 22.78 | 14.10 | 21.00 | 47.16 | 14.12 |
| Qwen2-72B-Instruct | 1-shot Random | 17.67 | 13.95 | 17.74 | 40.80 | 12.81 |
| Qwen2-72B-Instruct | 3-shot Random | 15.90 | 13.27 | 18.05 | 41.22 | 12.04 |
| Qwen2-72B-Instruct | 5-shot Random | 14.24 | 13.45 | 17.84 | 40.66 | 11.53 |
| Qwen2-72B-Instruct | 1-shot BM25 | 19.97 | 15.29 | 21.41 | 47.27 | 14.62 |
| Qwen2-72B-Instruct | 3-shot BM25 | 18.89 | 15.89 | 23.90 | 51.18 | 15.08 |
| Qwen2-72B-Instruct | 5-shot BM25 | 17.42 | 15.73 | 24.99 | 52.06 | 14.77 |
| Qwen2-72B-Instruct | 1-shot PubMedBERT-NeuML | 18.25 | 13.83 | 19.76 | 45.59 | 13.17 |
| Qwen2-72B-Instruct | 3-shot PubMedBERT-NeuML | 17.13 | 13.97 | 22.50 | 50.50 | 13.38 |
| Qwen2-72B-Instruct | 5-shot PubMedBERT-NeuML | 16.10 | 14.48 | 23.79 | 52.34 | 13.46 |
| Qwen2-72B-Instruct | 1-shot BioMedBERT-ft | 18.97 | 14.60 | 19.96 | 44.80 | 13.73 |
| Qwen2-72B-Instruct | 3-shot BioMedBERT-ft | 18.04 | 14.86 | 23.50 | 50.92 | 14.13 |
| Qwen2-72B-Instruct | 5-shot BioMedBERT-ft | 18.12 | 15.44 | 25.30 | 52.59 | 14.81 |
| Qwen2-72B-Instruct | 1-shot Gold heuristic | 21.32 | 17.84 | 25.05 | 48.46 | 16.00 |
| Qwen2-72B-Instruct | 3-shot Gold heuristic | 22.72 | 20.24 | 35.75 | 59.15 | 18.63 |
| Qwen2-72B-Instruct | 5-shot Gold heuristic | 22.83 | 21.21 | 40.06 | 61.66 | 19.70 |
