# CliniBench: all runs (unified evaluation)

Micro- and macro-averaged over the shared label space, top-20 predictions, unweighted mean of ICD-9 and ICD-10. Produced by `src/evaluation/evaluate_runs.py tables`; per-partition values in `unified_metrics.csv`.

## HOSP

| Model | Setting | Micro Rec. | Micro Prec. | Micro F1 | Macro Rec. | Macro Prec. | Macro F1 | MAP@20 | MD Acc. |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| BioMedBERT-110M | threshold-tuned | 44.06 | 23.85 | 30.88 | 22.91 | 29.55 | 20.71 | 32.83 | 59.16 |
| BioMedBERT-110M | untuned | 65.23 | 35.45 | 45.83 | 30.74 | 19.91 | 22.00 | 54.99 | 78.21 |
| GatorTronS-345M | threshold-tuned | 47.85 | 25.93 | 33.55 | 28.61 | 33.44 | 25.34 | 37.23 | 65.22 |
| GatorTronS-345M | untuned | 68.35 | 37.16 | 48.03 | 35.36 | 23.12 | 25.77 | 58.84 | 81.64 |
| M2-BERT-80M | threshold-tuned | 34.63 | 18.81 | 24.32 | 11.72 | 12.05 | 8.34 | 24.87 | 40.47 |
| M2-BERT-80M | untuned | 55.15 | 29.93 | 38.72 | 14.90 | 8.51 | 9.49 | 41.67 | 60.83 |
| Llama 3.1-8B | zero-shot | 19.34 | 25.88 | 22.10 | 16.61 | 12.16 | 11.06 | 14.69 | 35.22 |
| Llama 3.1-8B | zero-shot CoT | 16.57 | 22.71 | 19.12 | 14.85 | 10.89 | 9.68 | 13.06 | 37.96 |
| Llama 3.1-8B | 1-shot Random | 21.67 | 25.22 | 23.23 | 15.41 | 10.18 | 10.76 | 15.25 | 33.98 |
| Llama 3.1-8B | 3-shot Random | 22.89 | 29.27 | 25.55 | 15.85 | 12.19 | 11.72 | 17.61 | 35.09 |
| Llama 3.1-8B | 5-shot Random | 23.55 | 29.94 | 26.24 | 15.46 | 12.43 | 11.75 | 18.22 | 35.00 |
| Llama 3.1-8B | 1-shot BM25 | 33.34 | 31.76 | 32.42 | 20.88 | 14.17 | 15.76 | 22.04 | 45.06 |
| Llama 3.1-8B | 3-shot BM25 | 34.75 | 33.65 | 34.11 | 21.47 | 15.48 | 16.80 | 25.75 | 46.59 |
| Llama 3.1-8B | 5-shot BM25 | 35.74 | 34.43 | 35.01 | 21.76 | 15.99 | 17.22 | 27.15 | 47.23 |
| Llama 3.1-8B | 1-shot PubMedBERT-NeuML | 26.40 | 24.36 | 25.25 | 15.53 | 10.51 | 11.37 | 16.75 | 42.61 |
| Llama 3.1-8B | 3-shot PubMedBERT-NeuML | 27.90 | 27.71 | 27.72 | 16.12 | 12.21 | 12.81 | 21.09 | 46.00 |
| Llama 3.1-8B | 5-shot PubMedBERT-NeuML | 29.45 | 29.35 | 29.34 | 17.12 | 13.23 | 13.74 | 22.99 | 48.26 |
| Llama 3.1-8B | 1-shot BioMedBERT-ft | 28.39 | 29.68 | 28.94 | 17.56 | 12.83 | 13.47 | 19.33 | 43.63 |
| Llama 3.1-8B | 3-shot BioMedBERT-ft | 30.60 | 32.38 | 31.39 | 18.62 | 14.06 | 14.71 | 23.63 | 46.56 |
| Llama 3.1-8B | 5-shot BioMedBERT-ft | 32.48 | 33.57 | 32.95 | 19.22 | 14.85 | 15.43 | 25.77 | 48.25 |
| Llama 3.1-8B | 1-shot Gold heuristic | 42.71 | 47.81 | 45.00 | 27.08 | 22.85 | 22.24 | 36.91 | 54.58 |
| Llama 3.1-8B | 3-shot Gold heuristic | 47.71 | 52.34 | 49.83 | 28.72 | 25.52 | 24.93 | 44.22 | 58.25 |
| Llama 3.1-8B | 5-shot Gold heuristic | 49.98 | 52.77 | 51.25 | 29.86 | 26.08 | 25.88 | 47.19 | 60.76 |
| Llama 3.1-8B-Instruct | zero-shot | 26.38 | 23.30 | 24.71 | 22.31 | 10.88 | 11.61 | 17.89 | 47.53 |
| Llama 3.1-8B-Instruct | zero-shot CoT | 23.56 | 18.67 | 20.80 | 20.84 | 9.93 | 10.11 | 14.82 | 49.61 |
| Llama 3.1-8B-Instruct | 1-shot Random | 18.47 | 18.96 | 18.58 | 13.20 | 9.40 | 9.38 | 10.95 | 27.44 |
| Llama 3.1-8B-Instruct | 3-shot Random | 24.20 | 23.84 | 23.90 | 17.22 | 11.08 | 11.44 | 16.37 | 35.16 |
| Llama 3.1-8B-Instruct | 5-shot Random | 26.22 | 25.10 | 25.55 | 18.80 | 11.55 | 12.12 | 17.96 | 38.01 |
| Llama 3.1-8B-Instruct | 1-shot BM25 | 27.62 | 26.03 | 26.61 | 17.27 | 12.39 | 12.81 | 16.47 | 36.05 |
| Llama 3.1-8B-Instruct | 3-shot BM25 | 31.77 | 28.82 | 30.08 | 19.34 | 13.72 | 14.45 | 20.98 | 40.35 |
| Llama 3.1-8B-Instruct | 5-shot BM25 | 35.01 | 29.95 | 32.18 | 20.99 | 14.24 | 15.44 | 23.92 | 44.34 |
| Llama 3.1-8B-Instruct | 1-shot PubMedBERT-NeuML | 21.25 | 20.80 | 20.86 | 13.17 | 9.94 | 9.91 | 12.12 | 31.32 |
| Llama 3.1-8B-Instruct | 3-shot PubMedBERT-NeuML | 25.77 | 24.04 | 24.73 | 15.29 | 11.14 | 11.38 | 16.76 | 37.80 |
| Llama 3.1-8B-Instruct | 5-shot PubMedBERT-NeuML | 29.10 | 25.56 | 27.11 | 17.00 | 11.92 | 12.53 | 19.87 | 42.88 |
| Llama 3.1-8B-Instruct | 1-shot BioMedBERT-ft | 21.52 | 22.69 | 21.94 | 13.53 | 11.10 | 10.60 | 12.72 | 29.57 |
| Llama 3.1-8B-Instruct | 3-shot BioMedBERT-ft | 26.93 | 26.67 | 26.67 | 15.60 | 12.68 | 12.43 | 17.61 | 35.82 |
| Llama 3.1-8B-Instruct | 5-shot BioMedBERT-ft | 30.41 | 28.27 | 29.19 | 17.54 | 13.21 | 13.54 | 20.92 | 40.37 |
| Llama 3.1-8B-Instruct | 1-shot Gold heuristic | 28.48 | 31.38 | 29.67 | 17.62 | 17.84 | 14.96 | 20.96 | 34.29 |
| Llama 3.1-8B-Instruct | 3-shot Gold heuristic | 39.04 | 38.78 | 38.72 | 21.89 | 19.72 | 18.23 | 31.93 | 44.90 |
| Llama 3.1-8B-Instruct | 5-shot Gold heuristic | 44.94 | 41.85 | 43.21 | 24.67 | 20.99 | 20.19 | 38.66 | 52.16 |
| Llama 3.1-70B | zero-shot | 20.64 | 18.44 | 19.44 | 17.29 | 9.78 | 9.66 | 14.99 | 35.21 |
| Llama 3.1-70B | zero-shot CoT | 25.99 | 19.44 | 22.20 | 20.72 | 9.42 | 10.19 | 18.01 | 43.63 |
| Llama 3.1-70B | 1-shot Random | 17.23 | 17.68 | 17.37 | 9.38 | 6.22 | 6.83 | 10.27 | 23.15 |
| Llama 3.1-70B | 3-shot Random | 19.30 | 19.83 | 19.45 | 10.23 | 6.79 | 7.48 | 13.81 | 27.14 |
| Llama 3.1-70B | 5-shot Random | 21.29 | 22.65 | 21.73 | 11.95 | 8.27 | 8.88 | 16.25 | 31.01 |
| Llama 3.1-70B | 1-shot BM25 | 31.28 | 30.08 | 30.55 | 17.58 | 13.10 | 14.24 | 19.83 | 40.27 |
| Llama 3.1-70B | 3-shot BM25 | 35.42 | 28.07 | 31.22 | 20.09 | 12.31 | 14.70 | 24.45 | 47.78 |
| Llama 3.1-70B | 5-shot BM25 | 36.28 | 29.06 | 32.14 | 20.55 | 12.67 | 15.09 | 26.18 | 49.05 |
| Llama 3.1-70B | 1-shot PubMedBERT-NeuML | 22.55 | 21.68 | 22.03 | 10.66 | 8.80 | 8.88 | 13.93 | 37.78 |
| Llama 3.1-70B | 3-shot PubMedBERT-NeuML | 27.77 | 21.61 | 24.23 | 13.04 | 8.74 | 9.91 | 18.82 | 47.83 |
| Llama 3.1-70B | 5-shot PubMedBERT-NeuML | 29.29 | 23.42 | 25.93 | 14.39 | 10.05 | 11.10 | 21.39 | 49.67 |
| Llama 3.1-70B | 1-shot BioMedBERT-ft | 25.45 | 28.19 | 26.66 | 13.46 | 11.45 | 11.44 | 16.77 | 38.41 |
| Llama 3.1-70B | 3-shot BioMedBERT-ft | 31.90 | 27.82 | 29.65 | 17.10 | 11.77 | 13.00 | 22.38 | 47.74 |
| Llama 3.1-70B | 5-shot BioMedBERT-ft | 33.93 | 29.00 | 31.16 | 18.74 | 12.42 | 13.96 | 25.28 | 50.54 |
| Llama 3.1-70B | 1-shot Gold heuristic | 43.05 | 51.51 | 46.72 | 24.23 | 23.34 | 21.79 | 36.97 | 51.01 |
| Llama 3.1-70B | 3-shot Gold heuristic | 50.73 | 51.69 | 51.11 | 28.34 | 24.69 | 25.06 | 46.19 | 58.57 |
| Llama 3.1-70B | 5-shot Gold heuristic | 53.55 | 50.45 | 51.85 | 30.54 | 24.55 | 25.83 | 50.09 | 61.80 |
| Llama 3.1-70B-Instruct | zero-shot | 26.20 | 19.07 | 22.04 | 23.04 | 10.40 | 11.33 | 15.15 | 52.20 |
| Llama 3.1-70B-Instruct | zero-shot CoT | 28.50 | 18.58 | 22.46 | 23.70 | 10.00 | 10.59 | 18.07 | 51.39 |
| Llama 3.1-70B-Instruct | 1-shot Random | 24.60 | 25.23 | 24.74 | 17.37 | 11.63 | 11.63 | 16.09 | 32.20 |
| Llama 3.1-70B-Instruct | 3-shot Random | 31.42 | 26.35 | 28.46 | 21.99 | 12.30 | 13.37 | 22.20 | 42.99 |
| Llama 3.1-70B-Instruct | 5-shot Random | 29.42 | 26.66 | 27.67 | 20.13 | 12.51 | 13.15 | 21.18 | 40.32 |
| Llama 3.1-70B-Instruct | 1-shot BM25 | 29.42 | 29.36 | 29.20 | 18.57 | 13.56 | 13.49 | 18.81 | 35.09 |
| Llama 3.1-70B-Instruct | 3-shot BM25 | 37.02 | 30.42 | 33.21 | 22.78 | 14.00 | 15.63 | 26.01 | 46.80 |
| Llama 3.1-70B-Instruct | 5-shot BM25 | 36.09 | 30.38 | 32.76 | 21.89 | 14.43 | 15.81 | 26.05 | 45.49 |
| Llama 3.1-70B-Instruct | 1-shot PubMedBERT-NeuML | 24.38 | 25.94 | 24.96 | 15.38 | 11.45 | 11.15 | 15.34 | 30.83 |
| Llama 3.1-70B-Instruct | 3-shot PubMedBERT-NeuML | 32.28 | 26.76 | 29.08 | 19.76 | 11.96 | 13.21 | 22.72 | 44.88 |
| Llama 3.1-70B-Instruct | 5-shot PubMedBERT-NeuML | 31.30 | 26.93 | 28.71 | 18.43 | 12.38 | 13.24 | 22.90 | 44.29 |
| Llama 3.1-70B-Instruct | 1-shot BioMedBERT-ft | 24.01 | 27.12 | 25.32 | 14.82 | 11.73 | 11.03 | 14.82 | 29.31 |
| Llama 3.1-70B-Instruct | 3-shot BioMedBERT-ft | 32.96 | 29.23 | 30.81 | 19.72 | 12.85 | 13.88 | 22.87 | 42.86 |
| Llama 3.1-70B-Instruct | 5-shot BioMedBERT-ft | 32.91 | 29.57 | 30.95 | 18.97 | 13.25 | 14.15 | 23.85 | 43.02 |
| Llama 3.1-70B-Instruct | 1-shot Gold heuristic | 28.44 | 33.08 | 30.36 | 16.98 | 15.66 | 13.41 | 19.43 | 32.14 |
| Llama 3.1-70B-Instruct | 3-shot Gold heuristic | 44.62 | 39.54 | 41.72 | 25.89 | 18.21 | 18.83 | 36.12 | 51.28 |
| Llama 3.1-70B-Instruct | 5-shot Gold heuristic | 46.65 | 41.43 | 43.60 | 26.42 | 19.36 | 20.08 | 40.11 | 53.82 |
| Mistral-7B | zero-shot | 11.81 | 23.16 | 15.60 | 10.45 | 12.02 | 8.50 | 9.67 | 27.25 |
| Mistral-7B | zero-shot CoT | 0.23 | 0.94 | 0.37 | 0.40 | 3.86 | 0.16 | 0.20 | 0.69 |
| Mistral-7B | 1-shot Random | 12.82 | 14.36 | 13.51 | 5.67 | 4.26 | 4.58 | 5.83 | 14.89 |
| Mistral-7B | 3-shot Random | 17.13 | 22.25 | 19.30 | 10.53 | 8.31 | 8.39 | 12.61 | 26.34 |
| Mistral-7B | 5-shot Random | 18.17 | 22.83 | 20.18 | 10.78 | 8.85 | 8.75 | 13.34 | 27.11 |
| Mistral-7B | 1-shot BM25 | 29.57 | 29.84 | 29.64 | 16.28 | 13.02 | 13.89 | 17.94 | 36.83 |
| Mistral-7B | 3-shot BM25 | 32.30 | 29.30 | 30.63 | 18.56 | 13.24 | 14.83 | 22.30 | 43.34 |
| Mistral-7B | 5-shot BM25 | 33.34 | 29.41 | 31.14 | 19.15 | 13.78 | 15.29 | 23.64 | 44.13 |
| Mistral-7B | 1-shot PubMedBERT-NeuML | 21.33 | 21.40 | 21.31 | 9.45 | 8.88 | 8.40 | 12.65 | 35.40 |
| Mistral-7B | 3-shot PubMedBERT-NeuML | 25.48 | 23.31 | 24.23 | 12.76 | 10.39 | 10.60 | 17.74 | 43.52 |
| Mistral-7B | 5-shot PubMedBERT-NeuML | 27.25 | 24.25 | 25.54 | 13.65 | 10.94 | 11.38 | 19.65 | 45.23 |
| Mistral-7B | 1-shot BioMedBERT-ft | 24.14 | 28.85 | 26.25 | 11.84 | 11.19 | 10.78 | 15.29 | 35.55 |
| Mistral-7B | 3-shot BioMedBERT-ft | 28.84 | 29.18 | 28.91 | 15.55 | 12.40 | 12.92 | 20.52 | 43.83 |
| Mistral-7B | 5-shot BioMedBERT-ft | 30.91 | 29.54 | 30.08 | 16.60 | 12.73 | 13.53 | 22.80 | 46.10 |
| Mistral-7B | 1-shot Gold heuristic | 43.68 | 57.24 | 49.46 | 23.56 | 26.68 | 23.44 | 37.54 | 48.83 |
| Mistral-7B | 3-shot Gold heuristic | 48.92 | 53.34 | 50.91 | 27.31 | 26.27 | 25.27 | 44.48 | 56.52 |
| Mistral-7B | 5-shot Gold heuristic | 51.44 | 51.56 | 51.35 | 28.68 | 25.36 | 25.46 | 47.51 | 58.75 |
| Mistral-7B-Instruct | zero-shot | 30.91 | 21.14 | 25.06 | 26.45 | 10.52 | 12.16 | 19.91 | 54.06 |
| Mistral-7B-Instruct | zero-shot CoT | 28.37 | 18.55 | 22.39 | 23.96 | 9.46 | 10.85 | 17.30 | 49.76 |
| Mistral-7B-Instruct | 1-shot Random | 34.14 | 22.78 | 27.27 | 24.83 | 10.47 | 12.35 | 23.37 | 46.39 |
| Mistral-7B-Instruct | 3-shot Random | 33.15 | 23.45 | 27.40 | 23.52 | 10.84 | 12.30 | 22.97 | 42.97 |
| Mistral-7B-Instruct | 5-shot Random | 32.71 | 23.81 | 27.50 | 22.20 | 10.96 | 12.23 | 22.66 | 41.20 |
| Mistral-7B-Instruct | 1-shot BM25 | 37.73 | 25.29 | 30.22 | 27.36 | 12.36 | 14.39 | 26.61 | 50.67 |
| Mistral-7B-Instruct | 3-shot BM25 | 38.42 | 27.26 | 31.82 | 25.73 | 13.61 | 15.31 | 28.05 | 49.12 |
| Mistral-7B-Instruct | 5-shot BM25 | 39.37 | 28.33 | 32.88 | 25.33 | 14.29 | 15.93 | 29.07 | 48.89 |
| Mistral-7B-Instruct | 1-shot PubMedBERT-NeuML | 34.82 | 23.21 | 27.79 | 25.50 | 10.91 | 12.86 | 24.35 | 48.18 |
| Mistral-7B-Instruct | 3-shot PubMedBERT-NeuML | 34.82 | 24.47 | 28.67 | 23.98 | 11.85 | 13.32 | 25.54 | 47.49 |
| Mistral-7B-Instruct | 5-shot PubMedBERT-NeuML | 35.45 | 25.49 | 29.59 | 22.76 | 12.55 | 13.84 | 26.50 | 47.85 |
| Mistral-7B-Instruct | 1-shot BioMedBERT-ft | 35.10 | 23.56 | 28.13 | 25.90 | 11.18 | 13.10 | 24.70 | 48.87 |
| Mistral-7B-Instruct | 3-shot BioMedBERT-ft | 36.05 | 25.58 | 29.85 | 25.07 | 12.42 | 14.12 | 26.75 | 48.43 |
| Mistral-7B-Instruct | 5-shot BioMedBERT-ft | 37.15 | 27.04 | 31.23 | 24.45 | 13.29 | 14.84 | 28.23 | 48.72 |
| Mistral-7B-Instruct | 1-shot Gold heuristic | 39.31 | 26.50 | 31.59 | 28.38 | 13.44 | 15.02 | 29.96 | 52.42 |
| Mistral-7B-Instruct | 3-shot Gold heuristic | 45.77 | 32.79 | 38.11 | 29.93 | 16.60 | 18.06 | 38.61 | 55.83 |
| Mistral-7B-Instruct | 5-shot Gold heuristic | 49.25 | 36.32 | 41.71 | 30.71 | 18.38 | 19.76 | 43.00 | 57.35 |
| Mistral-Nemo-13B | zero-shot | 18.57 | 25.49 | 21.46 | 14.87 | 11.30 | 10.16 | 14.10 | 32.83 |
| Mistral-Nemo-13B | zero-shot CoT | 13.47 | 18.57 | 15.59 | 13.66 | 9.37 | 8.00 | 11.29 | 38.32 |
| Mistral-Nemo-13B | 1-shot Random | 14.24 | 15.69 | 14.80 | 9.33 | 7.17 | 7.18 | 7.58 | 22.06 |
| Mistral-Nemo-13B | 3-shot Random | 15.36 | 19.95 | 17.29 | 10.15 | 8.48 | 8.02 | 12.01 | 27.88 |
| Mistral-Nemo-13B | 5-shot Random | 16.76 | 22.49 | 19.16 | 10.68 | 9.14 | 8.42 | 13.28 | 30.18 |
| Mistral-Nemo-13B | 1-shot BM25 | 29.69 | 27.73 | 28.52 | 18.13 | 12.90 | 13.93 | 17.86 | 39.15 |
| Mistral-Nemo-13B | 3-shot BM25 | 32.34 | 28.80 | 30.37 | 19.01 | 13.08 | 14.69 | 22.21 | 45.09 |
| Mistral-Nemo-13B | 5-shot BM25 | 32.41 | 29.95 | 31.06 | 19.18 | 13.77 | 15.10 | 23.60 | 45.97 |
| Mistral-Nemo-13B | 1-shot PubMedBERT-NeuML | 20.88 | 19.99 | 20.29 | 11.63 | 9.51 | 9.33 | 12.20 | 34.17 |
| Mistral-Nemo-13B | 3-shot PubMedBERT-NeuML | 24.25 | 22.03 | 23.01 | 12.68 | 9.70 | 10.11 | 17.05 | 43.58 |
| Mistral-Nemo-13B | 5-shot PubMedBERT-NeuML | 25.53 | 23.84 | 24.60 | 13.59 | 10.41 | 10.86 | 19.20 | 46.52 |
| Mistral-Nemo-13B | 1-shot BioMedBERT-ft | 22.37 | 23.99 | 23.02 | 13.03 | 11.50 | 10.86 | 14.01 | 34.31 |
| Mistral-Nemo-13B | 3-shot BioMedBERT-ft | 27.01 | 26.72 | 26.78 | 15.24 | 11.92 | 12.26 | 19.24 | 42.75 |
| Mistral-Nemo-13B | 5-shot BioMedBERT-ft | 29.39 | 28.71 | 28.96 | 16.80 | 12.55 | 13.15 | 22.22 | 46.69 |
| Mistral-Nemo-13B | 1-shot Gold heuristic | 35.74 | 41.38 | 38.14 | 21.43 | 23.19 | 19.91 | 30.59 | 44.12 |
| Mistral-Nemo-13B | 3-shot Gold heuristic | 45.66 | 49.07 | 47.20 | 26.45 | 25.02 | 23.88 | 41.54 | 55.04 |
| Mistral-Nemo-13B | 5-shot Gold heuristic | 49.50 | 50.35 | 49.83 | 28.46 | 24.88 | 24.77 | 46.24 | 59.18 |
| Mistral-Nemo-13B-Instruct | zero-shot | 25.28 | 17.78 | 20.84 | 22.23 | 9.55 | 10.37 | 16.47 | 50.35 |
| Mistral-Nemo-13B-Instruct | zero-shot CoT | 23.10 | 15.73 | 18.68 | 21.19 | 9.22 | 9.57 | 14.61 | 51.61 |
| Mistral-Nemo-13B-Instruct | 1-shot Random | 21.04 | 21.92 | 21.43 | 14.55 | 8.97 | 9.78 | 13.92 | 33.07 |
| Mistral-Nemo-13B-Instruct | 3-shot Random | 18.68 | 26.24 | 21.82 | 13.02 | 10.70 | 10.20 | 14.57 | 32.39 |
| Mistral-Nemo-13B-Instruct | 5-shot Random | 17.99 | 26.81 | 21.53 | 11.76 | 11.12 | 10.02 | 14.39 | 30.61 |
| Mistral-Nemo-13B-Instruct | 1-shot BM25 | 31.65 | 29.99 | 30.73 | 19.84 | 13.32 | 14.70 | 20.63 | 43.33 |
| Mistral-Nemo-13B-Instruct | 3-shot BM25 | 30.07 | 31.26 | 30.62 | 18.17 | 14.45 | 15.21 | 22.44 | 43.96 |
| Mistral-Nemo-13B-Instruct | 5-shot BM25 | 34.08 | 31.20 | 32.52 | 21.15 | 15.27 | 16.05 | 26.02 | 48.41 |
| Mistral-Nemo-13B-Instruct | 1-shot PubMedBERT-NeuML | 25.13 | 23.05 | 23.99 | 14.90 | 9.70 | 10.63 | 15.96 | 41.27 |
| Mistral-Nemo-13B-Instruct | 3-shot PubMedBERT-NeuML | 24.44 | 25.20 | 24.78 | 13.43 | 10.82 | 11.11 | 18.33 | 43.90 |
| Mistral-Nemo-13B-Instruct | 5-shot PubMedBERT-NeuML | 25.16 | 26.18 | 25.64 | 13.14 | 11.99 | 11.71 | 19.64 | 44.32 |
| Mistral-Nemo-13B-Instruct | 1-shot BioMedBERT-ft | 22.34 | 19.94 | 20.98 | 16.89 | 11.22 | 11.21 | 14.75 | 34.49 |
| Mistral-Nemo-13B-Instruct | 3-shot BioMedBERT-ft | 36.71 | 27.60 | 31.45 | 25.57 | 12.99 | 14.95 | 26.84 | 53.94 |
| Mistral-Nemo-13B-Instruct | 5-shot BioMedBERT-ft | 37.35 | 29.21 | 32.75 | 25.34 | 14.05 | 15.78 | 28.64 | 54.83 |
| Mistral-Nemo-13B-Instruct | 1-shot Gold heuristic | 40.39 | 41.92 | 41.07 | 25.19 | 19.94 | 19.64 | 33.60 | 52.36 |
| Mistral-Nemo-13B-Instruct | 3-shot Gold heuristic | 44.13 | 53.16 | 48.16 | 24.81 | 25.07 | 23.10 | 41.29 | 54.18 |
| Mistral-Nemo-13B-Instruct | 5-shot Gold heuristic | 46.42 | 55.73 | 50.58 | 25.03 | 27.00 | 24.32 | 44.74 | 55.01 |
| Qwen2-7B | zero-shot | 30.07 | 23.14 | 26.09 | 22.86 | 11.32 | 12.26 | 20.54 | 43.37 |
| Qwen2-7B | zero-shot CoT | 26.54 | 24.50 | 25.42 | 21.04 | 11.62 | 12.04 | 18.80 | 42.01 |
| Qwen2-7B | 1-shot Random | 21.04 | 21.92 | 21.43 | 14.56 | 9.00 | 9.79 | 13.89 | 32.97 |
| Qwen2-7B | 3-shot Random | 18.66 | 26.24 | 21.80 | 13.03 | 10.70 | 10.20 | 14.57 | 32.36 |
| Qwen2-7B | 5-shot Random | 17.97 | 26.81 | 21.52 | 11.75 | 11.14 | 10.03 | 14.39 | 30.56 |
| Qwen2-7B | 1-shot BM25 | 31.64 | 29.99 | 30.73 | 19.83 | 13.31 | 14.69 | 20.62 | 43.35 |
| Qwen2-7B | 3-shot BM25 | 30.07 | 31.26 | 30.62 | 18.15 | 14.43 | 15.19 | 22.43 | 43.93 |
| Qwen2-7B | 5-shot BM25 | 30.44 | 31.64 | 31.01 | 17.71 | 14.94 | 15.31 | 23.15 | 43.40 |
| Qwen2-7B | 1-shot PubMedBERT-NeuML | 25.14 | 23.05 | 23.99 | 14.89 | 9.72 | 10.63 | 15.96 | 41.23 |
| Qwen2-7B | 3-shot PubMedBERT-NeuML | 24.44 | 25.20 | 24.78 | 13.41 | 10.81 | 11.10 | 18.33 | 43.88 |
| Qwen2-7B | 5-shot PubMedBERT-NeuML | 25.15 | 26.19 | 25.64 | 13.12 | 11.99 | 11.70 | 19.64 | 44.32 |
| Qwen2-7B | 1-shot BioMedBERT-ft | 27.26 | 27.17 | 27.16 | 17.31 | 11.74 | 12.44 | 18.44 | 43.04 |
| Qwen2-7B | 3-shot BioMedBERT-ft | 27.17 | 30.81 | 28.85 | 15.96 | 13.24 | 13.42 | 20.94 | 43.72 |
| Qwen2-7B | 5-shot BioMedBERT-ft | 28.50 | 32.04 | 30.14 | 16.02 | 14.03 | 13.96 | 22.84 | 44.44 |
| Qwen2-7B | 1-shot Gold heuristic | 40.38 | 41.99 | 41.10 | 25.26 | 20.01 | 19.71 | 33.58 | 52.26 |
| Qwen2-7B | 3-shot Gold heuristic | 44.13 | 53.11 | 48.14 | 24.77 | 24.98 | 23.08 | 41.24 | 54.11 |
| Qwen2-7B | 5-shot Gold heuristic | 46.47 | 55.78 | 50.64 | 25.14 | 27.19 | 24.41 | 44.73 | 54.98 |
| Qwen2-7B-Instruct | zero-shot | 34.53 | 22.61 | 27.28 | 25.80 | 11.27 | 12.90 | 22.30 | 48.95 |
| Qwen2-7B-Instruct | zero-shot CoT | 32.21 | 21.61 | 25.82 | 24.63 | 10.76 | 12.15 | 19.88 | 49.38 |
| Qwen2-7B-Instruct | 1-shot Random | 32.38 | 23.11 | 26.91 | 23.40 | 10.98 | 12.55 | 21.04 | 47.30 |
| Qwen2-7B-Instruct | 3-shot Random | 31.76 | 23.69 | 27.07 | 22.97 | 10.90 | 12.40 | 20.77 | 47.09 |
| Qwen2-7B-Instruct | 5-shot Random | 31.24 | 24.09 | 27.13 | 21.96 | 10.95 | 12.33 | 20.68 | 46.18 |
| Qwen2-7B-Instruct | 1-shot BM25 | 35.39 | 25.35 | 29.47 | 24.63 | 12.60 | 14.19 | 23.60 | 49.77 |
| Qwen2-7B-Instruct | 3-shot BM25 | 36.35 | 27.37 | 31.16 | 24.48 | 13.19 | 15.03 | 25.26 | 50.97 |
| Qwen2-7B-Instruct | 5-shot BM25 | 37.11 | 28.30 | 32.04 | 24.32 | 13.68 | 15.55 | 26.46 | 51.65 |
| Qwen2-7B-Instruct | 1-shot PubMedBERT-NeuML | 32.95 | 23.42 | 27.32 | 23.49 | 11.23 | 12.83 | 21.78 | 48.32 |
| Qwen2-7B-Instruct | 3-shot PubMedBERT-NeuML | 33.23 | 24.86 | 28.38 | 22.76 | 11.68 | 13.30 | 23.08 | 50.52 |
| Qwen2-7B-Instruct | 5-shot PubMedBERT-NeuML | 33.65 | 25.84 | 29.17 | 21.85 | 12.10 | 13.60 | 24.15 | 51.54 |
| Qwen2-7B-Instruct | 1-shot BioMedBERT-ft | 33.43 | 23.75 | 27.71 | 24.04 | 11.59 | 13.12 | 22.30 | 48.76 |
| Qwen2-7B-Instruct | 3-shot BioMedBERT-ft | 34.42 | 25.97 | 29.53 | 23.61 | 12.29 | 13.94 | 24.30 | 50.89 |
| Qwen2-7B-Instruct | 5-shot BioMedBERT-ft | 35.45 | 27.57 | 30.95 | 23.47 | 13.04 | 14.73 | 25.87 | 51.66 |
| Qwen2-7B-Instruct | 1-shot Gold heuristic | 38.19 | 27.14 | 31.66 | 26.64 | 14.33 | 15.21 | 27.66 | 52.47 |
| Qwen2-7B-Instruct | 3-shot Gold heuristic | 44.30 | 33.73 | 38.21 | 28.76 | 16.92 | 18.16 | 35.96 | 57.31 |
| Qwen2-7B-Instruct | 5-shot Gold heuristic | 47.89 | 38.16 | 42.38 | 29.59 | 18.51 | 20.00 | 40.96 | 59.79 |
| Qwen2-72B | zero-shot | 33.11 | 22.81 | 26.96 | 24.63 | 11.06 | 12.48 | 21.77 | 46.17 |
| Qwen2-72B | zero-shot CoT | 30.00 | 22.09 | 25.40 | 22.18 | 10.42 | 11.49 | 20.45 | 44.05 |
| Qwen2-72B | 1-shot Random | 26.35 | 23.82 | 24.93 | 17.42 | 10.14 | 11.08 | 17.86 | 37.58 |
| Qwen2-72B | 3-shot Random | 26.69 | 27.19 | 26.79 | 16.95 | 11.45 | 11.77 | 19.59 | 38.72 |
| Qwen2-72B | 5-shot Random | 26.35 | 27.87 | 26.94 | 16.15 | 11.50 | 11.61 | 19.66 | 37.75 |
| Qwen2-72B | 1-shot BM25 | 35.42 | 29.69 | 32.20 | 22.44 | 13.63 | 15.42 | 23.82 | 46.68 |
| Qwen2-72B | 3-shot BM25 | 36.79 | 30.52 | 33.26 | 22.35 | 13.71 | 15.90 | 26.74 | 49.33 |
| Qwen2-72B | 5-shot BM25 | 37.10 | 31.26 | 33.84 | 22.57 | 14.35 | 16.46 | 27.74 | 49.44 |
| Qwen2-72B | 1-shot PubMedBERT-NeuML | 29.16 | 23.94 | 26.20 | 18.05 | 10.54 | 11.80 | 19.16 | 44.57 |
| Qwen2-72B | 3-shot PubMedBERT-NeuML | 30.60 | 25.27 | 27.59 | 17.95 | 11.17 | 12.53 | 22.32 | 49.01 |
| Qwen2-72B | 5-shot PubMedBERT-NeuML | 31.36 | 26.26 | 28.50 | 17.82 | 11.64 | 12.99 | 23.70 | 49.47 |
| Qwen2-72B | 1-shot BioMedBERT-ft | 30.99 | 26.95 | 28.73 | 19.34 | 11.83 | 12.99 | 21.19 | 45.39 |
| Qwen2-72B | 3-shot BioMedBERT-ft | 33.62 | 29.49 | 31.32 | 20.76 | 13.11 | 14.61 | 24.92 | 49.22 |
| Qwen2-72B | 5-shot BioMedBERT-ft | 35.04 | 30.41 | 32.47 | 21.07 | 13.39 | 15.05 | 26.77 | 50.30 |
| Qwen2-72B | 1-shot Gold heuristic | 43.06 | 38.42 | 40.49 | 26.98 | 18.99 | 19.58 | 35.49 | 54.39 |
| Qwen2-72B | 3-shot Gold heuristic | 48.54 | 46.58 | 47.43 | 28.90 | 22.43 | 23.25 | 43.07 | 58.80 |
| Qwen2-72B | 5-shot Gold heuristic | 51.18 | 47.85 | 49.36 | 29.83 | 22.68 | 23.99 | 46.40 | 60.37 |
| Qwen2-72B-Instruct | zero-shot | 36.74 | 23.39 | 28.53 | 27.24 | 11.09 | 13.13 | 24.10 | 51.58 |
| Qwen2-72B-Instruct | zero-shot CoT | 35.16 | 22.58 | 27.45 | 25.66 | 10.91 | 12.74 | 23.03 | 50.24 |
| Qwen2-72B-Instruct | 1-shot Random | 30.27 | 24.73 | 27.10 | 21.53 | 11.77 | 12.70 | 21.44 | 42.66 |
| Qwen2-72B-Instruct | 3-shot Random | 30.68 | 26.28 | 28.12 | 20.14 | 12.25 | 12.66 | 22.17 | 42.86 |
| Qwen2-72B-Instruct | 5-shot Random | 30.19 | 26.95 | 28.30 | 18.90 | 12.32 | 12.48 | 22.16 | 41.59 |
| Qwen2-72B-Instruct | 1-shot BM25 | 37.15 | 28.27 | 31.97 | 25.12 | 13.97 | 15.56 | 26.72 | 49.62 |
| Qwen2-72B-Instruct | 3-shot BM25 | 39.46 | 30.91 | 34.50 | 25.48 | 15.17 | 16.92 | 29.83 | 52.32 |
| Qwen2-72B-Instruct | 5-shot BM25 | 40.62 | 31.99 | 35.64 | 25.29 | 15.83 | 17.58 | 31.26 | 53.00 |
| Qwen2-72B-Instruct | 1-shot PubMedBERT-NeuML | 32.67 | 25.07 | 28.24 | 22.38 | 11.90 | 13.32 | 23.33 | 46.86 |
| Qwen2-72B-Instruct | 3-shot PubMedBERT-NeuML | 34.57 | 27.50 | 30.48 | 22.20 | 13.35 | 14.46 | 26.42 | 50.97 |
| Qwen2-72B-Instruct | 5-shot PubMedBERT-NeuML | 35.87 | 28.71 | 31.75 | 21.87 | 13.87 | 14.90 | 28.19 | 52.63 |
| Qwen2-72B-Instruct | 1-shot BioMedBERT-ft | 33.10 | 25.74 | 28.84 | 23.05 | 12.59 | 13.78 | 23.89 | 46.59 |
| Qwen2-72B-Instruct | 3-shot BioMedBERT-ft | 36.58 | 29.16 | 32.29 | 23.52 | 13.89 | 15.34 | 28.15 | 51.24 |
| Qwen2-72B-Instruct | 5-shot BioMedBERT-ft | 38.60 | 30.76 | 34.08 | 24.25 | 14.63 | 16.30 | 30.49 | 53.04 |
| Qwen2-72B-Instruct | 1-shot Gold heuristic | 39.39 | 30.34 | 34.16 | 27.04 | 15.80 | 16.68 | 31.16 | 51.36 |
| Qwen2-72B-Instruct | 3-shot Gold heuristic | 48.33 | 38.12 | 42.42 | 30.42 | 19.50 | 20.71 | 42.26 | 60.04 |
| Qwen2-72B-Instruct | 5-shot Gold heuristic | 52.11 | 41.28 | 45.88 | 32.02 | 20.78 | 22.43 | 46.89 | 62.62 |

## ICU

| Model | Setting | Micro Rec. | Micro Prec. | Micro F1 | Macro Rec. | Macro Prec. | Macro F1 | MAP@20 | MD Acc. |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| BioMedBERT-110M | threshold-tuned | 36.20 | 27.97 | 31.52 | 15.56 | 23.13 | 14.09 | 29.85 | 55.91 |
| BioMedBERT-110M | untuned | 54.19 | 42.12 | 47.33 | 19.60 | 17.99 | 17.11 | 48.25 | 72.83 |
| GatorTronS-345M | threshold-tuned | 39.64 | 30.74 | 34.58 | 20.00 | 29.62 | 18.89 | 33.10 | 63.66 |
| GatorTronS-345M | untuned | 57.33 | 44.54 | 50.06 | 23.71 | 23.54 | 21.49 | 52.54 | 77.74 |
| M2-BERT-80M | threshold-tuned | 21.16 | 16.03 | 18.22 | 8.13 | 8.98 | 4.60 | 13.68 | 35.05 |
| M2-BERT-80M | untuned | 44.17 | 34.27 | 38.54 | 10.16 | 8.27 | 8.14 | 33.65 | 56.05 |
| Llama 3.1-8B | zero-shot | 16.08 | 29.84 | 20.87 | 13.97 | 14.81 | 11.20 | 12.69 | 33.43 |
| Llama 3.1-8B | zero-shot CoT | 14.16 | 27.49 | 18.68 | 12.25 | 13.41 | 9.76 | 11.24 | 35.57 |
| Llama 3.1-8B | 1-shot Random | 17.94 | 26.67 | 21.39 | 11.69 | 10.65 | 9.79 | 12.66 | 33.08 |
| Llama 3.1-8B | 3-shot Random | 17.41 | 30.17 | 22.00 | 11.23 | 12.11 | 9.99 | 13.39 | 33.03 |
| Llama 3.1-8B | 5-shot Random | 17.92 | 30.55 | 22.54 | 10.82 | 12.20 | 9.83 | 13.90 | 33.24 |
| Llama 3.1-8B | 1-shot BM25 | 26.31 | 30.26 | 28.08 | 15.13 | 13.14 | 12.93 | 17.46 | 46.05 |
| Llama 3.1-8B | 3-shot BM25 | 25.80 | 33.03 | 28.93 | 14.56 | 14.36 | 13.30 | 19.92 | 47.14 |
| Llama 3.1-8B | 5-shot BM25 | 26.58 | 34.25 | 29.91 | 14.12 | 14.60 | 13.20 | 21.16 | 48.33 |
| Llama 3.1-8B | 1-shot PubMedBERT-NeuML | 23.53 | 26.57 | 24.86 | 11.78 | 10.73 | 10.17 | 15.10 | 43.75 |
| Llama 3.1-8B | 3-shot PubMedBERT-NeuML | 23.26 | 29.67 | 26.03 | 11.14 | 11.78 | 10.43 | 17.69 | 46.20 |
| Llama 3.1-8B | 5-shot PubMedBERT-NeuML | 24.29 | 31.31 | 27.33 | 11.69 | 12.73 | 11.08 | 19.23 | 48.01 |
| Llama 3.1-8B | 1-shot BioMedBERT-ft | 24.74 | 30.29 | 27.15 | 13.52 | 12.37 | 11.61 | 16.49 | 44.64 |
| Llama 3.1-8B | 3-shot BioMedBERT-ft | 24.63 | 32.29 | 27.88 | 12.67 | 13.48 | 11.68 | 18.86 | 46.37 |
| Llama 3.1-8B | 5-shot BioMedBERT-ft | 25.55 | 33.64 | 29.00 | 12.49 | 13.69 | 11.80 | 20.38 | 48.43 |
| Llama 3.1-8B | 1-shot Gold heuristic | 36.46 | 48.61 | 41.59 | 20.05 | 21.41 | 18.64 | 31.01 | 55.44 |
| Llama 3.1-8B | 3-shot Gold heuristic | 38.83 | 51.39 | 44.16 | 19.15 | 22.36 | 19.05 | 35.94 | 58.28 |
| Llama 3.1-8B | 5-shot Gold heuristic | 40.23 | 52.65 | 45.56 | 19.47 | 23.41 | 19.69 | 38.19 | 60.42 |
| Llama 3.1-8B-Instruct | zero-shot | 23.64 | 28.49 | 25.81 | 19.59 | 13.93 | 13.03 | 16.49 | 47.34 |
| Llama 3.1-8B-Instruct | zero-shot CoT | 21.60 | 23.86 | 22.65 | 17.88 | 13.00 | 11.49 | 13.71 | 47.65 |
| Llama 3.1-8B-Instruct | 1-shot Random | 15.02 | 21.56 | 17.63 | 9.84 | 10.01 | 8.63 | 9.09 | 25.73 |
| Llama 3.1-8B-Instruct | 3-shot Random | 18.58 | 26.02 | 21.60 | 12.93 | 12.02 | 10.49 | 12.68 | 32.52 |
| Llama 3.1-8B-Instruct | 5-shot Random | 20.46 | 27.26 | 23.32 | 13.94 | 12.59 | 11.17 | 14.12 | 35.41 |
| Llama 3.1-8B-Instruct | 1-shot BM25 | 21.96 | 27.87 | 24.45 | 12.58 | 12.45 | 11.33 | 13.47 | 35.28 |
| Llama 3.1-8B-Instruct | 3-shot BM25 | 24.98 | 30.64 | 27.45 | 14.61 | 13.82 | 12.63 | 17.35 | 40.40 |
| Llama 3.1-8B-Instruct | 5-shot BM25 | 27.32 | 31.63 | 29.27 | 15.52 | 14.35 | 13.33 | 19.75 | 44.27 |
| Llama 3.1-8B-Instruct | 1-shot PubMedBERT-NeuML | 17.74 | 24.27 | 20.34 | 9.42 | 11.11 | 8.96 | 10.58 | 30.58 |
| Llama 3.1-8B-Instruct | 3-shot PubMedBERT-NeuML | 20.83 | 27.21 | 23.51 | 11.04 | 12.38 | 10.35 | 14.03 | 36.25 |
| Llama 3.1-8B-Instruct | 5-shot PubMedBERT-NeuML | 23.42 | 28.53 | 25.67 | 12.32 | 12.69 | 11.20 | 16.63 | 41.37 |
| Llama 3.1-8B-Instruct | 1-shot BioMedBERT-ft | 18.11 | 26.37 | 21.39 | 10.20 | 12.36 | 9.84 | 10.89 | 28.98 |
| Llama 3.1-8B-Instruct | 3-shot BioMedBERT-ft | 20.76 | 29.67 | 24.35 | 11.46 | 13.29 | 10.95 | 13.88 | 33.41 |
| Llama 3.1-8B-Instruct | 5-shot BioMedBERT-ft | 23.67 | 30.59 | 26.63 | 12.56 | 13.51 | 11.68 | 16.66 | 37.99 |
| Llama 3.1-8B-Instruct | 1-shot Gold heuristic | 24.24 | 37.38 | 29.24 | 13.67 | 19.37 | 14.01 | 18.01 | 34.14 |
| Llama 3.1-8B-Instruct | 3-shot Gold heuristic | 31.64 | 42.45 | 36.13 | 16.15 | 20.09 | 15.90 | 25.94 | 43.14 |
| Llama 3.1-8B-Instruct | 5-shot Gold heuristic | 36.51 | 44.97 | 40.20 | 18.38 | 20.92 | 17.58 | 31.70 | 50.76 |
| Llama 3.1-70B | zero-shot | 17.69 | 22.27 | 19.69 | 15.40 | 12.44 | 10.48 | 13.22 | 33.60 |
| Llama 3.1-70B | zero-shot CoT | 21.93 | 23.22 | 22.52 | 18.73 | 11.87 | 11.22 | 15.73 | 40.81 |
| Llama 3.1-70B | 1-shot Random | 15.87 | 19.14 | 17.31 | 7.46 | 6.05 | 6.11 | 8.79 | 23.41 |
| Llama 3.1-70B | 3-shot Random | 16.13 | 20.92 | 18.16 | 7.78 | 6.70 | 6.48 | 10.53 | 26.68 |
| Llama 3.1-70B | 5-shot Random | 17.31 | 23.66 | 19.90 | 8.43 | 7.89 | 7.38 | 12.27 | 30.97 |
| Llama 3.1-70B | 1-shot BM25 | 25.29 | 28.20 | 26.60 | 12.07 | 10.70 | 10.69 | 15.81 | 42.56 |
| Llama 3.1-70B | 3-shot BM25 | 27.25 | 28.43 | 27.78 | 13.10 | 11.02 | 11.30 | 18.79 | 49.65 |
| Llama 3.1-70B | 5-shot BM25 | 27.73 | 29.73 | 28.64 | 13.74 | 12.11 | 12.13 | 20.08 | 50.85 |
| Llama 3.1-70B | 1-shot PubMedBERT-NeuML | 22.08 | 24.21 | 23.04 | 8.02 | 8.54 | 7.54 | 13.10 | 40.50 |
| Llama 3.1-70B | 3-shot PubMedBERT-NeuML | 24.43 | 24.74 | 24.54 | 8.91 | 8.57 | 8.07 | 16.01 | 48.18 |
| Llama 3.1-70B | 5-shot PubMedBERT-NeuML | 24.94 | 26.37 | 25.58 | 9.68 | 9.59 | 8.91 | 17.64 | 50.06 |
| Llama 3.1-70B | 1-shot BioMedBERT-ft | 23.55 | 28.58 | 25.76 | 10.00 | 10.33 | 9.19 | 14.63 | 40.96 |
| Llama 3.1-70B | 3-shot BioMedBERT-ft | 26.78 | 28.35 | 27.48 | 11.47 | 10.57 | 9.91 | 17.86 | 49.63 |
| Llama 3.1-70B | 5-shot BioMedBERT-ft | 27.46 | 29.88 | 28.54 | 11.94 | 11.24 | 10.55 | 19.69 | 51.28 |
| Llama 3.1-70B | 1-shot Gold heuristic | 37.90 | 50.83 | 43.36 | 17.12 | 20.86 | 17.39 | 31.16 | 53.45 |
| Llama 3.1-70B | 3-shot Gold heuristic | 42.78 | 49.61 | 45.87 | 18.92 | 20.59 | 18.58 | 37.35 | 61.04 |
| Llama 3.1-70B | 5-shot Gold heuristic | 44.13 | 49.63 | 46.63 | 19.76 | 21.00 | 19.18 | 40.09 | 62.37 |
| Llama 3.1-70B-Instruct | zero-shot | 24.46 | 24.79 | 24.59 | 20.31 | 13.36 | 12.55 | 14.95 | 50.37 |
| Llama 3.1-70B-Instruct | zero-shot CoT | 26.68 | 24.54 | 25.53 | 21.24 | 12.66 | 12.08 | 18.00 | 50.18 |
| Llama 3.1-70B-Instruct | 1-shot Random | 20.60 | 29.46 | 24.07 | 14.85 | 14.22 | 11.98 | 14.06 | 30.84 |
| Llama 3.1-70B-Instruct | 3-shot Random | 26.34 | 30.35 | 28.06 | 18.94 | 14.54 | 13.90 | 19.13 | 41.78 |
| Llama 3.1-70B-Instruct | 5-shot Random | 23.41 | 30.14 | 26.16 | 15.97 | 14.34 | 12.73 | 17.12 | 37.63 |
| Llama 3.1-70B-Instruct | 1-shot BM25 | 24.11 | 32.06 | 27.38 | 16.46 | 15.16 | 13.43 | 16.10 | 34.13 |
| Llama 3.1-70B-Instruct | 3-shot BM25 | 29.77 | 32.64 | 31.05 | 19.03 | 15.26 | 14.96 | 22.07 | 47.33 |
| Llama 3.1-70B-Instruct | 5-shot BM25 | 28.22 | 32.50 | 30.08 | 17.06 | 15.38 | 14.47 | 21.36 | 44.90 |
| Llama 3.1-70B-Instruct | 1-shot PubMedBERT-NeuML | 20.64 | 30.15 | 24.32 | 13.68 | 14.28 | 11.80 | 13.63 | 29.55 |
| Llama 3.1-70B-Instruct | 3-shot PubMedBERT-NeuML | 27.59 | 30.68 | 28.95 | 16.65 | 13.99 | 13.37 | 20.34 | 46.01 |
| Llama 3.1-70B-Instruct | 5-shot PubMedBERT-NeuML | 26.06 | 30.76 | 28.09 | 14.84 | 14.06 | 12.76 | 19.87 | 44.36 |
| Llama 3.1-70B-Instruct | 1-shot BioMedBERT-ft | 19.99 | 31.58 | 24.33 | 13.10 | 14.43 | 11.64 | 12.90 | 27.11 |
| Llama 3.1-70B-Instruct | 3-shot BioMedBERT-ft | 27.77 | 32.48 | 29.83 | 16.73 | 14.71 | 13.83 | 20.17 | 43.16 |
| Llama 3.1-70B-Instruct | 5-shot BioMedBERT-ft | 26.73 | 32.68 | 29.24 | 15.43 | 14.86 | 13.53 | 20.13 | 42.42 |
| Llama 3.1-70B-Instruct | 1-shot Gold heuristic | 22.77 | 37.01 | 28.01 | 14.56 | 17.93 | 13.40 | 15.78 | 29.21 |
| Llama 3.1-70B-Instruct | 3-shot Gold heuristic | 37.49 | 43.48 | 40.10 | 21.09 | 19.73 | 17.87 | 31.34 | 52.15 |
| Llama 3.1-70B-Instruct | 5-shot Gold heuristic | 37.94 | 44.84 | 40.88 | 19.91 | 20.03 | 18.03 | 33.22 | 52.28 |
| Mistral-7B | zero-shot | 9.52 | 26.45 | 13.99 | 8.84 | 13.49 | 8.07 | 7.68 | 25.66 |
| Mistral-7B | zero-shot CoT | 0.86 | 5.22 | 1.48 | 0.34 | 2.63 | 0.13 | 0.72 | 1.38 |
| Mistral-7B | 1-shot Random | 12.81 | 16.50 | 14.40 | 4.19 | 4.10 | 3.95 | 5.90 | 15.64 |
| Mistral-7B | 3-shot Random | 13.14 | 23.24 | 16.78 | 6.78 | 8.05 | 6.63 | 9.31 | 23.57 |
| Mistral-7B | 5-shot Random | 13.04 | 23.39 | 16.73 | 5.99 | 7.77 | 6.06 | 9.15 | 22.50 |
| Mistral-7B | 1-shot BM25 | 23.75 | 27.76 | 25.56 | 10.74 | 10.60 | 10.11 | 14.48 | 39.60 |
| Mistral-7B | 3-shot BM25 | 23.17 | 29.20 | 25.79 | 11.36 | 11.90 | 10.87 | 16.55 | 42.97 |
| Mistral-7B | 5-shot BM25 | 23.37 | 29.62 | 26.07 | 11.08 | 11.91 | 10.78 | 17.21 | 43.16 |
| Mistral-7B | 1-shot PubMedBERT-NeuML | 20.89 | 24.04 | 22.31 | 7.16 | 8.16 | 6.96 | 12.31 | 37.82 |
| Mistral-7B | 3-shot PubMedBERT-NeuML | 21.10 | 26.23 | 23.33 | 8.35 | 9.86 | 8.28 | 14.69 | 42.53 |
| Mistral-7B | 5-shot PubMedBERT-NeuML | 21.62 | 26.89 | 23.90 | 8.40 | 9.84 | 8.37 | 15.78 | 43.58 |
| Mistral-7B | 1-shot BioMedBERT-ft | 22.34 | 28.44 | 24.98 | 8.90 | 10.00 | 8.51 | 13.70 | 38.22 |
| Mistral-7B | 3-shot BioMedBERT-ft | 22.56 | 29.77 | 25.60 | 9.85 | 11.38 | 9.54 | 16.01 | 43.04 |
| Mistral-7B | 5-shot BioMedBERT-ft | 23.21 | 30.37 | 26.22 | 9.75 | 11.44 | 9.65 | 17.27 | 44.74 |
| Mistral-7B | 1-shot Gold heuristic | 37.61 | 53.90 | 44.28 | 16.10 | 22.19 | 17.55 | 31.13 | 50.90 |
| Mistral-7B | 3-shot Gold heuristic | 39.13 | 50.71 | 44.07 | 17.65 | 21.68 | 18.18 | 35.15 | 55.77 |
| Mistral-7B | 5-shot Gold heuristic | 39.92 | 50.08 | 44.29 | 17.72 | 21.74 | 18.36 | 36.75 | 57.13 |
| Mistral-7B-Instruct | zero-shot | 27.08 | 26.11 | 26.55 | 23.65 | 13.67 | 13.69 | 18.36 | 51.37 |
| Mistral-7B-Instruct | zero-shot CoT | 25.52 | 23.46 | 24.41 | 21.59 | 12.12 | 12.30 | 16.28 | 46.10 |
| Mistral-7B-Instruct | 1-shot Random | 28.41 | 26.61 | 27.44 | 22.67 | 12.91 | 13.32 | 20.14 | 44.62 |
| Mistral-7B-Instruct | 3-shot Random | 27.70 | 27.20 | 27.41 | 21.23 | 12.95 | 13.12 | 19.79 | 41.57 |
| Mistral-7B-Instruct | 5-shot Random | 27.49 | 27.44 | 27.43 | 20.31 | 13.09 | 13.01 | 19.46 | 39.99 |
| Mistral-7B-Instruct | 1-shot BM25 | 30.06 | 28.06 | 28.97 | 23.58 | 13.89 | 14.30 | 21.87 | 48.11 |
| Mistral-7B-Instruct | 3-shot BM25 | 30.63 | 29.80 | 30.16 | 22.10 | 14.77 | 14.79 | 23.22 | 47.75 |
| Mistral-7B-Instruct | 5-shot BM25 | 31.44 | 30.80 | 31.08 | 20.96 | 15.08 | 14.95 | 24.09 | 47.72 |
| Mistral-7B-Instruct | 1-shot PubMedBERT-NeuML | 29.16 | 27.17 | 28.09 | 22.73 | 13.23 | 13.74 | 21.27 | 47.23 |
| Mistral-7B-Instruct | 3-shot PubMedBERT-NeuML | 29.38 | 28.43 | 28.86 | 21.31 | 13.74 | 13.77 | 22.17 | 47.17 |
| Mistral-7B-Instruct | 5-shot PubMedBERT-NeuML | 30.34 | 29.62 | 29.94 | 20.17 | 14.18 | 13.99 | 23.33 | 47.87 |
| Mistral-7B-Instruct | 1-shot BioMedBERT-ft | 29.36 | 27.31 | 28.25 | 23.40 | 13.42 | 13.92 | 21.36 | 47.00 |
| Mistral-7B-Instruct | 3-shot BioMedBERT-ft | 29.91 | 29.04 | 29.42 | 22.03 | 14.22 | 14.36 | 22.86 | 47.50 |
| Mistral-7B-Instruct | 5-shot BioMedBERT-ft | 31.05 | 30.63 | 30.79 | 20.54 | 14.75 | 14.53 | 24.18 | 48.06 |
| Mistral-7B-Instruct | 1-shot Gold heuristic | 32.20 | 30.09 | 31.06 | 24.45 | 15.47 | 15.31 | 24.96 | 50.32 |
| Mistral-7B-Instruct | 3-shot Gold heuristic | 37.78 | 36.87 | 37.26 | 24.74 | 18.21 | 17.48 | 32.28 | 53.39 |
| Mistral-7B-Instruct | 5-shot Gold heuristic | 41.15 | 40.72 | 40.87 | 24.75 | 19.58 | 18.69 | 36.27 | 55.11 |
| Mistral-Nemo-13B | zero-shot | 15.39 | 29.49 | 20.22 | 13.49 | 14.31 | 10.86 | 12.25 | 31.63 |
| Mistral-Nemo-13B | zero-shot CoT | 11.64 | 24.01 | 15.66 | 11.11 | 11.63 | 8.23 | 9.67 | 37.51 |
| Mistral-Nemo-13B | 1-shot Random | 12.23 | 18.40 | 14.58 | 7.00 | 7.45 | 6.37 | 6.62 | 21.43 |
| Mistral-Nemo-13B | 3-shot Random | 12.44 | 22.97 | 16.13 | 6.81 | 8.30 | 6.41 | 9.14 | 27.04 |
| Mistral-Nemo-13B | 5-shot Random | 12.78 | 24.35 | 16.76 | 6.16 | 8.21 | 5.96 | 9.53 | 27.53 |
| Mistral-Nemo-13B | 1-shot BM25 | 22.83 | 27.54 | 24.87 | 11.98 | 11.32 | 10.75 | 14.09 | 39.70 |
| Mistral-Nemo-13B | 3-shot BM25 | 23.37 | 29.52 | 26.08 | 12.07 | 12.05 | 11.06 | 17.05 | 44.53 |
| Mistral-Nemo-13B | 5-shot BM25 | 22.75 | 30.86 | 26.18 | 11.31 | 12.73 | 11.01 | 17.66 | 44.95 |
| Mistral-Nemo-13B | 1-shot PubMedBERT-NeuML | 18.22 | 22.84 | 20.11 | 8.26 | 9.92 | 8.04 | 10.99 | 34.41 |
| Mistral-Nemo-13B | 3-shot PubMedBERT-NeuML | 20.63 | 26.25 | 23.08 | 8.62 | 9.60 | 8.16 | 14.93 | 42.94 |
| Mistral-Nemo-13B | 5-shot PubMedBERT-NeuML | 20.92 | 27.83 | 23.87 | 8.61 | 10.48 | 8.47 | 16.14 | 44.68 |
| Mistral-Nemo-13B | 1-shot BioMedBERT-ft | 19.95 | 26.28 | 22.56 | 9.65 | 10.93 | 9.05 | 12.33 | 35.33 |
| Mistral-Nemo-13B | 3-shot BioMedBERT-ft | 21.93 | 28.96 | 24.92 | 9.65 | 10.93 | 9.14 | 15.91 | 42.67 |
| Mistral-Nemo-13B | 5-shot BioMedBERT-ft | 22.57 | 30.50 | 25.92 | 10.09 | 11.54 | 9.58 | 17.32 | 44.98 |
| Mistral-Nemo-13B | 1-shot Gold heuristic | 29.93 | 44.48 | 35.62 | 15.03 | 20.73 | 15.71 | 24.97 | 44.03 |
| Mistral-Nemo-13B | 3-shot Gold heuristic | 36.80 | 49.04 | 41.98 | 17.02 | 21.25 | 17.47 | 33.33 | 54.59 |
| Mistral-Nemo-13B | 5-shot Gold heuristic | 38.42 | 50.14 | 43.44 | 17.73 | 21.96 | 18.12 | 36.04 | 56.91 |
| Mistral-Nemo-13B-Instruct | zero-shot | 23.28 | 22.94 | 23.08 | 19.64 | 11.97 | 11.58 | 15.95 | 48.39 |
| Mistral-Nemo-13B-Instruct | zero-shot CoT | 21.54 | 20.48 | 20.97 | 19.03 | 11.96 | 10.78 | 14.03 | 48.54 |
| Mistral-Nemo-13B-Instruct | 1-shot Random | 18.00 | 24.06 | 20.57 | 11.65 | 9.49 | 9.17 | 11.88 | 31.12 |
| Mistral-Nemo-13B-Instruct | 3-shot Random | 15.31 | 27.64 | 19.71 | 9.60 | 10.76 | 8.86 | 11.83 | 29.86 |
| Mistral-Nemo-13B-Instruct | 5-shot Random | 14.38 | 27.77 | 18.95 | 7.95 | 10.80 | 7.94 | 11.21 | 27.59 |
| Mistral-Nemo-13B-Instruct | 1-shot BM25 | 24.93 | 29.21 | 26.85 | 14.58 | 12.40 | 12.22 | 16.49 | 43.53 |
| Mistral-Nemo-13B-Instruct | 3-shot BM25 | 22.51 | 30.88 | 26.03 | 11.90 | 12.55 | 11.32 | 17.21 | 43.55 |
| Mistral-Nemo-13B-Instruct | 5-shot BM25 | 22.65 | 31.77 | 26.44 | 11.24 | 13.35 | 11.37 | 17.82 | 43.28 |
| Mistral-Nemo-13B-Instruct | 1-shot PubMedBERT-NeuML | 22.48 | 25.83 | 24.00 | 11.42 | 10.13 | 9.57 | 14.56 | 41.77 |
| Mistral-Nemo-13B-Instruct | 3-shot PubMedBERT-NeuML | 20.80 | 27.78 | 23.78 | 9.61 | 10.77 | 9.25 | 15.59 | 42.50 |
| Mistral-Nemo-13B-Instruct | 5-shot PubMedBERT-NeuML | 21.30 | 28.65 | 24.43 | 8.87 | 11.08 | 9.10 | 16.58 | 43.45 |
| Mistral-Nemo-13B-Instruct | 1-shot BioMedBERT-ft | 18.93 | 24.54 | 21.30 | 13.04 | 13.18 | 10.96 | 12.82 | 32.62 |
| Mistral-Nemo-13B-Instruct | 3-shot BioMedBERT-ft | 29.61 | 31.35 | 30.42 | 20.14 | 14.87 | 14.71 | 22.76 | 53.77 |
| Mistral-Nemo-13B-Instruct | 5-shot BioMedBERT-ft | 29.80 | 32.77 | 31.19 | 19.64 | 15.57 | 14.94 | 24.06 | 55.45 |
| Mistral-Nemo-13B-Instruct | 1-shot Gold heuristic | 33.89 | 43.64 | 38.12 | 18.89 | 19.22 | 16.90 | 27.82 | 52.17 |
| Mistral-Nemo-13B-Instruct | 3-shot Gold heuristic | 35.63 | 51.21 | 41.99 | 16.28 | 21.25 | 16.93 | 33.10 | 52.65 |
| Mistral-Nemo-13B-Instruct | 5-shot Gold heuristic | 37.33 | 53.19 | 43.83 | 15.54 | 22.13 | 16.98 | 35.83 | 53.29 |
| Qwen2-7B | zero-shot | 24.49 | 26.31 | 25.33 | 20.95 | 13.92 | 13.11 | 17.78 | 40.97 |
| Qwen2-7B | zero-shot CoT | 22.12 | 27.91 | 24.64 | 19.05 | 13.96 | 12.55 | 16.68 | 40.19 |
| Qwen2-7B | 1-shot Random | 17.99 | 24.06 | 20.57 | 11.63 | 9.48 | 9.16 | 11.89 | 30.99 |
| Qwen2-7B | 3-shot Random | 15.30 | 27.61 | 19.69 | 9.62 | 10.79 | 8.87 | 11.83 | 29.86 |
| Qwen2-7B | 5-shot Random | 14.36 | 27.75 | 18.92 | 7.92 | 10.78 | 7.90 | 11.20 | 27.52 |
| Qwen2-7B | 1-shot BM25 | 24.92 | 29.17 | 26.84 | 14.50 | 12.30 | 12.13 | 16.48 | 43.55 |
| Qwen2-7B | 3-shot BM25 | 22.53 | 30.89 | 26.05 | 11.93 | 12.54 | 11.33 | 17.21 | 43.50 |
| Qwen2-7B | 5-shot BM25 | 22.65 | 31.74 | 26.43 | 11.23 | 13.35 | 11.35 | 17.80 | 43.35 |
| Qwen2-7B | 1-shot PubMedBERT-NeuML | 22.48 | 25.80 | 23.99 | 11.41 | 10.14 | 9.56 | 14.56 | 41.72 |
| Qwen2-7B | 3-shot PubMedBERT-NeuML | 20.81 | 27.77 | 23.78 | 9.65 | 10.80 | 9.28 | 15.60 | 42.50 |
| Qwen2-7B | 5-shot PubMedBERT-NeuML | 21.33 | 28.69 | 24.46 | 8.83 | 11.11 | 9.10 | 16.59 | 43.49 |
| Qwen2-7B | 1-shot BioMedBERT-ft | 23.70 | 29.15 | 26.11 | 13.45 | 12.05 | 11.26 | 15.96 | 43.33 |
| Qwen2-7B | 3-shot BioMedBERT-ft | 22.09 | 31.31 | 25.89 | 11.05 | 12.86 | 10.71 | 17.06 | 43.43 |
| Qwen2-7B | 5-shot BioMedBERT-ft | 22.75 | 32.51 | 26.76 | 10.25 | 12.69 | 10.41 | 18.31 | 43.98 |
| Qwen2-7B | 1-shot Gold heuristic | 33.95 | 43.61 | 38.15 | 18.90 | 19.21 | 16.91 | 27.87 | 52.23 |
| Qwen2-7B | 3-shot Gold heuristic | 35.66 | 51.20 | 42.00 | 16.30 | 21.23 | 16.97 | 33.10 | 52.54 |
| Qwen2-7B | 5-shot Gold heuristic | 37.25 | 53.17 | 43.77 | 15.49 | 22.03 | 16.93 | 35.78 | 53.47 |
| Qwen2-7B-Instruct | zero-shot | 28.82 | 26.78 | 27.73 | 23.06 | 14.14 | 14.10 | 20.11 | 45.86 |
| Qwen2-7B-Instruct | zero-shot CoT | 27.74 | 26.35 | 26.99 | 22.13 | 13.89 | 13.64 | 18.59 | 46.30 |
| Qwen2-7B-Instruct | 1-shot Random | 26.93 | 27.15 | 26.99 | 20.22 | 13.33 | 13.30 | 18.27 | 44.70 |
| Qwen2-7B-Instruct | 3-shot Random | 26.12 | 27.74 | 26.86 | 18.85 | 13.14 | 12.87 | 17.89 | 44.80 |
| Qwen2-7B-Instruct | 5-shot Random | 25.37 | 28.01 | 26.59 | 17.86 | 13.04 | 12.67 | 17.46 | 43.81 |
| Qwen2-7B-Instruct | 1-shot BM25 | 28.47 | 28.61 | 28.49 | 20.47 | 14.04 | 13.94 | 20.02 | 47.76 |
| Qwen2-7B-Instruct | 3-shot BM25 | 28.61 | 30.13 | 29.31 | 18.77 | 14.19 | 13.95 | 20.98 | 50.07 |
| Qwen2-7B-Instruct | 5-shot BM25 | 29.32 | 31.24 | 30.22 | 18.23 | 14.40 | 14.15 | 21.98 | 50.52 |
| Qwen2-7B-Instruct | 1-shot PubMedBERT-NeuML | 27.75 | 27.67 | 27.66 | 20.34 | 13.80 | 13.63 | 19.34 | 47.24 |
| Qwen2-7B-Instruct | 3-shot PubMedBERT-NeuML | 27.82 | 28.92 | 28.32 | 18.11 | 13.00 | 13.08 | 20.24 | 50.52 |
| Qwen2-7B-Instruct | 5-shot PubMedBERT-NeuML | 28.08 | 29.84 | 28.90 | 16.69 | 13.25 | 12.96 | 21.00 | 51.38 |
| Qwen2-7B-Instruct | 1-shot BioMedBERT-ft | 28.00 | 27.78 | 27.85 | 20.27 | 13.69 | 13.59 | 19.54 | 47.23 |
| Qwen2-7B-Instruct | 3-shot BioMedBERT-ft | 28.42 | 29.50 | 28.91 | 18.87 | 13.81 | 13.64 | 20.80 | 50.02 |
| Qwen2-7B-Instruct | 5-shot BioMedBERT-ft | 29.20 | 30.98 | 30.03 | 18.33 | 14.01 | 13.88 | 22.10 | 51.87 |
| Qwen2-7B-Instruct | 1-shot Gold heuristic | 31.72 | 31.41 | 31.51 | 22.25 | 16.26 | 15.32 | 23.64 | 50.99 |
| Qwen2-7B-Instruct | 3-shot Gold heuristic | 36.75 | 38.31 | 37.46 | 22.19 | 17.80 | 16.86 | 30.62 | 56.69 |
| Qwen2-7B-Instruct | 5-shot Gold heuristic | 39.48 | 42.40 | 40.83 | 21.37 | 18.90 | 17.55 | 34.63 | 58.92 |
| Qwen2-72B | zero-shot | 27.23 | 26.53 | 26.84 | 22.55 | 14.06 | 13.70 | 18.46 | 43.28 |
| Qwen2-72B | zero-shot CoT | 25.11 | 25.90 | 25.46 | 20.32 | 13.14 | 12.56 | 17.80 | 40.93 |
| Qwen2-72B | 1-shot Random | 21.78 | 26.56 | 23.87 | 15.24 | 11.66 | 11.32 | 15.12 | 37.11 |
| Qwen2-72B | 3-shot Random | 21.00 | 29.12 | 24.34 | 13.64 | 12.05 | 10.90 | 15.73 | 37.47 |
| Qwen2-72B | 5-shot Random | 20.70 | 29.47 | 24.26 | 12.60 | 11.98 | 10.52 | 15.82 | 36.24 |
| Qwen2-72B | 1-shot BM25 | 27.53 | 30.10 | 28.69 | 17.36 | 13.15 | 13.37 | 19.14 | 46.55 |
| Qwen2-72B | 3-shot BM25 | 27.79 | 31.49 | 29.48 | 16.21 | 13.61 | 13.49 | 21.01 | 49.26 |
| Qwen2-72B | 5-shot BM25 | 28.05 | 32.49 | 30.07 | 15.27 | 13.79 | 13.32 | 21.96 | 49.40 |
| Qwen2-72B | 1-shot PubMedBERT-NeuML | 25.31 | 26.98 | 26.04 | 14.68 | 11.46 | 11.35 | 17.08 | 44.63 |
| Qwen2-72B | 3-shot PubMedBERT-NeuML | 25.79 | 28.54 | 27.04 | 13.24 | 11.30 | 11.07 | 19.02 | 48.00 |
| Qwen2-72B | 5-shot PubMedBERT-NeuML | 26.08 | 29.20 | 27.50 | 13.18 | 11.84 | 11.28 | 19.99 | 49.01 |
| Qwen2-72B | 1-shot BioMedBERT-ft | 26.46 | 29.28 | 27.73 | 16.79 | 13.31 | 12.89 | 18.14 | 44.90 |
| Qwen2-72B | 3-shot BioMedBERT-ft | 27.25 | 30.88 | 28.89 | 14.82 | 12.83 | 12.43 | 20.34 | 48.69 |
| Qwen2-72B | 5-shot BioMedBERT-ft | 27.99 | 31.79 | 29.72 | 15.08 | 13.50 | 12.95 | 21.52 | 49.70 |
| Qwen2-72B | 1-shot Gold heuristic | 36.03 | 41.06 | 38.29 | 21.79 | 19.03 | 17.55 | 29.50 | 54.07 |
| Qwen2-72B | 3-shot Gold heuristic | 40.40 | 47.53 | 43.60 | 20.67 | 20.41 | 18.71 | 35.77 | 59.32 |
| Qwen2-72B | 5-shot Gold heuristic | 42.17 | 48.85 | 45.21 | 20.30 | 20.76 | 18.97 | 38.41 | 59.97 |
| Qwen2-72B-Instruct | zero-shot | 31.04 | 28.17 | 29.50 | 24.11 | 14.20 | 14.42 | 21.29 | 48.11 |
| Qwen2-72B-Instruct | zero-shot CoT | 30.21 | 27.50 | 28.76 | 22.78 | 14.10 | 14.12 | 21.00 | 47.16 |
| Qwen2-72B-Instruct | 1-shot Random | 24.33 | 28.57 | 26.19 | 17.67 | 13.95 | 12.81 | 17.74 | 40.80 |
| Qwen2-72B-Instruct | 3-shot Random | 24.02 | 29.42 | 26.33 | 15.90 | 13.27 | 12.04 | 18.05 | 41.22 |
| Qwen2-72B-Instruct | 5-shot Random | 23.33 | 29.94 | 26.14 | 14.24 | 13.45 | 11.53 | 17.84 | 40.66 |
| Qwen2-72B-Instruct | 1-shot BM25 | 28.51 | 30.95 | 29.59 | 19.97 | 15.29 | 14.62 | 21.41 | 47.27 |
| Qwen2-72B-Instruct | 3-shot BM25 | 29.98 | 33.26 | 31.44 | 18.89 | 15.89 | 15.08 | 23.90 | 51.18 |
| Qwen2-72B-Instruct | 5-shot BM25 | 30.58 | 34.18 | 32.21 | 17.42 | 15.73 | 14.77 | 24.99 | 52.06 |
| Qwen2-72B-Instruct | 1-shot PubMedBERT-NeuML | 26.80 | 29.15 | 27.84 | 18.25 | 13.83 | 13.17 | 19.76 | 45.59 |
| Qwen2-72B-Instruct | 3-shot PubMedBERT-NeuML | 28.46 | 31.37 | 29.74 | 17.13 | 13.97 | 13.38 | 22.50 | 50.50 |
| Qwen2-72B-Instruct | 5-shot PubMedBERT-NeuML | 29.21 | 32.40 | 30.64 | 16.10 | 14.48 | 13.46 | 23.79 | 52.34 |
| Qwen2-72B-Instruct | 1-shot BioMedBERT-ft | 26.81 | 29.78 | 28.14 | 18.97 | 14.60 | 13.73 | 19.96 | 44.80 |
| Qwen2-72B-Instruct | 3-shot BioMedBERT-ft | 29.48 | 32.35 | 30.74 | 18.04 | 14.86 | 14.13 | 23.50 | 50.92 |
| Qwen2-72B-Instruct | 5-shot BioMedBERT-ft | 30.96 | 33.69 | 32.19 | 18.12 | 15.44 | 14.81 | 25.30 | 52.59 |
| Qwen2-72B-Instruct | 1-shot Gold heuristic | 31.06 | 34.61 | 32.64 | 21.32 | 17.84 | 16.00 | 25.05 | 48.46 |
| Qwen2-72B-Instruct | 3-shot Gold heuristic | 40.00 | 43.13 | 41.39 | 22.72 | 20.24 | 18.63 | 35.75 | 59.15 |
| Qwen2-72B-Instruct | 5-shot Gold heuristic | 43.59 | 46.20 | 44.75 | 22.83 | 21.21 | 19.70 | 40.06 | 61.66 |
