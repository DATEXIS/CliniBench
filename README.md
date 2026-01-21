# CliniBench: Clinical Outcome Prediction Benchmark

**CliniBench** is the first comprehensive benchmark for comparing encoder-based classifiers and generative large language models (LLMs) for discharge diagnosis prediction from admission notes in the MIMIC-IV dataset.
## Installation

### Prerequisites
- Python 3.10
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

### Generating the admission notes

### Generating the dataset splits

## Usage
### Zero-shot

**Step 1: Start the vLLM server**
Launch a vLLM server that provides access to your model. By default, the script assumes it's running on `localhost`. If your server is on a different host, you'll need to adjust the `--vllm_ip` parameter in Step 2.

**Step 2: Generate diagnosis predictions**
Run the prediction script with your model and data:
```bash
python src/generative_models/predict_diagnoses.py \
   --vllm_ip=http://localhost:8000 \
   --test_data=data/mimic-iv/icd-10/icu/test_10_icu.parquet \
   --icd10_code_file=data/icd10_codes.csv \
   --icd9_code_file=data/icd9_codes.csv \
   --results_file=qwen2.5-3B-mimic-iv-icu-icd10.parquet
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