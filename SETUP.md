# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
nano .env
```

### 3. Setup MongoDB

Ensure MongoDB is running with Atlas Vector Search enabled:

```bash
# Test connection
mongosh $MONGO_URI --eval "db.adminCommand('ping')"
```

### 4. Prepare Data

Place your CSV file with embeddings in the project directory or specify the path in `.env`:

```env
CSV_FILE_PATH=/path/to/cardioRef_embbeding_keywords.csv
```

### 5. Populate Database

```bash
python RAG_poblate_db.py
```

### 6. Setup Ollama (for Evaluation)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.3
ollama pull llama3.1
ollama pull phi4
ollama pull qwen2.5:7b
ollama pull deepseek-r1:7b
```

## Verification

Test the RAG system:

```bash
python RAG_Mongodb.py
```

Test evaluation:

```bash
python LLM_answer_supervised_evaluation_strucutred_output.py
```

## Common Issues

### MongoDB Connection Error

Check your connection string format:
```
mongodb://[username:password@]host:port/?directConnection=true
```

### CUDA Out of Memory

- Reduce batch size
- Use quantized models
- Enable CPU offloading

### Model Download Issues

Set HuggingFace cache directory:
```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

## Directory Structure

After setup, your directory should look like:

```
paper_code/
├── .env                    # Your configuration (not in git)
├── .env.example           # Configuration template
├── .gitignore
├── README.md
├── SETUP.md              # This file
├── requirements.txt
├── clean_notebooks.sh
├── RAG_Mongodb.py
├── RAG_poblate_db.py
├── LLM_answer_supervised_evaluation_strucutred_output.py
├── evaluator_variability_vllm_*.py
├── *.ipynb               # Analysis notebooks
├── mechanistic/          # Probing, steering, shuffling (Fig 3c–d, Fig 4)
└── replicas_vllm_deterministic/  # Output directory (created automatically)
```

## Mechanistic Interpretability Setup (`mechanistic/`)

The mechanistic subpackage runs `Llama3.1:8b` locally via `TransformerLens`. It needs:

1. **A HuggingFace token** for the gated Llama-3.1-8B-Instruct model. After accepting the
   model licence on HuggingFace, set it in `.env`:

   ```bash
   echo "HF_TOKEN=hf_..." >> .env
   ```

2. **A CUDA GPU with ≥40 GB VRAM** (H100 / A100). Extraction and steering on the primary
   8B model is single-GPU. The optional 70B comparison models in
   `mechanistic/config/config.yaml` require 4-bit AWQ quantization or 2× H100s.

3. **`mechanistic/`-specific dependencies** (already in `requirements.txt`):
   `transformer-lens`, `datasets`, `h5py`, `scikit-learn`, `statsmodels`, `plotly`,
   `kaleido`, `accelerate`.

Verify the install before launching the pipeline:

```bash
python -c "import transformer_lens, h5py, plotly, sklearn, statsmodels; print('OK')"
```

See `mechanistic/README.md` for the full pipeline and per-script outputs.

## Next Steps

1. Review the README.md for detailed usage instructions
2. Run the analysis notebooks to reproduce paper results
3. Customize evaluation parameters in the scripts
4. Explore variability analysis results
5. Reproduce Figures 3c–d and Figure 4 via `mechanistic/`
