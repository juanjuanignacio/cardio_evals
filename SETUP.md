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
python RAG_Mongodb_gemma2.py
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
├── RAG_Mongodb_gemma2.py
├── RAG_poblate_db.py
├── LLM_answer_supervised_evaluation_strucutred_output.py
├── evaluator_variability_vllm_*.py
├── *.ipynb               # Analysis notebooks
└── replicas_vllm_deterministic/  # Output directory (created automatically)
```

## Next Steps

1. Review the README.md for detailed usage instructions
2. Run the analysis notebooks to reproduce paper results
3. Customize evaluation parameters in the scripts
4. Explore variability analysis results
