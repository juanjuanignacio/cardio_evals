# RAGQA: RAG-based Question Answering System for Cardiovascular Research

This repository contains the code and analysis for the DRAGQA (Domain-specific Retrieval-Augmented Generation for Question Answering) paper, which implements and evaluates a RAG-based question answering system for cardiovascular research.

## Overview

The repository contains:
- **RAG system implementation** using MongoDB vector search and LLMs
- **Evaluation framework** for assessing answer quality
- **Variability analysis** of LLM evaluators
- **Statistical analysis notebooks** for paper results

## Repository Structure

```
paper_code/
├── RAG_Mongodb_gemma2.py              # Main RAG system implementation
├── RAG_poblate_db.py                  # Database population script
├── LLM_answer_supervised_evaluation_strucutred_output.py  # Supervised evaluation
├── evaluator_variability_vllm_batch.py                    # Batch evaluation (vLLM)
├── evaluator_variability_vllm_deterministic.py            # Sequential evaluation (vLLM)
├── evaluator_variability_vllm_deterministic_quantized.py  # Quantized 70B models
├── evaluator_variability_analysis.ipynb                   # Variability analysis
├── evaluator_variability_analysis_seed.ipynb              # Seed effect analysis
├── evaluator_variability_comparison.ipynb                 # Comparison across settings
├── statistics_RAGQA_2.ipynb                              # Main statistical analysis
├── statistics_RAGQA_2linguistic.ipynb                    # Linguistic analysis
├── .env.example                                          # Configuration template
├── .gitignore                                            # Git ignore rules
└── README.md                                             # This file
```

## Requirements

### System Requirements
- Python 3.8+
- MongoDB 5.0+ with Atlas Vector Search support
- CUDA-capable GPU (recommended for LLM inference)
- 16GB+ RAM (32GB+ recommended for large models)

### Python Dependencies

```bash
pip install pandas numpy pymongo sentence-transformers transformers torch langchain langchain-ollama pydantic tqdm jupyter vllm
```

For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd paper_code
```

### 2. Configure Environment Variables

Copy the example configuration file and edit it with your settings:

```bash
cp .env.example .env
```

Edit `.env` with your MongoDB credentials and configuration:

```env
MONGODB_HOST=your-mongodb-host
MONGODB_PORT=27017
MONGO_URI=mongodb://username:password@host:port/?directConnection=true
CSV_FILE_PATH=path/to/cardioRef_embbeding_keywords.csv
```

### 3. Prepare the Database

First, ensure you have the CSV file with pre-computed embeddings for the cardiovascular references dataset. Then populate the MongoDB database:

```bash
python RAG_poblate_db.py
```

This script will:
- Connect to MongoDB
- Process the CSV file in chunks
- Create the vector search index
- Optionally run a test query

### 4. Set Up Ollama (for evaluation)

Install and configure Ollama with the required models:

```bash
ollama pull llama3.3
ollama pull llama3.1
ollama pull deepseek-r1:7b
ollama pull deepseek-r1:70b
ollama pull phi4
ollama pull qwen2.5:7b
```

## Usage

### Running the RAG System

To use the RAG system for question answering:

```bash
# Set environment variables
export MONGO_URI="your-mongodb-uri"
export MODEL_NAME="google/gemma-2-2b-it"

# Run the RAG system
python RAG_Mongodb_gemma2.py
```

You can modify the query in the `main()` function or import the functions for programmatic use:

```python
from RAG_Mongodb_gemma2 import get_search_result, create_rag_prompt, generate_answer

# Query the system
query = "What are the behaviours of neutrophils?"
context = get_search_result(query, collection)
answer = generate_answer(prompt, model, tokenizer)
```

### Running Evaluations

#### Supervised Evaluation

Evaluate answers using multiple LLM judges:

```bash
python LLM_answer_supervised_evaluation_strucutred_output.py
```

This will evaluate all answers in the database using the configured models and store results in MongoDB.

#### Evaluator Variability Analysis

To study evaluator variability with vLLM:

```bash
# Batch processing mode (faster)
python evaluator_variability_vllm_batch.py

# Sequential mode (more deterministic)
python evaluator_variability_vllm_deterministic.py llama3.1:7b

# Quantized 70B models (single GPU)
python evaluator_variability_vllm_deterministic_quantized.py llama3.3:70b
```

Arguments:
- First argument: model name (optional, runs all models if omitted)
- Second argument: replica ID 0-4 (optional, runs all replicas if omitted)

### Analysis Notebooks

Open Jupyter notebooks for statistical analysis:

```bash
jupyter notebook
```

Available notebooks:
- `statistics_RAGQA_2.ipynb`: Main statistical analysis for the paper
- `statistics_RAGQA_2linguistic.ipynb`: Linguistic analysis of answers
- `evaluator_variability_analysis.ipynb`: Analysis of evaluator variability
- `evaluator_variability_analysis_seed.ipynb`: Impact of random seeds
- `evaluator_variability_comparison.ipynb`: Cross-condition comparison

## Key Features

### RAG System
- **Vector Search**: Semantic search using MongoDB Atlas Vector Search
- **Embeddings**: High-quality embeddings with `thenlper/gte-large` (1024 dimensions)
- **LLM Generation**: Flexible LLM backend (Gemma-2, Llama, etc.)
- **Configurable Retrieval**: Adjust number of candidates and results

### Evaluation Framework
- **Multi-dimensional Scoring**: Accuracy, Clarity, Completeness (1-7 scale)
- **Structured Output**: Validated JSON output using Pydantic
- **Multiple Evaluators**: Support for various LLM judges
- **Retry Logic**: Robust evaluation with automatic retries

### Variability Analysis
- **Deterministic Mode**: Fixed seed for reproducibility
- **Batch Processing**: Efficient parallel evaluation
- **Quantized Models**: Support for 4-bit AWQ quantization
- **Replica Analysis**: Multiple evaluation runs for variance estimation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ns,
  title={ns},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_HOST` | MongoDB server address | `localhost` |
| `MONGODB_PORT` | MongoDB port | `27017` |
| `MONGODB_NAME` | Database name | `RAGQA` |
| `MONGO_URI` | Full MongoDB connection string | Required for RAG scripts |
| `MONGO_DB_NAME` | RAG database name | `cardio_refs_2` |
| `MONGO_COLLECTION_NAME` | Collection name | `abst_refs` |
| `CSV_FILE_PATH` | Path to embeddings CSV | `cardioRef_embbeding_keywords.csv` |
| `CHUNK_SIZE` | Rows per processing chunk | `5000` |
| `MODEL_NAME` | HuggingFace model name | `google/gemma-2-2b-it` |
| `USE_GPU` | Enable GPU inference | `true` |
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://localhost:11434` |
| `TEST_QUERY` | Test query for validation | (optional) |

## Troubleshooting

### MongoDB Connection Issues
- Verify MongoDB is running: `mongosh --eval "db.adminCommand('ping')"`
- Check firewall settings and network connectivity
- Ensure credentials are correct in `.env`

### Out of Memory Errors
- Reduce `CHUNK_SIZE` for database population
- Use quantized models for large LLMs
- Reduce batch size in vLLM scripts
- Enable CPU offloading if needed

### Model Loading Issues
- Ensure HuggingFace cache has sufficient space
- Check internet connectivity for model downloads
- Verify CUDA compatibility for GPU models
- Use `trust_remote_code=True` for custom architectures

### Evaluation Failures
- Check Ollama is running: `ollama list`
- Verify models are pulled: `ollama pull <model-name>`
- Increase `max_retries` in evaluation scripts
- Check MongoDB has sufficient storage

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].

## Acknowledgments

This work uses:
- MongoDB Atlas Vector Search
- HuggingFace Transformers
- Sentence Transformers (GTE-Large)
- vLLM for efficient inference
- Ollama for local LLM serving
