#!/usr/bin/env python3
"""
LLM Evaluator Variability Study - VLLM Batch Processing Mode

This script runs evaluations using vllm with BATCH PROCESSING for faster inference.
Multiple requests are processed simultaneously in batches.

Configuration:
- BATCH processing (multiple inferences at once)
- Temperature = 0
- Fixed seed = 42
- Batch processing enabled for speed
- Expected result: Zero variability across replicas (determinism maintained)
"""

# Import libraries (NO environment variables to force sequential processing)
import os
import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from tqdm import tqdm
import warnings
import json
import re
from typing import Optional, Dict, Any
from transformers import AutoTokenizer
warnings.filterwarnings('ignore')

# vllm imports
from vllm import LLM, SamplingParams
from pydantic import BaseModel, field_validator
from typing import Literal

# ==================== CONFIGURATION ====================

MONGODB_HOST = os.getenv('MONGODB_HOST', 'localhost')
MONGODB_PORT = int(os.getenv('MONGODB_PORT', '27017'))
MONGODB_NAME = os.getenv('MONGODB_NAME', 'RAGQA')

# Models to test (use HuggingFace model paths)
MODEL_CONFIGS = {
    "llama3.1:7b": {
        "hf_path": "meta-llama/Llama-3.1-8B-Instruct",
        "supports_tools": True
    },
    "qwen2.5:7b": {
        "hf_path": "Qwen/Qwen2.5-7B-Instruct",
        "supports_tools": True
    },
    "phi4:14b": {
        "hf_path": "microsoft/phi-4",
        "supports_tools": True  # Changed to True to use structured output instead of manual JSON parsing
    },
    "llama3.3:70b": {
        "hf_path": "meta-llama/Llama-3.3-70B-Instruct",
        "supports_tools": True
    },
    "deepseek-r1:7b": {
        "hf_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "supports_tools": False
    },
    "deepseek-r1:70b": {
        "hf_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "supports_tools": False
    },
}

N_REPLICAS = 5
N_SAMPLES = None  # Use None for all samples
SEED = 42
TEMPERATURE = 0.0

# Output directory
OUTPUT_DIR = './replicas_vllm_deterministic'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("VLLM DETERMINISTIC EVALUATION - CONFIGURATION")
print("="*80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Models to test: {list(MODEL_CONFIGS.keys())}")
print(f"Number of replicas: {N_REPLICAS}")
print(f"Sample size: {'ALL' if N_SAMPLES is None else N_SAMPLES}")
print(f"Seed: {SEED} (deterministic)")
print(f"Temperature: {TEMPERATURE}")
print("="*80)

# ==================== EVALUATION SCHEMA ====================

class Eval(BaseModel):
    """Evaluation schema for structured output."""
    Accuracy: int
    Clarity: int
    Completeness: int
    Source: Literal['human', 'ai']
    Justification: str

    @field_validator('Accuracy', 'Clarity', 'Completeness')
    @classmethod
    def validate_score_range(cls, v):
        if not (1 <= v <= 7):
            raise ValueError(f'Score must be between 1 and 7, got {v}')
        return v

# ==================== UTILITY FUNCTIONS ====================

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from LLM response, handling various formats."""
    # Try to find JSON in code blocks first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError("No JSON found in response")

    return json.loads(json_str)

def get_db():
    """Connect to MongoDB."""
    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    return client[MONGODB_NAME]

def load_answers_sample(n_samples=None):
    """Load answers from MongoDB."""
    db = get_db()
    answers_collection = db["answers"]

    if n_samples is None:
        print("Loading ALL responses from MongoDB...")
        answers_data = list(answers_collection.find())
    else:
        print(f"Loading {n_samples} random responses from MongoDB...")
        pipeline = [{"$sample": {"size": n_samples}}]
        answers_data = list(answers_collection.aggregate(pipeline))

    df = pd.DataFrame(answers_data)
    print(f"Loaded {len(df)} responses from MongoDB")
    return df

# ==================== EVALUATION FUNCTIONS ====================

def create_evaluation_prompt(response: str, question: str, tokenizer=None, model_name: str = None) -> str:
    """Create evaluation prompt with optional chat template formatting."""
    base_prompt = f"""Evaluate the following response and provide a structured evaluation in JSON format.

Evaluation criteria:
1. Accuracy: Is the response accurate and factually correct? (Score 1-7)
2. Clarity: Is the response clear and easy to understand? (Score 1-7)
3. Completeness: Is the response complete and covers all aspects of the question? (Score 1-7)
4. Source: Is this from a human or AI? (Must be exactly 'human' or 'ai')
5. Justification: A text explaining your evaluation

**Question**: {question}

**Response to evaluate**: {response}

IMPORTANT: Respond ONLY with a valid JSON object in this exact format (no additional text):
{{
    "Accuracy": <integer 1-7>,
    "Clarity": <integer 1-7>,
    "Completeness": <integer 1-7>,
    "Source": "<human or ai>",
    "Justification": "<your explanation>"
}}"""

    # Apply chat template for models that need it (phi-4 and llama3.3)
    if tokenizer is not None and model_name and ("phi" in model_name.lower() or "llama3.3" in model_name.lower()):
        try:
            messages = [{"role": "user", "content": base_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted_prompt
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
            return base_prompt

    return base_prompt

def evaluate_single_response_vllm(
    response: str,
    question: str,
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer=None,
    model_name: str = None,
    max_retries: int = 10
) -> Dict[str, Any]:
    """
    Evaluate a single response using vllm.

    Args:
        response: The answer to evaluate
        question: The original question
        llm: vllm LLM instance
        sampling_params: vllm SamplingParams instance
        tokenizer: HuggingFace tokenizer (optional, for chat template)
        model_name: Model name (optional, to determine if chat template needed)
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary with evaluation results
    """
    prompt = create_evaluation_prompt(response, question, tokenizer, model_name)

    for retry in range(max_retries):
        try:
            # Generate with vllm
            outputs = llm.generate([prompt], sampling_params)
            response_text = outputs[0].outputs[0].text

            # Parse JSON
            eval_dict = extract_json_from_text(response_text)

            # Validate with Pydantic
            eval_result = Eval(**eval_dict)

            # Return validated result
            return {
                "accuracy_score": eval_result.Accuracy,
                "clarity_score": eval_result.Clarity,
                "completeness_score": eval_result.Completeness,
                "source_guessed": eval_result.Source,
                "justification": eval_result.Justification,
                "success": True,
                "raw_response": response_text[:500]  # Store first 500 chars
            }

        except Exception as e:
            if retry == max_retries - 1:
                return {
                    "accuracy_score": None,
                    "clarity_score": None,
                    "completeness_score": None,
                    "source_guessed": "unknown",
                    "justification": f"Failed after {max_retries} attempts: {str(e)}",
                    "success": False,
                    "raw_response": str(e)[:500]
                }

def run_sequential_replicas_vllm(
    df: pd.DataFrame,
    model_name: str,
    model_path: str,
    n_replicas: int = N_REPLICAS,
    single_replica_id: int = None
) -> pd.DataFrame:
    """
    Run sequential evaluation replicas using vllm (deterministic).

    IMPORTANT: All replicas use the SAME seed (42) to ensure:
    - Zero variability: All 5 replicas should produce IDENTICAL results
    - This tests vllm's determinism guarantee

    Args:
        df: DataFrame with answers to evaluate
        model_name: Name of the model (for output files)
        model_path: HuggingFace model path
        n_replicas: Number of replicas to run
        single_replica_id: If specified, only run this specific replica (0-4)

    Returns:
        DataFrame with all evaluation results
    """
    print(f"\n{'='*80}")
    print(f"BATCH VLLM EVALUATION: {model_name}")
    print(f"Model path: {model_path}")
    if single_replica_id is not None:
        print(f"Single replica mode: Running only replica {single_replica_id}")
    else:
        print(f"Replicas: {n_replicas}")
    print(f"Samples per replica: {len(df)}")
    print(f"Seed: {SEED} (SAME for all replicas - expect variability = 0)")
    print(f"Processing mode: BATCH (multiple inferences simultaneously)")
    print(f"{'='*80}")

    # Determine which replicas to run
    if single_replica_id is not None:
        replicas_to_run = [single_replica_id]
    else:
        replicas_to_run = range(n_replicas)

    # Run replicas
    all_results = []

    for replica_id in replicas_to_run:
        print(f"\n--- Replica {replica_id} (seed={SEED}) ---")

        # Initialize vllm model with SAME seed for all replicas
        # This should produce identical results across all replicas
        print(f"Loading model with vllm (seed={SEED})...")

        # Adjust memory utilization and parallelism based on model size
        # When running in parallel (one model per GPU set), use high utilization
        # Memory is freed between replica loads
        if "70b" in model_name.lower():
            # 70B models: Use tensor parallelism across 2 GPUs
            tensor_parallel = 2
            gpu_mem_util = 0.90  # High utilization with tensor parallelism
            max_len = 4096  # Full length context
            print(f"  Large model (70B): tensor_parallel_size={tensor_parallel}, gpu_memory_utilization={gpu_mem_util}, max_model_len={max_len}")
        else:
            # Small models: Single GPU
            tensor_parallel = 1
            gpu_mem_util = 0.95  # 95% utilization
            max_len = 4096
            print(f"  Small model: tensor_parallel_size={tensor_parallel}, gpu_memory_utilization={gpu_mem_util}, max_model_len={max_len}")

        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel,
            seed=SEED,  # SAME seed for all replicas
            trust_remote_code=True,
            max_model_len=max_len,
            gpu_memory_utilization=gpu_mem_util,
            attention_backend="FLASH_ATTN"  # Required for batch_invariant mode
        )

        # Load tokenizer for chat template support (phi-4 and llama3.3 need it)
        tokenizer = None
        if "phi" in model_name.lower() or "llama3.3" in model_name.lower():
            try:
                print(f"Loading tokenizer for chat template support...")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print(f"✓ Tokenizer loaded (chat template will be applied)")
            except Exception as e:
                print(f"⚠ Warning: Could not load tokenizer: {e}")
                tokenizer = None

        # Create sampling parameters with same seed
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            seed=SEED,  # SAME seed for all replicas
            max_tokens=1024,
            top_p=1.0,
            top_k=-1
        )

        print(f"Model loaded successfully!")
        print(f"Sampling params: temp={TEMPERATURE}, seed={SEED}")
        print(f"BATCH MODE: Processing all {len(df)} evaluations simultaneously")

        # ============================================================
        # BATCH PROCESSING: Prepare ALL prompts first
        # ============================================================
        print(f"Step 1: Preparing all prompts...")
        all_prompts = []
        all_metadata = []

        for idx, row in df.iterrows():
            # Prepare answer
            answer = row['answer']
            if 'bibliography_sources' in row and pd.notna(row['bibliography_sources']):
                if str(row['bibliography_sources']).strip():
                    answer = f"{answer}\n\nReferences:\n{row['bibliography_sources']}"

            # Create prompt
            prompt = create_evaluation_prompt(answer, row['question_text'], tokenizer, model_name)
            all_prompts.append(prompt)

            # Store metadata for later
            all_metadata.append({
                'replica_id': replica_id,
                'mode': 'batch_vllm',
                'model': model_name,
                'model_path': model_path,
                'answer_id': str(row['_id']),
                'question_id': row['question_id'],
                'question_text': row['question_text'],
                'response_text': answer,
                'true_source': 'human' if row['user'] not in ['AI', 'Deepseek'] else 'ai',
                'llm_category': 0 if row['user'] not in ['AI', 'Deepseek'] else (1 if row['user'] == 'AI' else 2),
                'seed': SEED,
                'temperature': TEMPERATURE,
            })

        print(f"✓ Prepared {len(all_prompts)} prompts")

        # ============================================================
        # BATCH PROCESSING: Send ALL prompts at once to vllm
        # ============================================================
        print(f"Step 2: Generating responses in BATCH (vllm will process multiple at once)...")
        outputs = llm.generate(all_prompts, sampling_params)
        print(f"✓ Batch generation completed!")

        # ============================================================
        # BATCH PROCESSING: Parse results
        # ============================================================
        print(f"Step 3: Parsing {len(outputs)} responses...")
        replica_results = []

        for i, (output, metadata) in enumerate(tqdm(zip(outputs, all_metadata), total=len(outputs), desc=f"Parsing responses")):
            response_text = output.outputs[0].text

            # Parse JSON from response
            try:
                eval_dict = extract_json_from_text(response_text)
                eval_result_obj = Eval(**eval_dict)

                eval_result = {
                    "accuracy_score": eval_result_obj.Accuracy,
                    "clarity_score": eval_result_obj.Clarity,
                    "completeness_score": eval_result_obj.Completeness,
                    "source_guessed": eval_result_obj.Source,
                    "justification": eval_result_obj.Justification,
                    "success": True,
                    "raw_response": response_text[:500]
                }
            except Exception as e:
                eval_result = {
                    "accuracy_score": None,
                    "clarity_score": None,
                    "completeness_score": None,
                    "source_guessed": "unknown",
                    "justification": f"Failed to parse: {str(e)}",
                    "success": False,
                    "raw_response": str(e)[:500]
                }

            # Combine metadata with evaluation result
            result = {**metadata, **eval_result}
            replica_results.append(result)

        print(f"✓ Parsed all responses")

        # Convert to DataFrame and save individual replica
        replica_df = pd.DataFrame(replica_results)
        output_file = f"{OUTPUT_DIR}/batch_vllm_{model_name.replace(':', '_')}_replica{replica_id}.csv"
        replica_df.to_csv(output_file, index=False)

        success_rate = replica_df['success'].mean() * 100
        print(f"Replica {replica_id}: {len(replica_df)} evaluations, {success_rate:.1f}% success")

        all_results.extend(replica_results)

        # Clean up GPU memory after each replica
        del llm
        del sampling_params
        import torch
        torch.cuda.empty_cache()
        print(f"GPU memory cleared after replica {replica_id}")

    # Combine all replicas
    combined_df = pd.DataFrame(all_results)
    combined_file = f"{OUTPUT_DIR}/batch_vllm_{model_name.replace(':', '_')}_all_replicas.csv"
    combined_df.to_csv(combined_file, index=False)
    print(f"\nCombined file saved: {combined_file}")

    return combined_df

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function."""
    # Parse command line arguments
    # Usage: python script.py [MODEL_NAME] [REPLICA_ID]
    target_model = None
    target_replica = None

    if len(sys.argv) > 1:
        target_model = sys.argv[1]
        if target_model not in MODEL_CONFIGS:
            print(f"ERROR: Model '{target_model}' not found in MODEL_CONFIGS")
            print(f"Available models: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            target_replica = int(sys.argv[2])
            if target_replica < 0 or target_replica >= N_REPLICAS:
                print(f"ERROR: Replica ID must be between 0 and {N_REPLICAS-1}")
                sys.exit(1)
        except ValueError:
            print(f"ERROR: Replica ID must be an integer")
            sys.exit(1)

    print("\n" + "="*80)
    print("STARTING VLLM DETERMINISTIC EVALUATION")
    print("="*80)
    if target_model:
        print(f"Target model: {target_model} (single model mode)")
    else:
        print(f"Running all models: {list(MODEL_CONFIGS.keys())}")
    if target_replica is not None:
        print(f"Target replica: {target_replica} (single replica mode - PROCESS ISOLATION)")
    print("="*80)

    # Load data
    df_answers = load_answers_sample(n_samples=N_SAMPLES)
    print(f"\nDataset info:")
    print(f"  Total responses: {len(df_answers)}")
    print(f"  Unique questions: {df_answers['question_id'].nunique()}")

    # Determine which models to run
    if target_model:
        models_to_run = {target_model: MODEL_CONFIGS[target_model]}
    else:
        models_to_run = MODEL_CONFIGS

    # Run evaluations for each model
    all_results = {}

    for model_name, config in models_to_run.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'='*80}")

        try:
            model_path = config['hf_path']
            results = run_sequential_replicas_vllm(
                df=df_answers,
                model_name=model_name,
                model_path=model_path,
                n_replicas=N_REPLICAS,
                single_replica_id=target_replica
            )
            all_results[model_name] = results
            if target_replica is not None:
                print(f"✓ Completed {model_name} - Replica {target_replica}")
            else:
                print(f"✓ Completed {model_name}")
        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final report
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models evaluated: {len(all_results)}")
    print(f"Total evaluations: {sum(len(df) for df in all_results.values())}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
