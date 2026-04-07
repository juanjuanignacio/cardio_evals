#!/usr/bin/env python3
"""
Example Usage of RAGQA RAG System

This script demonstrates how to use the RAG system programmatically.
"""

import os
from dotenv import load_dotenv
import pymongo
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables
load_dotenv()

# Import RAG functions
from RAG_Mongodb_gemma2 import (
    get_mongo_client,
    get_search_result,
    create_rag_prompt,
    generate_answer,
    embedding_model
)


def main():
    """Example usage of the RAG system."""

    print("="*80)
    print("RAGQA RAG System - Example Usage")
    print("="*80)

    # 1. Setup MongoDB connection
    print("\n1. Connecting to MongoDB...")
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not set. Please configure .env file.")

    mongo_client = get_mongo_client(mongo_uri)
    db_name = os.getenv("MONGO_DB_NAME", "cardio_refs_2")
    collection_name = os.getenv("MONGO_COLLECTION_NAME", "abst_refs")

    db = mongo_client[db_name]
    collection = db[collection_name]
    print(f"✓ Connected to {db_name}.{collection_name}")

    # 2. Load language model
    print("\n2. Loading language model...")
    model_name = os.getenv("MODEL_NAME", "google/gemma-2-2b-it")
    device = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device if device == "cuda" else None
    )
    print(f"✓ Loaded {model_name} on {device}")

    # 3. Example queries
    example_queries = [
        "What are the behaviors of neutrophils?",
        "How does the immune system respond to infection?",
        "What is the role of T cells in immunity?"
    ]

    print("\n3. Running example queries...")
    print("="*80)

    for i, query in enumerate(example_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-"*80)

        # Retrieve relevant documents
        context = get_search_result(query, collection)

        # Create prompt
        prompt = create_rag_prompt(query, context)

        # Generate answer
        answer = generate_answer(prompt, model, tokenizer, device=device)

        # Extract just the answer part (after the model's response marker)
        if "<start_of_turn>model" in answer:
            answer_only = answer.split("<start_of_turn>model")[-1].strip()
        else:
            answer_only = answer

        print(f"\nAnswer:\n{answer_only}\n")
        print("-"*80)

    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
