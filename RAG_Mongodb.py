"""
RAG System with MongoDB Vector Search and Gemma-2 Model

This script implements a Retrieval-Augmented Generation (RAG) system that:
1. Connects to MongoDB with vector search capabilities
2. Performs semantic search using sentence embeddings
3. Generates answers using the Gemma-2 language model

Requirements:
- MongoDB instance with vector search index configured
- CardioRefs dataset with embeddings
- GPU for model inference (recommended)
"""

import os
import pymongo
from pymongo.operations import SearchIndexModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Initialize embedding model
# Using thenlper/gte-large for high-quality embeddings (1024 dimensions)
embedding_model = SentenceTransformer("thenlper/gte-large")

def get_embedding(text: str) -> list[float]:
    """
    Generate embedding vector for input text.

    Args:
        text: Input text to embed

    Returns:
        List of floats representing the embedding vector (1024 dimensions)
    """
    if not text.strip():
        print("Warning: Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)
    return embedding.tolist()


def get_mongo_client(mongo_uri):
    """
    Establish connection to MongoDB.

    Args:
        mongo_uri: MongoDB connection string

    Returns:
        MongoDB client instance or None if connection fails
    """
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None


# MongoDB configuration
# Set MONGO_URI environment variable with your connection string
# Example: export MONGO_URI="mongodb://localhost:27017/?directConnection=true"
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/?directConnection=true")
if not mongo_uri:
    raise ValueError("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

# Database and collection setup
DB_NAME = os.getenv("MONGO_DB_NAME", "cardio_refs_2")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "abst_refs")

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# Vector search index specification
# Note: This index should be created once during database setup
# Uncomment the code below if you need to create the index
"""
index_spec = SearchIndexModel(
    definition={
        "fields": [{
            "numDimensions": 1024,      # Embedding dimensions (gte-large)
            "path": "embedding",        # Field path for embedding vectors
            "similarity": "cosine",     # Similarity metric
            "type": "vector"            # Vector index type
            },
        ]
    },
    name="vector_index",
    type="vectorSearch",
)

# Create the index (run once)
result = collection.create_search_index(model=index_spec)
print(f"Vector search index created: {result}")
"""

def vector_search(user_query, collection, num_candidates=150, limit=4):
    """
    Perform vector similarity search in MongoDB collection.

    Args:
        user_query: The user's query string
        collection: MongoDB collection to search
        num_candidates: Number of candidate matches to consider (default: 150)
        limit: Maximum number of results to return (default: 4)

    Returns:
        List of matching documents with abstracts, references, and similarity scores
    """
    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if not query_embedding:
        print("Error: Invalid query or embedding generation failed")
        return []

    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": num_candidates,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "Abstract": 1,
                "Reference": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result(query, collection):
    """
    Format vector search results as a string.

    Args:
        query: User query string
        collection: MongoDB collection to search

    Returns:
        Formatted string with abstracts and references
    """
    search_results = vector_search(query, collection)

    formatted_result = ""
    for result in search_results:
        abstract = result.get('Abstract', 'N/A')
        reference = result.get('Reference', 'N/A')
        formatted_result += f"Abstract: {abstract}, Reference: {reference}\n"

    return formatted_result


def create_rag_prompt(query, context):
    """
    Create a RAG prompt for the LLM with retrieved context.

    Args:
        query: User's question
        context: Retrieved context from vector search

    Returns:
        Formatted prompt string
    """
    prompt = f"""<start_of_turn>user Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the question is to request references, please only return the source references with no answer.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
3. If you find the answer, write the answer in a concise way and add the list of sources that are **directly** used to derive the answer. Exclude the references that are irrelevant to the final answer.

Context: {context}

Query: {query}
<end_of_turn>
<start_of_turn>model """
    return prompt


def generate_answer(prompt, model, tokenizer, max_new_tokens=1000, device="cuda"):
    """
    Generate answer using the language model.

    Args:
        prompt: Formatted input prompt
        model: Language model instance
        tokenizer: Tokenizer instance
        max_new_tokens: Maximum tokens to generate (default: 1000)
        device: Device to use for inference (default: "cuda")

    Returns:
        Generated text answer
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = inputs.to(device)

    # Generate response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and return
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded


def main():
    """
    Main function to demonstrate RAG system usage.
    """
    # Load model and tokenizer
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-2b-it")
    DEVICE = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE if DEVICE == "cuda" else None
    )
    print("Model loaded successfully")

    # Example query
    query = "behaviours of neutrophils"
    print(f"\nQuery: {query}")

    # Retrieve relevant documents
    print("Retrieving relevant documents...")
    context = get_search_result(query, collection)

    # Create prompt
    prompt = create_rag_prompt(query, context)

    # Generate answer
    print("Generating answer...")
    answer = generate_answer(prompt, model, tokenizer, device=DEVICE)

    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(answer)
    print("="*80)


if __name__ == "__main__":
    main()