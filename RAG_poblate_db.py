"""
MongoDB Database Population Script

This script populates a MongoDB collection with document embeddings from a CSV file.
It processes the data in chunks to efficiently handle large datasets.

Requirements:
- CSV file with pre-computed embeddings
- MongoDB instance with appropriate access

Configuration via environment variables:
- MONGO_URI: MongoDB connection string
- MONGO_DB_NAME: Database name (default: cardio_refs_2)
- MONGO_COLLECTION_NAME: Collection name (default: abst_refs)
- CSV_FILE_PATH: Path to CSV file with embeddings
- CHUNK_SIZE: Number of rows to process at once (default: 5000)
"""

import os
import sys
import ast
import time
import pandas as pd
import pymongo
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer

# Configuration - Load from environment variables
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5000"))
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable must be set")

CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "cardioRef_embbeding_keywords.csv")
DB_NAME = os.getenv("MONGO_DB_NAME", "cardio_refs_2")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "abst_refs") 

def get_mongo_client(mongo_uri):
    """
    Establish connection to MongoDB.

    Args:
        mongo_uri: MongoDB connection string

    Returns:
        MongoDB client instance or None if connection fails
    """
    try:
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✓ Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"✗ Connection failed: {e}")
        return None


def main():
    """Main execution function."""
    print("="*80)
    print("MongoDB Database Population Script")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Database: {DB_NAME}")
    print(f"  - Collection: {COLLECTION_NAME}")
    print(f"  - CSV file: {CSV_FILE_PATH}")
    print(f"  - Chunk size: {CHUNK_SIZE}")
    print("="*80)

    # 1. Connect to database
    print("\n1. Connecting to MongoDB...")
    mongo_client = get_mongo_client(MONGO_URI)
    if mongo_client is None:
        sys.exit("Exiting script due to MongoDB connection failure.")

    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # 2. Clean previous collection if exists
    print(f"\n2. Checking existing collection...")
    if COLLECTION_NAME in db.list_collection_names():
        print(f"⚠ Collection '{COLLECTION_NAME}' already exists. Deleting it to re-ingest data.")
        collection.drop()
    else:
        print(f"✓ Collection '{COLLECTION_NAME}' does not exist. It will be created.")

    # 3. Read and insert data in chunks
    print(f"\n3. Starting data ingestion from '{CSV_FILE_PATH}'...")
    print(f"   Processing in chunks of {CHUNK_SIZE} rows")
    total_documents_inserted = 0

    try:
        # Process CSV in chunks to handle large files efficiently
        for i, chunk_df in enumerate(pd.read_csv(CSV_FILE_PATH, chunksize=CHUNK_SIZE)):
            # Convert embedding strings to lists
            chunk_df['embedding'] = chunk_df['embedding'].apply(ast.literal_eval)
            # Filter out empty embeddings
            chunk_df = chunk_df[chunk_df['embedding'].apply(len) > 0]

            # Insert chunk if not empty
            if not chunk_df.empty:
                documents = chunk_df.to_dict('records')
                collection.insert_many(documents)
                total_documents_inserted += len(documents)
                print(f"  ✓ Inserted chunk {i+1}, total documents so far: {total_documents_inserted}")

    except FileNotFoundError:
        sys.exit(f"✗ Error: File '{CSV_FILE_PATH}' not found.")
    except Exception as e:
        sys.exit(f"✗ An error occurred while processing the CSV file: {e}")

    print(f"\n✓ Data ingestion completed. Total: {total_documents_inserted} documents inserted.")

    # 4. Create vector search index
    print(f"\n4. Creating vector search index...")
    index_spec = SearchIndexModel(
        definition={
            "fields": [{
                "numDimensions": 1024,      # Embedding dimensions (gte-large)
                "path": "embedding",        # Field path for vectors
                "similarity": "cosine",     # Similarity metric
                "type": "vector"            # Vector index type
                },
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )

    try:
        result = collection.create_search_index(model=index_spec)
        print(f"✓ Index creation command sent. Result: {result}")
        print("  Note: Index building may take a few minutes in the background.")
    except Exception as e:
        print(f"⚠ Warning: Could not create index: {e}")
        print("  Index may already exist or you may not have permissions.")

    # 5. Optional: Test the setup with a sample query
    test_query = os.getenv("TEST_QUERY")
    if test_query:
        print(f"\n5. Running test query: '{test_query}'")
        print("   Waiting 15 seconds for index to build...")
        time.sleep(15)

        # Load embedding model for testing
        embedding_model = SentenceTransformer("thenlper/gte-large")

        def get_embedding(text: str) -> list[float]:
            if not text.strip():
                return []
            embedding = embedding_model.encode(text)
            return embedding.tolist()

        def vector_search(user_query, collection):
            query_embedding = get_embedding(user_query)
            if not query_embedding:
                return []
            pipeline = [
                {"$vectorSearch": {"index": "vector_index", "queryVector": query_embedding,
                                 "path": "embedding", "numCandidates": 150, "limit": 4}},
                {"$project": {"_id": 0, "Abstract": 1, "Reference": 1,
                            "score": {"$meta": "vectorSearchScore"}}}
            ]
            results = collection.aggregate(pipeline)
            return list(results)

        results = vector_search(test_query, collection)
        print(f"✓ Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Score: {result.get('score', 'N/A'):.4f}")
            print(f"    Abstract: {result.get('Abstract', 'N/A')[:100]}...")
            print(f"    Reference: {result.get('Reference', 'N/A')}")

    print("\n" + "="*80)
    print("Database population completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
