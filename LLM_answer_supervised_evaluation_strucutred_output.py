"""
LLM Answer Evaluation System - Supervised Evaluation

This script evaluates LLM-generated answers using structured output validation.
It uses multiple LLM models to assess answer quality across multiple dimensions:
- Accuracy: Factual correctness
- Clarity: Ease of understanding
- Completeness: Coverage of all aspects

The evaluation results are stored in MongoDB for analysis.

Configuration via environment variables:
- MONGODB_HOST: MongoDB server address
- MONGODB_PORT: MongoDB port (default: 27017)
- MONGODB_NAME: Database name (default: RAGQA)
- OLLAMA_BASE_URL: Ollama API endpoint (optional)
"""

import os
import pandas as pd
from pymongo import MongoClient
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, validator
from typing import Literal
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# MongoDB configuration
MONGODB_HOST = os.getenv('MONGODB_HOST', 'localhost')
MONGODB_PORT = int(os.getenv('MONGODB_PORT', '27017'))
MONGODB_NAME = os.getenv('MONGODB_NAME', 'RAGQA')

# Model and collection configuration
MODEL_NAME_LIST = [
    "llama3.3",
    "llama3.1",
    "deepseek-r1:7b",
    "deepseek-r1:70b",
    "phi4",
    "qwen2.5:7b"
]

EVAL_COLLECTION_LIST = [
    "ia_llama3_3_70b_eval_supervised",
    "ia_eval_supervised",
    "ia_deepseek_eval_supervised",
    "ia_deepseek-r1_70b_eval_supervised",
    "ia_phi4_eval_supervised",
    "ia_qwen_2_5_eval_supervised"
]

# Current evaluation collection (set dynamically in main)
EVAL_COLLECTION = ""

def get_db():
    """
    Get MongoDB database connection.

    Returns:
        MongoDB database instance
    """
    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    return client[MONGODB_NAME]


class Eval(BaseModel):
    """
    Evaluation schema for structured LLM output.

    Attributes:
        Accuracy: Score from 1-7 for factual correctness
        Clarity: Score from 1-7 for readability and clarity
        Completeness: Score from 1-7 for completeness of answer
        Source: Origin of the answer ('human' or 'ai')
        Justification: Detailed explanation of the evaluation
    """
    Accuracy: int = Field(description="Score from 1-7 for accuracy")
    Clarity: int = Field(description="Score from 1-7 for clarity")
    Completeness: int = Field(description="Score from 1-7 for completeness")
    Source: Literal['human', 'ai']
    Justification: str = Field(description="Detailed explanation of the evaluation")

    @validator('Accuracy', 'Clarity', 'Completeness')
    def validate_score_range(cls, v):
        """Validate that scores are within 1-7 range."""
        if not (1 <= v <= 7):
            raise ValueError(f'Score must be between 1 and 7, got {v}')
        return v

def create_evaluation_chain(llm: ChatOllama):
    """
    Create a LangChain evaluation chain with structured output parsing.

    Args:
        llm: ChatOllama instance for evaluation

    Returns:
        Configured evaluation chain
    """
    parser = PydanticOutputParser(pydantic_object=Eval)
    
    prompt_template = """
    Evaluate the following response and provide a structured evaluation in JSON format.
    Make sure to include ALL required fields exactly as specified:

    Evaluation criteria:
    1. Accuracy: Is the response accurate and factually correct? (MUST be between 1-7, no other values allowed)
    2. Clarity: Is the response clear and easy to understand? (MUST be between 1-7, no other values allowed)
    3. Completeness: Is the response complete and covers all aspects of the question? (MUST be between 1-7, no other values allowed)
    4. Source: MUST be exactly 'human' or 'ai' (no other values are valid)
    5. Justification: A text explaining your evaluation (this field is MANDATORY)

    IMPORTANT: All scores MUST be integers between 1 and 7 inclusive. Any other values will be rejected.

    **Question**: {question}
    **Correct answer**: {answer}
    **Response to evaluate**: {response}

    Please provide your evaluation following EXACTLY this format:
    {{
        "Accuracy": <integer from 1-7>,
        "Clarity": <integer from 1-7>,
        "Completeness": <integer from 1-7>,
        "Source": <"human" or "ai">,
        "Justification": "<detailed explanation>"
    }}

    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "response"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = (
        prompt 
        | llm 
        | StrOutputParser() 
        | parser
    )
    
    return chain

def evaluate_response_with_ai(response: str, correct_answer: str, question: str, llm: ChatOllama, max_retries=30):
    """
    Evaluate a response using AI with retry logic for robustness.

    Args:
        response: The answer to evaluate
        correct_answer: The gold standard answer
        question: The original question
        llm: ChatOllama instance for evaluation
        max_retries: Maximum number of retry attempts (default: 30)

    Returns:
        Dictionary with evaluation scores and metadata
    """
    chain = create_evaluation_chain(llm)
    
    retries = 0
    while retries < max_retries:
        try:
            # Invoke the chain with the input
            eval_result = chain.invoke({
                "question": question,
                "answer": correct_answer,
                "response": response,

            })
            
            # Additional validation check (although Pydantic should catch this)
            scores = [eval_result.Accuracy, eval_result.Clarity, eval_result.Completeness]
            if not all(1 <= score <= 7 for score in scores):
                raise ValueError("One or more scores outside valid range (1-7)")
            
            # Create the evaluation document
            eval_doc = {
                "question": question,
                "response": response,
                "accuracy_score": eval_result.Accuracy,
                "clarity_score": eval_result.Clarity,
                "completeness_score": eval_result.Completeness,
                "source_guessed": eval_result.Source,
                "justification": eval_result.Justification,
                "evaluator": "AI"
            }
            
            return eval_doc
            
        except ValueError as ve:
            print(f"Validation error on attempt {retries + 1}: {ve}")
            retries += 1
        except Exception as e:
            print(f"Error on attempt {retries + 1}: {e}")
            retries += 1
    
    # Return default values if max retries exceeded
    return {
        "question": question,
        "response": response,
        "accuracy_score": None,
        "clarity_score": None,
        "completeness_score": None,
        "source_guessed": "unknown",
        "justification": "Could not obtain a valid evaluation after multiple attempts.",
        "evaluator": "AI"
    }

def save_evaluation_to_mongo(eval_doc):
    """
    Save evaluation document to MongoDB.

    Args:
        eval_doc: Dictionary with evaluation data
    """
    db = get_db()
    ia_eval_collection = db[EVAL_COLLECTION]
    ia_eval_collection.insert_one(eval_doc)


def is_already_evaluated(question_id, answer_id):
    """
    Check if a response has already been evaluated.

    Args:
        question_id: ID of the question
        answer_id: ID of the answer

    Returns:
        Boolean indicating if evaluation exists
    """
    db = get_db()
    ia_eval_collection = db[EVAL_COLLECTION]
    return ia_eval_collection.find_one({
        "question_id": question_id,
        "original_answer_id": answer_id
    }) is not None


def evaluate_all_responses(df: pd.DataFrame, llm: ChatOllama):
    """
    Evaluate all responses in the dataframe.

    Args:
        df: DataFrame with questions and answers
        llm: ChatOllama instance for evaluation
    """
    for _, row in df.iterrows():
        question_id = row["question_id"]
        answer_id = str(row["_id"])

        if is_already_evaluated(question_id, answer_id):
            print(f"Response with ID {answer_id} for question ID {question_id} has already been evaluated. Skipping...")
            continue

        if 'bibliography_sources' in row and row['bibliography_sources']:
            if str(row['bibliography_sources']) != 'nan':
                if str(row['bibliography_sources']).strip():
                    row['answer'] = f"{row['answer']}\n\nReferences:\n{row['bibliography_sources']}"
        
        eval_doc = evaluate_response_with_ai(
            response=row["answer"],
            correct_answer=row["correct_answer"],
            question=row["question_text"],
            llm=llm
        )

        is_llm = 1
        if row['user'] != "AI" and row['user'] != "Deepseek":
            is_llm=0

        if row['user'] == "Deepseek":
            is_llm=2
            
        eval_doc.update({
            "original_user_email": row["user"],
            "original_user_name": row["full_name"],
            "original_user_gender": row["gender"],
            "original_user_professional_title": row["professional_title"],
            "original_user_main_specialty": row["main_specialty"],
            "original_user_years_of_experience": row["years_of_experience"],
            "original_user_institution": row["current_institution"],
            "original_user_country": row["country_of_practice"],
            "question_id": question_id,
            "original_answer_id": answer_id,
            "source_doc": row["source_doc"],
            "groundedness_score": row["groundedness_score"],
            "relevance_score": row["relevance_score"],
            "standalone_score": row["standalone_score"],
            'main_specialty': row["main_specialty"],
            "LLM": is_llm,
        })

        save_evaluation_to_mongo(eval_doc)

def load_answers_data():
    """
    Load answers data from MongoDB.

    Returns:
        DataFrame with all answers
    """
    db = get_db()
    answers_collection = db["answers"]
    answers_data = list(answers_collection.find({}))
    return pd.DataFrame(answers_data)


def load_questions_data():
    """
    Load questions data from MongoDB.

    Returns:
        DataFrame with all questions
    """
    db = get_db()
    questions_collection = db["questions"]
    questions_data = list(questions_collection.find({}))
    return pd.DataFrame(questions_data)


def main():
    """Main execution function."""
    print("="*80)
    print("LLM Answer Evaluation System - Supervised Evaluation")
    print("="*80)
    print(f"MongoDB: {MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_NAME}")
    print(f"Models to evaluate: {len(MODEL_NAME_LIST)}")
    print("="*80)

    # Load data
    print("\nLoading data from MongoDB...")
    df = load_answers_data()
    df_questions = load_questions_data()

    # Prepare questions dataframe
    df_questions2 = pd.DataFrame()
    df_questions2["correct_answer"] = df_questions["answer"]
    df_questions2["id"] = df_questions["_id"].astype(str)

    # Merge questions with answers
    df = pd.merge(df_questions2, df, left_on="id", right_on="question_id", how="right")
    print(f"✓ Loaded {len(df)} answer-question pairs")

    # Evaluate with each model
    for MODEL_NAME, EVAL_COLLEC in zip(MODEL_NAME_LIST, EVAL_COLLECTION_LIST):
        print(f"\n{'='*80}")
        print(f"Evaluating with model: {MODEL_NAME}")
        print(f"Target collection: {EVAL_COLLEC}")
        print(f"{'='*80}")

        global EVAL_COLLECTION
        EVAL_COLLECTION = EVAL_COLLEC

        # Initialize LLM
        ollama_base_url = os.getenv('OLLAMA_BASE_URL')
        if ollama_base_url:
            llm = ChatOllama(model=MODEL_NAME, base_url=ollama_base_url)
        else:
            llm = ChatOllama(model=MODEL_NAME)

        # Run evaluation
        evaluate_all_responses(df, llm)
        print(f"✓ Completed evaluation with {MODEL_NAME}")

    print("\n" + "="*80)
    print("All evaluations completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()