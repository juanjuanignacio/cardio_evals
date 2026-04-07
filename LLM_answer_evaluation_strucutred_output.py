from pymongo import MongoClient
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import pandas as pd
from pydantic import BaseModel, Field, validator
from typing import Literal
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os 
from dotenv import load_dotenv

load_dotenv()

# MongoDB configuration
MONGODB_HOST = os.getenv('MONGODB_HOST', 'localhost')
MONGODB_PORT = int(os.getenv('MONGODB_PORT', '27017'))
MONGODB_NAME = os.getenv('MONGODB_NAME', 'RAGQA')

MODEL_NAME_LIST = ["llama3.3", "llama3.1", "deepseek-r1:7b", "deepseek-r1:70b", "phi4", "qwen2.5:7b" ]

EVAL_COLLECTION_LIST = ["ia_llama3_3_70b_eval", "ia_eval", "ia_deepseek_eval", "ia_deepseek-r1_70b_eval", "ia_phi_4_eval", "ia_qwen_2_5_eval"]

EVAL_COLLECTION = ""


def get_db():
    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    return client[MONGODB_NAME]

class Eval(BaseModel):
    Accuracy: int = Field(description="Score from 0-7 for accuracy")
    Clarity: int = Field(description="Score from 0-7 for clarity")
    Completeness: int = Field(description="Score from 0-7 for completeness")
    Source: Literal['human', 'ai']
    Justification: str = Field(description="Detailed explanation of the evaluation")

    @validator('Accuracy', 'Clarity', 'Completeness')
    def validate_score_range(cls, v):
        if not (0 <= v <= 7):
            raise ValueError(f'Score must be between 0 and 7, got {v}')
        return v

def create_evaluation_chain(llm: ChatOllama):
    parser = PydanticOutputParser(pydantic_object=Eval)
    
    prompt_template = """
    Evaluate the following response and provide a structured evaluation in JSON format.
    Make sure to include ALL required fields exactly as specified:

    Evaluation criteria:
    1. Accuracy: Is the response accurate and factually correct? (MUST be between 0-7, no other values allowed)
    2. Clarity: Is the response clear and easy to understand? (MUST be between 0-7, no other values allowed)
    3. Completeness: Is the response complete and covers all aspects of the question? (MUST be between 0-7, no other values allowed)
    4. Source: MUST be exactly 'human' or 'ai' (no other values are valid)
    5. Justification: A text explaining your evaluation (this field is MANDATORY)

    IMPORTANT: All scores MUST be integers between 0 and 7 inclusive. Any other values will be rejected.

    **Question**: {question}
    **Response to evaluate**: {response}

    Please provide your evaluation following EXACTLY this format:
    {{
        "Accuracy": <integer from 0-7>,
        "Clarity": <integer from 0-7>,
        "Completeness": <integer from 0-7>,
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

def evaluate_response_with_ai(response: str, question: str, llm: ChatOllama, max_retries=30):
    chain = create_evaluation_chain(llm)
    
    retries = 0
    while retries < max_retries:
        try:
            # Invoke the chain with the input
            eval_result = chain.invoke({
                "question": question,
                "response": response
            })
            
            # Additional validation check (although Pydantic should catch this)
            scores = [eval_result.Accuracy, eval_result.Clarity, eval_result.Completeness]
            if not all(0 <= score <= 7 for score in scores):
                raise ValueError("One or more scores outside valid range (0-7)")
            
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
    db = get_db()
    ia_eval_collection = db[EVAL_COLLECTION]
    ia_eval_collection.insert_one(eval_doc)

def is_already_evaluated(question_id, answer_id):
    db = get_db()
    ia_eval_collection = db[EVAL_COLLECTION]
    return ia_eval_collection.find_one({"question_id": question_id, "original_answer_id": answer_id}) is not None

def evaluate_all_responses(df: pd.DataFrame, llm: ChatOllama):
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
    db = get_db()
    answers_collection = db["answers"]
    answers_data = list(answers_collection.find({}))
    return pd.DataFrame(answers_data)

def main():
    df = load_answers_data()
    for MODEL_NAME, EVAL_COLLEC in zip(MODEL_NAME_LIST, EVAL_COLLECTION_LIST):
        global EVAL_COLLECTION
        EVAL_COLLECTION = EVAL_COLLEC
        llm = ChatOllama(model=MODEL_NAME)
        evaluate_all_responses(df, llm)

if __name__ == "__main__":
    main()