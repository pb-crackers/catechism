"""
Testing the same functionality with DSPy as opposed to LangGraph
"""

import dspy
from dotenv import load_dotenv
import os

from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Initialize the open AI client
open_ai_client = OpenAI(api_key=OPENAI_API_KEY)

#! DSPy
lm = dspy.LM('openai/o3-mini', temperature=1.0, max_tokens=5000)
dspy.configure(lm=lm)

#* define the signature 👇
qa = dspy.ChainOfThought("context: list[str], question: str -> answer: str")
response = qa(question="here is my question", context=["here is my context"])

#* define our functions in order:
#* input > vectorize it > format that for Supabase query > query Supabase for semantic data > provide that + the Q to the model for answering

def get_embedding(text: str) -> list:
        """
        Generate a vector embedding for the given text using OpenAI's model.
        """
        try:
            response = open_ai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
        
def format_vector_literal(embedding: list) -> str:
    # Convert the list of floats into a string like "[0.1, 0.2, ...]"
    return "[" + ", ".join(map(str, embedding)) + "]"

def semantic_search(query_embedding: list, top_k: int = 3) -> list:
        """
        Perform a semantic search against the 'vectors' table by calling
        the Supabase stored procedure 'match_vectors'. This procedure compares
        the query embedding with stored embeddings and returns the top results.
        
        Make sure that you have created the stored procedure in Supabase.
        """
        try:
            vector_literal = format_vector_literal(query_embedding)

            response = supabase.rpc(
                fn="match_vectors", 
                params={"match_count": top_k, "query_embedding": vector_literal}
            ).execute()
            
            # print(f"\n\nDEBUG RESPONSE: {response}\n\n")
            
            return response.data
        except Exception as e:
            print("Exception during semantic search:", e)
            return []




def main():
    print("-" * 50)
    print("Interactive Mode: Press Q to Quit")
    
    while True:
        question = input("\nUser: ")
        if question.lower() in ['q', 'quit', 'exit', 'e']:
             print("Quitting...")
             break
        else:
            q_embedding = get_embedding(question)
            search_results = semantic_search(q_embedding)
            list_of_results = []
            for result in search_results:
                 list_of_results.append(result['chunk'])
            ai_thinking = qa(question=question, context=list_of_results)
            print(f"\nREASONING: {ai_thinking.reasoning}")
            answer = ai_thinking.answer
            print(f"\nAI: {answer}")

if __name__ == "__main__":
    main()