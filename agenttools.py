import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool, tool

import time

# Load credentials from environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Initialize the open AI client
open_ai_client = OpenAI(api_key=OPENAI_API_KEY)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

#? trying to define a class to structure the output more for the response
class CatechismOutput(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    citations: str = Field(description="The citations of where in the Catechism the answers came from")

structured_llm_answerer = llm.with_structured_output(CatechismOutput)


#! tool declaration contains the same functions as the main script which can be used in testing
@tool
def search_catechism(input_text: str) -> str:
    """
    This tool takes the entire user's input and performs a semantic search to find relevant information from the Catechism to answer the question.
    """
    print("\nSearching the Catechism...\n")
    
    def improve_question(text: str) -> str:
        """
        let's rewrite the input using an LLM that will think with RAG in mind to improve our semantic search results.
        """
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
            prompt_template = PromptTemplate(
                input_variables=['input_text'],
                template="""
                Improve the user's query by rewording it as you see fit to improve the chances that it returns relevant results when doing a vector semantic search.
                Do not alter what the user is asking semantically, but adjust the language as necessary to format the question better.
                Text: {input_text}
                Output:
                """
            )
            prompt = prompt_template.format(input_text=text)
            response = llm.invoke(prompt)
            # print(f"\n\nDEBUG LLM REFORMATED Q: {response.content}\n\n")
            return response.content
        except Exception as e:
            print(f"Error generating improved query with LLM: {e}")
            return []



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

    def format_search_results(results: list) -> dict:
        """
        format the results into a dictionary to be added in as a config to the agent when this tool is called
        """
        try:
            #print(f"\nFormatting the results of the query response...\n")
            #print(f"\nDEBUG WHAT ARE RESULTS FOR FORMATTING: {results}\n")
            formatted_results = {}
            formatted_results["rag_context"] = results[0]["chunk"] + results[1]["chunk"]
            # dictionary = {"id": id, "chunk": chunk, "distance": distance}
            return formatted_results
        except Exception as e:
            print(f"Exception while formatting the results: {e}")
            return None
    
    def answer_question(llm_question: str, rag_context: dict) -> str:
        """
        rerun the LLM with the context from RAG to answer the question
        """
        try:
            context = rag_context["rag_context"]

            prompt_template = PromptTemplate(
                input_variables=['llm_question','rag_context'],
                template="""
                Answer the query utilizing the Catechism context retrieved from the database.
                Your answer should be written in common language at a 7th grade reading level so that general audiences can understand while maintaining the semantic meaning of the context.
                Do NOT add any information that you do not find in the provided context.
                Your response should only be based on the context and should not include presumptions.
                Your responses to questions should include both an explanation of what the Catechism says while using direct quotes and the citations to support your explanation."
                User Query: {llm_question}
                Catechism: {rag_context}
                Output:
                """
            )
            prompt = prompt_template.format(llm_question=llm_question, rag_context=context)
            answer = structured_llm_answerer.invoke(prompt)
            #response = structured_llm_answerer.invoke(answer.content)
            # print(f"\n\nDEBUG LLM REFORMATED Q: {response.content}\n\n")
            return answer
        except Exception as e:
            print(f"Error answering the question with the llm: {e}")
            return []
    
    """llm_reword_start = time.time()
    llm_question = improve_question(input_text)
    llm_reword_end = time.time()
    if not llm_question:
        print("Failed to generate an LLM version of the question.")
        return"""
    
    embed_query_start = time.time()
    # Vectorize the user query
    query_embedding = get_embedding(input_text)
    embed_query_end = time.time()
    if not query_embedding:
        print("Failed to generate query embedding.")
        return
    
    semantic_search_start = time.time()
    # Perform semantic search against the vectors in Supabase
    results = semantic_search(query_embedding, top_k=3)
    semantic_search_end = time.time()
    if not results:
        print("No relevant results found.")
        return
    
    # Format the results as required
    format_results_start = time.time()
    formatted = format_search_results(results)
    format_results_end = time.time()
    if not formatted:
        print("Failed to format the search results of the query.")
        return
    #! this takes anywhere from 10 seconds to 30 -> this was due to using the o3-mini model, using the 4o is much quicker
    answer_question_start = time.time()
    answer = answer_question(input_text, formatted)
    answer_question_end = time.time()
    #print(f"\n\nSTRUCTURED OUTPUT of TOOL: {answer}")
    if not answer:
        print("Failed to formulate an answer to the question.")
        return
    #print(f"\nTime for LLM reword of Q = {llm_reword_end - llm_reword_start:.2f}")
    print(f"Time to Embed the Query = {embed_query_end - embed_query_start:.2f}")
    print(f"Time to Semantic Search = {semantic_search_end - semantic_search_start:.2f}")
    print(f"Time to format results = {format_results_end - format_results_start:.2f}")
    print(f"Time to Answer the Q = {answer_question_end - answer_question_start:.2f}")
    return answer
