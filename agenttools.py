import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool, tool

# Load credentials from environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPA_SERVICE_ROLE_KEY")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# Initialize the open AI client
open_ai_client = OpenAI(api_key=OPENAI_API_KEY)


#! tool declaration contains the same functions as the main script which can be used in testing
@tool
def search_catechism(input_text: str) -> str:
    """
    This tool vectorizes a user input and searches through a database to find semantically relevant chunks from the Catechsim of the Catholic Church
    """
    
    def improve_question(text: str) -> str:
        """
        let's rewrite the input using an LLM that will think with RAG in mind to improve our semantic search results.
        """
        try:
            llm = ChatOpenAI(model="o3-mini", temperature=1)
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
        # print(f"DEBUG WHAT ARE RESULTS FOR FORMATTING: {results}")
        formatted_results = {}
        formatted_results["rag_context"] = results[0]["chunk"]
        formatted_results["url"] = results[0]["url"]
        # dictionary = {"id": id, "chunk": chunk, "source_file": source, "distance": distance}
        return formatted_results
    
    def answer_question(llm_question: str, rag_context: dict) -> str:
        """
        rerun the LLM with the context from RAG to answer the question
        """
        try:
            context = rag_context["rag_context"]

            llm = ChatOpenAI(model="o3-mini", temperature=1)
            prompt_template = PromptTemplate(
                input_variables=['llm_question','rag_context'],
                template="""
                Answer the query utilizing the Catechism context retrieved from the database.
                Your answer should be written in common language so that general audiences can understand while maintaining the semantic meaning of the context.
                Do NOT add any information that you do not find in the provided context.
                You response should only be based on the context and should not include presumptions."
                User Query: {llm_question}
                Company Docs Context: {rag_context}
                Output:
                """
            )
            prompt = prompt_template.format(llm_question=llm_question, rag_context=context)
            response = llm.invoke(prompt)
            # print(f"\n\nDEBUG LLM REFORMATED Q: {response.content}\n\n")
            return response.content
        except Exception as e:
            print(f"Error answering the question with the llm: {e}")
            return []
    
    llm_question = improve_question(input_text)
    if not llm_question:
        print("Failed to generate an LLM version of the question.")
        return
    
    # Vectorize the user query
    query_embedding = get_embedding(llm_question)
    if not query_embedding:
        print("Failed to generate query embedding.")
        return
    
    # Perform semantic search against the vectors in Supabase
    results = semantic_search(query_embedding, top_k=3)
    if not results:
        print("No relevant results found.")
        return
    
    # Format the results as required
    formatted = format_search_results(results)

    answer = answer_question(llm_question, formatted)

    return answer
