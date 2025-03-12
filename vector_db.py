import os
from dotenv import load_dotenv
import uuid
from openai import OpenAI
import PyPDF2
from PIL import Image
import pytesseract
from supabase import create_client, Client
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Load credentials from environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
CATECHISM_PATH = os.getenv("CATECHISM_PATH")

# Initialize the Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

open_ai_client = OpenAI(api_key=OPENAI_API_KEY)

def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

class SemanticChunkerWithMax(SemanticChunker):
    def __init__(self, max_tokens: int, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens

    def split_text(self, text: str) -> list:
        chunks = super().split_text(text)
        final_chunks = []
        # Initialize a recursive splitter (or use any splitter you prefer)
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,  # Adjust based on token-to-character ratio
            chunk_overlap=100,
            separators=["\n\n", "\n", r"(?<=[.?!])\s+"],
            is_separator_regex=True
        )
        for chunk in chunks:
            if count_tokens(chunk) > self.max_tokens:
                # Further split this chunk
                sub_chunks = recursive_splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        return final_chunks

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

def extract_text_from_png(png_path: str) -> str:
    """Extract text from a PNG image using OCR (pytesseract)."""
    image = Image.open(png_path)
    text = pytesseract.image_to_string(image)
    return text

def chunk_text(text: str) -> list:
    text_splitter = SemanticChunkerWithMax(
        embeddings=OpenAIEmbeddings(),
        breakpoint_threshold_type="gradient",
        min_chunk_size=200,
        max_tokens=8000  # set your desired token limit below the model's max
    )
    docs = text_splitter.create_documents([text])
    return [doc.page_content for doc in docs]


def get_embedding(text: str) -> list:
    """Get the vector embedding for a given text chunk using OpenAI's model."""
    try:
        response = open_ai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        # embedding = response['data'][0]['embedding']
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

def store_vector(chunk_id: str, chunk_text: str, embedding: list):
    """
    Store the chunk data in Supabase.
    Assumes a table 'vectors' exists with columns: 
    - id (string)
    - chunk (text)
    - embedding (json/array)
    """
    data = {
        "id": chunk_id,
        "chunk": chunk_text,
        "vector": embedding,
    }
    result = supabase.table("vectors").insert(data).execute()
    if result.data[0].get("error"):
        print(f"Error storing chunk {chunk_id}: {result.get('error')}")
    else:
        print(f"Successfully stored chunk {chunk_id}.")

def process_file(file_path: str):
    """
    Process a file: extract text, chunk it, vectorize each chunk, and store in the database.
    Supports PDFs and PNGs.
    """
    file_path_lower = file_path.lower()
    if file_path_lower.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path_lower.endswith('.png'):
        text = extract_text_from_png(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return

    if not text.strip():
        print(f"No text extracted from {file_path}")
        return

    # Get only the file name for storage purposes
    # source_file = os.path.basename(file_path)
    chunks = chunk_text(text)
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        embedding = get_embedding(chunk)
        if embedding:
            print(f"Storing chunk {chunk_id} in DB.")
            store_vector(chunk_id, chunk, embedding)
        else:
            print(f"Skipping chunk {chunk_id} due to embedding error.")

if __name__ == "__main__":
    # Example list of file paths to process; update these paths as needed.
    
    catechism_path = CATECHISM_PATH
    print(f"Processing {catechism_path}...")
    process_file(catechism_path)
    