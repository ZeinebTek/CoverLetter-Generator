from io import BytesIO
import os
import tempfile
from pdf_2_docx.pdf_2_docx_v2 import convert_pdf_to_docx
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# Constants
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-1.5-pro"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Load environment variables
load_dotenv()


def load_and_split_documents(cv_file, transcript_file=None, certificates_file=None):
    """Loads and splits the CV, transcripts, and certificates."""

    def ensure_docx_content(file):
        """If the file is a PDF, convert it to an in-memory DOCX; otherwise return the DOCX content."""
        if file.name.lower().endswith(".pdf"):
            return convert_pdf_to_docx(file.name, save=False)
        return file

    cv_docx = ensure_docx_content(cv_file)
    cv_docx = load_docx_from_memory(cv_docx) if isinstance(
        cv_docx, BytesIO) else load_docx_from_file(cv_docx)

    transcript_docx = ensure_docx_content(transcript_file)
    transcript_docx = load_docx_from_memory(transcript_docx) if isinstance(
        transcript_docx, BytesIO) else load_docx_from_file(transcript_docx)
    
    certificates_docx = ensure_docx_content(certificates_file)
    certificates_docx = load_docx_from_memory(certificates_docx) if isinstance(
        certificates_docx, BytesIO) else load_docx_from_file(certificates_docx)

    all_docs = cv_docx + transcript_docx + certificates_docx

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(all_docs)

    return splits


def load_docx_from_memory(docx_io):
    """Load DOCX content from a BytesIO object using LangChain's Docx2txtLoader."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
        temp_file.write(docx_io.getvalue())
        temp_file_path = temp_file.name

    loader = Docx2txtLoader(temp_file_path)
    docs = loader.load()
    os.remove(temp_file_path)

    return docs 


def load_docx_from_file(file_path):
    """Load DOCX content using LangChain's Docx2txtLoader."""
    loader = Docx2txtLoader(file_path)
    docs = loader.load()

    return docs


class GoogleEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using Google Generative AI."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed_documents(self, texts: Documents) -> Embeddings:
        embeddings = []
        for text in texts:
            embedding_result = genai.embed_content(
                model=self.embedding_model, content=text
            )
            embeddings.append(embedding_result["embedding"])
        return embeddings

    def embed_query(self, text: str) -> Embeddings:
        embedding_result = genai.embed_content(
            model=self.embedding_model, content=text
        )
        return embedding_result["embedding"]


def create_vectorstore(splits):
    """Creates a Chroma vectorstore from the document splits."""
    embedding_function = GoogleEmbeddingFunction(EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(splits, embedding=embedding_function)
    return vectorstore


def create_qa_chain(vectorstore, temperature=0.0):
    """Creates the RetrievalQA chain with the specified LLM and temperature."""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )
    return qa_chain


def generate_cover_letter(job_description, qa_chain):
    """Generates a cover letter based on the job description and context."""
    prompt_template = """
    You are a helpful AI assistant that generates cover letters based on a provided CV, transcripts, certificates and job description.

    Here's the job description:
    ```
    {job_description}
    ```

    Based on the provided CV and transcripts, write a complete cover letter tailored to this job description. 
    Highlight relevant skills and experiences from my background that align with the requirements and responsibilities mentioned in the job description.
    I am a highly motivated and detail-oriented individual, known for my rigorous approach to tasks and my ability to work independently with minimal supervision. I take pride in my strong work ethic, consistently delivering high-quality results. 
    I excel in communication, ensuring clarity and collaboration within teams, and thrive in environments that require both individual contribution and teamwork. 
    My adaptability and problem-solving skills allow me to tackle challenges efficiently, making me a reliable and self-sufficient team member who can lead or support projects with equal effectiveness.
    I want to make sure that the cover letter is well-structured, professional, and engaging to the reader to make them want to learn more about me as a candidate.
    """

    prompt = prompt_template.format(job_description=job_description)
    cover_letter = qa_chain.invoke(prompt)
    return cover_letter["result"]


"""
if __name__ == "__main__":
    print("Loading Done.")
    splits = load_and_split_documents(
        "path/to/your/cv.docx",
        "path/to/your/transcripts.docx",
        "path/to/your/certificates_ZeinebTekaya.docx",
    )
    print("Text Extracted.")
    print("Model Loading...")
    vectorstore = create_vectorstore(splits)
    qa_chain = create_qa_chain(temperature=1.0)  # Set desired temperature

    # Example job description
    job_description = "put your job description"

    cover_letter = generate_cover_letter(job_description, qa_chain)
    print(cover_letter)"""
