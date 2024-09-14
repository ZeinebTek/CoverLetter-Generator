from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# Constants
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-1.5-pro"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Load environment variables
load_dotenv()


def load_and_split_documents(cv_path, transcript_path, certificates_path):
    """Loads and splits the CV, transcripts, and certificates."""
    cv_loader = Docx2txtLoader(cv_path)
    transcript_loader = Docx2txtLoader(transcript_path)
    certificates_loader = Docx2txtLoader(certificates_path)
    cv_docs = cv_loader.load()
    transcript_docs = transcript_loader.load()
    certificates_docs = certificates_loader.load()
    all_docs = cv_docs + transcript_docs + certificates_docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(all_docs)
    return splits


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