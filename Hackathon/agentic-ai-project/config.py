import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Google API key for Gemini LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set. Please set it in the environment or .env file.")
    raise ValueError("GOOGLE_API_KEY is required for Gemini LLM authentication.")

# FAISS vector store path
VECTORSTORE_PATH = "vectorstore/faiss_index"

# HuggingFace embedding model for RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Gemini model name (valid options: gemini-1.5-pro, gemini-1.5-flash, gemini-pro)
GEMINI_MODEL = "gemini-2.0-flash"