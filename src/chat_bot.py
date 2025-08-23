import os
import glob
import logging
from typing import List
from .gemini_integration import GeminiIntegration
from .pinecone_manager import PineconeManager
from .docx_processor import DocxProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ChatBot:
    def __init__(self, docx_dir: str, google_api_key: str, namespace: str):
        self.pinecone_manager = PineconeManager(
            google_api_key=google_api_key,
            namespace=namespace
        )
        self.gemini_integration = GeminiIntegration()
        self.docx_processor = DocxProcessor()
        self.docx_dir = docx_dir
        self._documents_loaded = self.pinecone_manager.namespace_has_vectors()

        if not os.path.isdir(self.docx_dir):
            logger.warning(f"documents directory '{self.docx_dir}' not founded. Create it before processing documents.")

    def load_and_process_documents(self) -> bool:
        if not self.docx_dir or not os.path.exists(self.docx_dir):
            logger.error(f"directory '{self.docx_dir}' not founded.")
            return False

        docx_files = glob.glob(os.path.join(self.docx_dir, "*.docx"))

        if not docx_files:
            logger.warning(f"no file .docx founded in '{self.docx_dir}'")
            return False

        documents = []
        for docx_file in docx_files:
            try:
                logger.info(f"Processing file: {os.path.basename(docx_file)}")
                text = self.docx_processor.extract_text_from_docx(docx_file)
                if text.strip():
                    documents.append(text)
            except Exception as e:
                logger.error(f"Error to process {docx_file}: {e}")

        if documents:
            logger.info(f"Adding {len(documents)} documents to Pinecone...")
            self.pinecone_manager.add_documents(documents)
            self._documents_loaded = True
            return True
        else:
            logger.warning("No documents with valid content were found.")
            return False

    def clear_vector_store(self) -> bool:
        success = self.pinecone_manager.delete_all_vectors()
        if success:
            self._documents_loaded = False
        return success

    def is_ready(self) -> bool:
        return self._documents_loaded

    def get_response(self, query: str) -> str:
        if not self.is_ready():
            return "The documents have not been processed yet. Please initiate processing first."

        try:
            relevant_chunks = self.pinecone_manager.search_documents(query, k=5)
            if relevant_chunks:
                context = "\n\n".join(relevant_chunks)
                prompt = f"Com base no seguinte contexto, responda à pergunta do usuário: CONTEXTO: {context}\n\nPERGUNTA: {query}"
            else:
                prompt = f"Não encontrei informações nos documentos. Responda com base no seu conhecimento geral: {query}"

            return self.gemini_integration.get_gemini_response(prompt)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error processing message: {e}"