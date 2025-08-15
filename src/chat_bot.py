import os
from .docx_processor import DocxProcessor
from .embedding_manager import EmbeddingManager
from .gemini_integration import GeminiIntegration

class ChatBot:
    def __init__(self, docx_dir: str = "./docs"):
        self.docx_processor = DocxProcessor()
        self.embedding_manager = EmbeddingManager()
        self.gemini_integration = GeminiIntegration()
        self.docx_dir = docx_dir

        if not os.path.isdir(self.docx_dir):
            raise FileNotFoundError(f"The specified document directory does not exist: {self.docx_dir}")

        self._load_all_docx_documents()

    def _load_all_docx_documents(self):
        print(f"Loading documents from: {self.docx_dir}")
        for filename in os.listdir(self.docx_dir):
            if filename.endswith(".docx"):
                filepath = os.path.join(self.docx_dir, filename)
                print(f"Processing {filepath}")
                text = self.docx_processor.extract_text_from_docx(filepath)
                chunks = self.docx_processor.chunk_text(text)
                self.embedding_manager.add_embeddings(chunks)
        print("All documents processed and uploaded.")

    def get_response(self, query: str) -> str:
        relevant_chunks = self.embedding_manager.search_vector_store(query, k=3)
        context = "\n".join(relevant_chunks)

        if context:
            prompt = f"#Com base nas informações abaixo: \n{context}\n\n##Responda à seguinte pergunta: {query}"
        else:
            prompt = query

        return self.gemini_integration.get_gemini_response(prompt)

    def run_chat(self):
        print("Bem-vindo ao Bot Gemini DOCX! Digite 'sair' para encerrar.")
        while True:
            user_input = input("Você: ")
            if user_input.lower() == 'sair':
                print("Até mais!")
                break
            response = self.get_response(user_input)
            print(f"Bot: {response}")

    def is_ready(self) -> bool:
        return (
                self.embedding_manager.index is not None and
                self.embedding_manager.index.ntotal > 0
        )