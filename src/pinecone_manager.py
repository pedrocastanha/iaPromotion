import os
import uuid
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PineconeManager:
    def __init__(self, google_api_key: str, namespace: str):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")

        if not api_key or not index_name:
            raise ValueError("PINECONE_API_KEY e PINECONE_INDEX_NAME must be defined at .env")

        if not namespace:
            raise ValueError("namespace cannot be empty.")
        self.namespace = namespace

        self.pinecone = Pinecone(api_key=api_key)
        self.index_name = index_name

        if not google_api_key:
            raise ValueError("API key from google was not received by PineconeManager.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=60,
        )

        self._create_index_if_not_exists()
        self.index = self.pinecone.Index(self.index_name)

    def namespace_has_vectors(self) -> bool:
        try:
            stats = self.index.describe_index_stats()

            if self.namespace in stats.namespaces and stats.namespaces[self.namespace].vector_count > 0:
                logger.info(
                    f"Namespace '{self.namespace}' already contains  {stats.namespaces[self.namespace].vector_count} vectors.")
                return True
            else:
                logger.info(f"Namespace '{self.namespace}' is empty or does not exist in the statistics.")
                return False
        except Exception as e:
            logger.error(f"Error checking namespace statistics '{self.namespace}': {e}")
            return False

    def _create_index_if_not_exists(self):
        list_of_index_objects = self.pinecone.list_indexes()
        existing_index_names = [index.name for index in list_of_index_objects]

        if self.index_name not in existing_index_names:
            logger.info(f"creating index {self.index_name}...")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws',region='us-east-1')
            )
            logger.info(f"Index {self.index_name} successfully created.")
        else:
            logger.info(f"Using existing index: {self.index_name}")

    def add_documents(self, documents: list[str]):
        if not documents:
            logger.info("No document to add.")
            return

        logger.info("Splitting documents in chunks...")
        all_chunks = self.text_splitter.split_text("\n\n".join(documents))

        if not all_chunks:
            logger.info("No chunk generated.")
            return

        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        vectors_to_upsert = []
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            try:
                embeddings_batch = self.embeddings.embed_documents(batch)
                for chunk, embedding in zip(batch, embeddings_batch):
                    vectors_to_upsert.append({
                        "id": str(uuid.uuid4()),
                        "values": embedding,
                        "metadata": {"text": chunk}
                    })
                logger.info(f"batch {i//batch_size + 1} of embeddings sucessfully generated.")
            except Exception as e:
                logger.error(f"Error to process lote {i // batch_size + 1}: {e}")
                continue

        if vectors_to_upsert:
            try:
                logger.info(f"Uploading {len(vectors_to_upsert)} vectors to namespace '{self.namespace}'...")
                self.index.upsert(vectors=vectors_to_upsert, namespace=self.namespace, batch_size=100)
                logger.info("Upload to Pinecone completed successfully.")
            except Exception as e:
                logger.error(f"Error uploading to Pinecone: {e}")

    def search_documents(self, query: str, k: int = 5) -> list[str]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                namespace=self.namespace
            )
            return [match.metadata['text'] for match in results.matches if 'text' in match.metadata]
        except Exception as e:
            logger.error(f"Error during namespace lookup '{self.namespace}': {e}")
            return []

    def get_index_stats(self):
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error when obtaining statistics of index: {e}")
            return None

    def delete_all_vectors(self) -> bool:
        try:
            logger.info(f"Starting namespace cleanup: '{self.namespace}'...")
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.info(f"Namespace '{self.namespace}' successfully cleaned.")
            return True
        except Exception as e:
            logger.error(f"error while trying to clear namespace '{self.namespace}': {e}")
            return False