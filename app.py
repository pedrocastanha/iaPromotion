import os
import logging
from fastapi import FastAPI, HTTPException, Request
from src.chat_bot import ChatBot
from dotenv import load_dotenv
from src.schemas import ChatRequest, ChatResponse

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = FastAPI()
DOCUMENT_NAMESPACE = os.getenv("DOCUMENT_NAMESPACE")

@app.on_event("startup")
async def startup_event():
    project_root = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(project_root, 'docs')

    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logger.critical("GOOGLE_API_KEY was not found in the file .env.")
            raise ValueError("GOOGLE_API_KEY was not configured.")

        logger.info(f"Starting ChatBot to namespace: '{DOCUMENT_NAMESPACE}'")

        app.state.chatbot = ChatBot(
            docx_dir=docs_path,
            google_api_key=google_api_key,
            namespace=DOCUMENT_NAMESPACE
        )

        logger.info("Chatbot service initialized and ready.")

    except Exception as e:
        logger.critical(f"A critical error occurred during chatbot initialization: {e}", exc_info=True)
        app.state.chatbot = None


@app.post("/chat", response_model=ChatResponse)
async def chat_route(request: Request, request_body: ChatRequest):
    bot_instance = request.app.state.chatbot
    if bot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot service is unavailable.")

    if not bot_instance.is_ready():
        raise HTTPException(status_code=400,
                            detail="Documents were not processed. Please, call /process-documents first.")

    user_message = request_body.message
    company_name = request_body.company_name
    company_type = request_body.company_type

    if not user_message or not user_message.strip():
        raise HTTPException(status_code=400, detail="The 'message' field cannot be empty.")

    try:
        bot_response = bot_instance.get_response(
            query=user_message,
            company_name=company_name,
            company_type=company_type
        )
        return bot_response
    except Exception as e:
        logger.exception("An error occurred during get_response")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.post("/process-documents")
async def process_documents(request: Request):
    bot_instance = request.app.state.chatbot
    if bot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot service is unavailable.")

    try:
        success = bot_instance.load_and_process_documents()
        if success:
            stats = bot_instance.pinecone_manager.get_index_stats()
            return {
                "status": "success",
                "message": "documents processed and embeddings generated successfully.",
                "total_vectors": stats.total_vector_count if stats else "N/A"
            }
        else:
            return {
                "status": "warning",
                "message": "no documents were processed. Check the logs and the 'docs' directory.",
            }
    except Exception as e:
        logger.exception("Error processing documents")
        raise HTTPException(status_code=500, detail=f"Error to process document: {str(e)}")


@app.post("/clear-documents")
async def clear_documents(request: Request):
    bot_instance = request.app.state.chatbot
    if bot_instance is None:
        raise HTTPException(status_code=503, detail="Chatbot service is unavailable.")

    try:
        success = bot_instance.clear_vector_store()
        if success:
            return {
                "status": "success",
                "message": f"namespace '{bot_instance.pinecone_manager.namespace}' was successfully cleaned."
            }
        else:
            raise HTTPException(status_code=500, detail="error when trying to clean the vector store.")
    except Exception as e:
        logger.exception("Error during vector store clearing.")
        raise HTTPException(status_code=500, detail=f"internal error: {str(e)}")