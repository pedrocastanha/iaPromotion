import os
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from src.chat_bot import ChatBot
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    project_root = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(project_root, 'docs')
    logger.info(f"Attempting to load documents from: {docs_path}")

    try:
        app.state.chatbot = ChatBot(docx_dir=docs_path)

        if app.state.chatbot.is_ready():
            logger.info("Chatbot initialized successfully and stored in app.state.")
        else:
            logger.warning("Chatbot initialized, but no documents were found.")

    except Exception as e:
        logger.critical(f"A critical error occurred during chatbot initialization: {e}", exc_info=True)
        app.state.chatbot = None


@app.post("/chat", response_model=dict)
async def chat_route(request: Request, request_body: ChatRequest):
    bot_instance = request.app.state.chatbot

    if bot_instance is None:
        logger.error("Chatbot service is unavailable (retrieved None from app.state).")
        raise HTTPException(status_code=503, detail="chatbot service is unavailable.")

    user_message = request_body.message
    if not user_message or not user_message.strip():
        logger.warning("Received an empty or whitespace-only message from the user.")
        raise HTTPException(status_code=400, detail="The 'message' field cannot be empty.")

    try:
        bot_response = bot_instance.get_response(user_message)
        logger.info(f"Successfully processed message: '{user_message[:50]}...'.")
        return {"response": bot_response}
    except Exception as e:
        logger.exception(f"An error occurred during get_response for message: '{user_message[:50]}...'")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")