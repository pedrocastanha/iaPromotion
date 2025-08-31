import google.generativeai as genai
import os
import json
import logging
from .schemas import ChatResponse

logger = logging.getLogger(__name__)

class GeminiIntegration:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be defined at .env")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 0.8,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
                "response_mime_type": "application/json",
            }
        )

    def get_gemini_response(self, prompt: str) -> ChatResponse:
        try:
            response = self.model.generate_content(prompt)
            response_dict = json.loads(response.text)

            return ChatResponse(**response_dict)

        except json.JSONDecodeError as e:
            logger.error(f"Erro ao decodificar JSON da resposta do modelo: {e}")
            logger.error(f"Resposta recebida que causou o erro: {response.text}")
            raise ValueError(f"A resposta do modelo não foi um JSON válido.") from e
        except Exception as e:
            logger.error(f"Erro inesperado ao gerar resposta do Gemini: {e}")
            raise