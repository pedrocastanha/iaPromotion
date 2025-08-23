import google.generativeai as genai
import os


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
            }
        )

    def get_gemini_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"error generating answer: {e}"