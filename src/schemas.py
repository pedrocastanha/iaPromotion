from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    company_name: str
    company_type: str