"""
FastAPI backend for chatbot communication
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from src.bot import Portfolio

app = FastAPI(title="Portfolio Chatbot API")
portfolio = Portfolio()

class Message(BaseModel):
    """Message model for chat requests"""
    text: str

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    emotion: str = "neutral"

@app.post("/chat", response_model=ChatResponse)
async def chat(message: Message) -> Dict[str, Any]:
    """
    Process chat message and return response

    Args:
        message: Message object containing input text

    Returns:
        Dictionary containing response text and emotion
    """
    try:
        response = portfolio.process_message(message.text)
        return ChatResponse(
            response=response["text"],
            emotion=response.get("emotion", "neutral")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}
