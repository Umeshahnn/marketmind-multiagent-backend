import os
import logging
from typing import Optional
from langchain_groq import ChatGroq

logger = logging.getLogger("marketmind.llm")

def get_groq_llm(temperature: float = 0.2, max_tokens: Optional[int] = 800) -> ChatGroq:
    """
    Instantiates and returns the Groq LLM client.
    Raises ValueError if GROQ_API_KEY is missing or invalid.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key.strip() == "" or "your_groq_api_key" in groq_key:
        logger.error("GROQ_API_KEY is missing or invalid in environment variables!")
        raise ValueError("GROQ_API_KEY is not configured in environment variables.")
    
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    logger.debug(f"Initializing ChatGroq (model={model_name}, temp={temperature}, max_tokens={max_tokens})")
    
    kwargs = {
        "model": model_name,
        "groq_api_key": groq_key,
        "temperature": temperature
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
        
    return ChatGroq(**kwargs)
