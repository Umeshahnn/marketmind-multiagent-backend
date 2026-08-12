import os
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("marketmind.main")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.research import router as research_router

app = FastAPI(
    title="MarketMind AI API",
    description="Multi-Agent Market Intelligence Application powered by LangGraph, Groq, and Tavily",
    version="1.0.0"
)

# CORS configuration for React frontend
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"--> Incoming Request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000
        logger.info(f"<-- Request Completed: {request.method} {request.url.path} | Status: {response.status_code} | Duration: {duration:.1f}ms")
        return response
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"<-- Request Failed: {request.method} {request.url.path} | Duration: {duration:.1f}ms | Error: {str(e)}", exc_info=True)
        raise

app.include_router(research_router, prefix="/api")

@app.get("/")
def root():
    return {
        "app": "MarketMind AI",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    groq_key = os.getenv("GROQ_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    langsmith_key = os.getenv("LANGSMITH_API_KEY", "")

    groq_ok = bool(groq_key and "your_groq_api_key" not in groq_key)
    tavily_ok = bool(tavily_key and "your_tavily_api_key" not in tavily_key)
    langsmith_ok = bool(langsmith_key and "your_langsmith_api_key" not in langsmith_key)

    logger.info(f"Health Check Status - Groq: {groq_ok}, Tavily: {tavily_ok}, LangSmith: {langsmith_ok}")

    return {
        "status": "ok",
        "services": {
            "groq_llm": groq_ok,
            "tavily_search": tavily_ok,
            "langsmith_tracing": langsmith_ok
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Unhandled Exception at {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": f"Server internal error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
