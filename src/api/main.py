"""
main.py
-------
FastAPI entry point for KrishiRakshak.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import LoggingMiddleware
from src.api.routes     import router

logging.basicConfig(
    level  =logging.INFO,
    format ="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(
    title      ="KrishiRakshak API",
    version    ="1.0.0",
    description="AI-powered crop disease diagnosis for Indian farmers",
    docs_url   ="/docs",
    redoc_url  ="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins    =["*"],
    allow_credentials=True,
    allow_methods    =["*"],
    allow_headers    =["*"],
)

app.add_middleware(LoggingMiddleware)
app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "KrishiRakshak",
        "version": "1.0.0",
        "docs"   : "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host   =os.getenv("HOST", "0.0.0.0"),
        port   =int(os.getenv("PORT", 8000)),
        workers=int(os.getenv("WORKERS", 1)),
        reload =os.getenv("DEBUG", "false").lower() == "true",
    )
