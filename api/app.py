"""
api/app.py — FastAPI application factory for MemeIQ

Run:
    uvicorn api.app:app --reload --port 8000

Docs:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from utils.logger import logger


def create_app() -> FastAPI:
    app = FastAPI(
        title="MemeIQ API",
        description=(
            "Multimodal AI system for meme analysis, hate speech detection, "
            "sentiment classification, and meme categorization."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="")

    @app.on_event("startup")
    async def startup():
        logger.info("MemeIQ API starting up...")

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("MemeIQ API shutting down")

    return app


app = create_app()