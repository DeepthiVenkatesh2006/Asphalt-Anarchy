import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse

from src.api.routes import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap-API] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pathsnap.app")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Asphalt-Anarchy PathSnap Map-Matching API",
        description="Machine Learning GPS Trajectory Snapping & HMM/Viterbi Evaluation Service",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    ui_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ui"))
    if os.path.exists(ui_dir):
        app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")
        logger.info(f"Mounted Evaluator UI from {ui_dir}")

    @app.get("/", include_in_schema=False)
    def root():
        if os.path.exists(ui_dir):
            return RedirectResponse(url="/ui/")
        return JSONResponse({
            "service": "Asphalt-Anarchy PathSnap API",
            "docs": "/docs",
            "health": "/api/v1/health",
            "trips": "/api/v1/trips"
        })

    logger.info("FastAPI Application initialized successfully.")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
