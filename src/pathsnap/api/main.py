"""PathSnap FastAPI main application entry point."""
from src.api.app import app, create_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.pathsnap.api.main:app", host="0.0.0.0", port=8000, reload=True)
