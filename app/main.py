from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.controllers.controller import router as api_router
from app.utils.model_loader import ModelLoader

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 SERVER STARTING... LOADING AI MODELS...")
    ModelLoader.load_models()
    print("🚀 SYSTEM READY!\n")
    
    yield
    
    print("🛑 Server shutting down...")

app = FastAPI(title="Recommender System API", version="1.0", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1", tags=["Recommendation"])

@app.get("/")
def health_check():
    return {"status": "online", "models_loaded": list(ModelLoader._models.keys())}