from fastapi import FastAPI
from app.api.routes import classify

app = FastAPI(title="Template Classifier API", version="1.0")

app.include_router(classify.router, prefix="/classify", tags=["classify"])
