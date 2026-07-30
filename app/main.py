from fastapi import FastAPI

from app.api import routes_ingest, routes_search

app = FastAPI(title="GoFolyX AI Service")

app.include_router(routes_ingest.router)
app.include_router(routes_search.router)


@app.get("/health")
def health():
    return {"status": "ok"}
