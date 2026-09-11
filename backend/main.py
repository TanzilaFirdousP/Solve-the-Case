from fastapi import FastAPI

app = FastAPI(title="Detective AI")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Detective AI backend is running"
    }