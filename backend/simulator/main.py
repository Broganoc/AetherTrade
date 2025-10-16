from fastapi import FastAPI

app = FastAPI(title="AetherTrade Simulator")

@app.get("/")
def root():
    return {"status": "Simulator service running"}
