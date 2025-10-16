from fastapi import FastAPI

app = FastAPI(title="AetherTrade Trainer")

@app.get("/")
def root():
    return {"status": "Trainer service running"}
