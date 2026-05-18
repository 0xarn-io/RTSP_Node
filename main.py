from fastapi import FastAPI, Response
import time

app = FastAPI()
START = time.monotonic()

@app.get("/health")
async def health():
    return {"status": "ok", "uptime_s": round(time.monotonic() - START, 1)}

@app.get("/")
async def root():
    return {"message": "Hello World"}
