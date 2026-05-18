from fastapi import FastAPI, Response
import time
import cv2


START = time.monotonic()

APP_INFO = {
    "name":            "RTSP Driver Node",
    "description":     "Restful RTSP stream driver",
    "version":         "0.0.1",
    "author":          "Amontplet",
    "email":           "amontplet@warak.com",
    "company":         "Warak Group",
    "company_website": "https://warak.com",
}

app = FastAPI(
    title=APP_INFO["name"],
    description=APP_INFO["description"],
    version=APP_INFO["version"],
    contact={
        "name":  APP_INFO["author"],
        "email": APP_INFO["email"],
        "url":   APP_INFO["company_website"],
    },
)


@app.get("/health")
async def health():
    return {"status": "ok", "uptime_s": round(time.monotonic() - START, 1)}

@app.get("/")
async def about():
    return APP_INFO
