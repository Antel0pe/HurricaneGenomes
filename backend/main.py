from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from .track_processing.hurdat2_processing import load_storm_tracks, STORM_TRACKS
except ImportError:
    from track_processing.hurdat2_processing import load_storm_tracks, STORM_TRACKS

app = FastAPI(title="HurricaneGenomes API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "message": "backend working"}


@app.get("/echo")
def echo(text: str = "hello") -> dict:
    return {"echo": text}


def on_server_startup() -> None:
    app.state.tracks = STORM_TRACKS if STORM_TRACKS else load_storm_tracks()


@app.on_event("startup")
def _startup_event() -> None:
    on_server_startup()


@app.get("/tracks")
def get_tracks() -> dict:
    return app.state.tracks if hasattr(app.state, "tracks") else {}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


