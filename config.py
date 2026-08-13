import os


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


BOT_TOKEN = _required("BOT_TOKEN")
MONGO_URL = _required("MONGO_URL")
GEMINI_API_KEY = _required("GEMINI_API_KEY")
# Fast, low-latency model for chat. Override with GEMINI_MODEL if desired.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
