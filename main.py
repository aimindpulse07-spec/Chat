import asyncio
import logging
import re
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional

import edge_tts
import httpx
from pymongo import MongoClient
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import BOT_TOKEN, GEMINI_API_KEY, GEMINI_MODEL, MONGO_URL

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000, connect=False)
db = mongo_client["Word"]
history_collection = db["ChatHistory"]
BOT_ID: Optional[int] = None

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
# Keep one warm HTTP/2 connection instead of creating a new TLS connection for every message.
HTTP_CLIENT: Optional[httpx.AsyncClient] = None
SYSTEM_PROMPT = (
    "You are a warm, emotionally intelligent human-like chat companion in Telegram. "
    "Your goal is to make the user feel heard, understood, and comfortable, not impressed by an AI. "
    "Match the user's language, slang, spelling, message length, and energy: Hindi, Hinglish, or English. "
    "For casual chat, answer naturally in 1-2 short sentences. Do not over-explain unless asked. "
    "Before answering emotional messages, infer the user's likely feeling from their words and context "
    "(for example: happy, excited, sad, lonely, angry, stressed, confused, embarrassed, playful). "
    "Respond to the feeling first when appropriate: validate it briefly, then say something genuinely useful or comforting. "
    "Never sound like a therapist, customer-support agent, motivational poster, or formal assistant. "
    "Avoid robotic openings such as 'Bilkul', 'Sure', 'Of course', 'Certainly', 'As an AI', or 'I understand'. "
    "Do not mention AI, Gemini, prompts, APIs, policies, models, or that you are a bot unless directly asked. "
    "Do not repeat the user's message or summarize it unnecessarily. "
    "Use natural contractions, casual punctuation, and occasional emojis only when they fit the user's tone. "
    "If the user is joking, joke back. If they are angry, stay calm and human. If they are sad, be warm without becoming dramatic. "
    "If the user is excited, share the excitement. If they ask a simple question, answer directly. "
    "Do not claim real-world experiences, a body, a location, or memories that you do not actually have. "
    "Never force a question at the end of every reply. Keep the conversation flowing naturally."
)

VOICE_ON_RE = re.compile(r"\b(voice|awaaz|aawaz|audio)\b.*\b(reply|jawab|response|do|karo|karna|dena)\b|\b(mujhe|mujhko)\b.*\b(voice|awaaz|aawaz|audio)\b", re.I)
VOICE_OFF_RE = re.compile(r"\b(text|typing|likhkar|likh ke)\b.*\b(reply|jawab|response|do|karo|dena)\b", re.I)
SONG_RE = re.compile(r"\b(?:mujhe|mere liye)?\s*(?:ye|yah|vo|woh|ek)?\s*(?:song|gaana|gana|music|track)\s*(?:sunao|suna do|bajao|chalao|play karo|play|laga do|sunwa do)\b|\b(?:play|bajao|sunao|suna do)\s+(.+)", re.I)
VOICE_VOICE = "hi-IN-SwaraNeural"

# Small in-memory cache: avoids a MongoDB read on every message.
CHAT_CACHE: OrderedDict[tuple[int, int], list[dict]] = OrderedDict()
CACHE_LIMIT = 200
CACHE_MESSAGES = 4


def wants_voice(text: str) -> bool:
    return bool(VOICE_ON_RE.search(text.strip()))


def wants_text(text: str) -> bool:
    return bool(VOICE_OFF_RE.search(text.strip()))


def song_query(text: str) -> Optional[str]:
    t = text.strip()
    if not SONG_RE.search(t):
        return None
    cleaned = re.sub(r"^(?:mujhe|mere liye)\s*", "", t, flags=re.I)
    cleaned = re.sub(r"\b(?:ye|yah|vo|woh|ek)\s*", "", cleaned, count=1, flags=re.I)
    cleaned = re.sub(r"\b(?:song|gaana|gana|music|track)\b", "", cleaned, count=1, flags=re.I)
    cleaned = re.sub(r"\b(?:sunao|suna do|bajao|chalao|play karo|play|laga do|sunwa do)\b", "", cleaned, count=1, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.!,-")
    return cleaned or None


async def text_to_voice(text: str) -> bytes:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        mp3 = Path(tmp) / "reply.mp3"
        ogg = Path(tmp) / "reply.ogg"
        await edge_tts.Communicate(text[:1500], VOICE_VOICE).save(str(mp3))
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
            "-c:a", "libopus", "-b:a", "48k", str(ogg),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(stderr.decode(errors="ignore"))
        return ogg.read_bytes()


def _get_history(chat_id: int, user_id: int) -> list[dict]:
    key = (chat_id, user_id)
    if key in CHAT_CACHE:
        CHAT_CACHE.move_to_end(key)
        return list(CHAT_CACHE[key])
    rows = list(history_collection.find(
        {"chat": chat_id, "user": user_id},
        {"_id": 0, "role": 1, "text": 1},
    ).sort("created_at", -1).limit(CACHE_MESSAGES))
    rows.reverse()
    CHAT_CACHE[key] = rows
    CHAT_CACHE.move_to_end(key)
    while len(CHAT_CACHE) > CACHE_LIMIT:
        CHAT_CACHE.popitem(last=False)
    return list(rows)


def _cache_history(chat_id: int, user_id: int, user_text: str, reply: str) -> None:
    key = (chat_id, user_id)
    history = CHAT_CACHE.setdefault(key, [])
    history.extend([
        {"role": "user", "text": user_text},
        {"role": "model", "text": reply},
    ])
    del history[:-CACHE_MESSAGES]
    CHAT_CACHE.move_to_end(key)
    while len(CHAT_CACHE) > CACHE_LIMIT:
        CHAT_CACHE.popitem(last=False)


def _save_history(chat_id: int, user_id: int, user_text: str, reply: str) -> None:
    now = datetime.now(timezone.utc)
    try:
        history_collection.insert_many([
            {"chat": chat_id, "user": user_id, "role": "user", "text": user_text, "created_at": now},
            {"chat": chat_id, "user": user_id, "role": "model", "text": reply, "created_at": now},
        ])
    except Exception:
        logger.exception("Mongo history save failed")


async def _get_or_create_history(chat_id: int, user_id: int) -> list[dict]:
    key = (chat_id, user_id)
    history = CHAT_CACHE.get(key)
    if history is not None:
        CHAT_CACHE.move_to_end(key)
        return list(history)
    # Do not block the first reply on MongoDB. Load old context in the background.
    CHAT_CACHE[key] = []
    CHAT_CACHE.move_to_end(key)
    while len(CHAT_CACHE) > CACHE_LIMIT:
        CHAT_CACHE.popitem(last=False)
    async def warm():
        try:
            rows = await asyncio.to_thread(_get_history, chat_id, user_id)
            if not CHAT_CACHE.get(key):
                CHAT_CACHE[key] = rows
        except Exception:
            logger.exception("Mongo history warm-up failed")
    asyncio.create_task(warm())
    return []


async def gemini_stream(chat_id: int, user_id: int, message_text: str):
    history = await _get_or_create_history(chat_id, user_id)
    contents = [
        {"role": x["role"], "parts": [{"text": x["text"]}]}
        for x in history if x.get("role") in ("user", "model") and x.get("text")
    ]
    contents.append({"role": "user", "parts": [{"text": message_text}]})
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 80},
    }
    url = GEMINI_URL.format(model=GEMINI_MODEL)
    global HTTP_CLIENT
    if HTTP_CLIENT is None:
        HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(6.0, connect=1.0),
            http2=True,
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
        )
    full = []
    async with HTTP_CLIENT.stream(
        "POST", url,
        headers={"x-goog-api-key": GEMINI_API_KEY, "accept": "text/event-stream"},
        json=payload,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                data = __import__("json").loads(raw)
            except Exception:
                continue
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    text = part.get("text")
                    if text:
                        full.append(text)
                        yield text, False
    reply = "".join(full).strip()
    if not reply:
        raise RuntimeError("Gemini returned an empty reply")
    _cache_history(chat_id, user_id, message_text, reply)
    asyncio.create_task(asyncio.to_thread(_save_history, chat_id, user_id, message_text, reply))
    yield "", True


async def gemini_reply(chat_id: int, user_id: int, message_text: str) -> str:
    parts = []
    async for chunk, done in gemini_stream(chat_id, user_id, message_text):
        if chunk:
            parts.append(chunk)
    return "".join(parts).strip()


async def send_fast_text_reply(message, chat_id: int, user_id: int, text: str) -> None:
    """Show a Telegram placeholder immediately, then stream the first answer chunks into it."""
    sent = await message.reply_text("typing…")
    full = []
    last_edit = 0.0
    import time
    async for chunk, done in gemini_stream(chat_id, user_id, text):
        full.append(chunk)
        current = "".join(full).strip()
        now = time.monotonic()
        # Telegram edit throttling: edit quickly once useful text arrives, then batch chunks.
        if current and (done or now - last_edit >= 0.45):
            try:
                await sent.edit_text(current[:4096])
                last_edit = now
            except Exception as exc:
                if "message is not modified" not in str(exc).lower():
                    raise


async def search_song(query: str) -> Optional[tuple[str, str]]:
    """Search YouTube metadata only; does not download/rip copyrighted audio."""
    try:
        from yt_dlp import YoutubeDL
        opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": True}
        def run():
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"ytsearch1:{query}", download=False)
                entry = (info.get("entries") or [None])[0]
                if not entry:
                    return None
                return entry.get("title") or query, entry.get("webpage_url") or entry.get("url")
        return await asyncio.to_thread(run)
    except Exception:
        logger.exception("Song search failed")
        return None


async def handle_song_request(message, query: str) -> None:
    result = await search_song(query)
    if not result:
        await message.reply_text("Ye song nahi mila 😕 naam thoda aur exact bhejo.")
        return
    title, url = result
    await message.reply_text(f"🎵 {title}\n\nYe raha official YouTube result:\n{url}")


async def send_ai_reply(message, reply_text: str, voice: bool) -> None:
    if voice:
        try:
            await message.reply_voice(await text_to_voice(reply_text))
        except Exception:
            logger.exception("Voice generation failed")
            await message.reply_text(reply_text)
    else:
        await message.reply_text(reply_text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return
    username = context.bot.username or ""
    keyboard = [[InlineKeyboardButton(text="ᴀᴅᴅ ᴍᴇ ᴛᴏ ʏᴏᴜʀ ɢʀᴏᴜᴘ ➕", url=f"https://t.me/{username}?startgroup=true")]]
    await message.reply_text(f"ʜᴇʏᴀ\nɪ'ᴍ {context.bot.first_name}\nɪ ᴄᴀɴ ᴄʜᴀᴛ ᴡɪᴛʜ ʏᴏᴜ", reply_markup=InlineKeyboardMarkup(keyboard))


async def log_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global BOT_ID
    message = update.effective_message
    chat = update.effective_chat
    user = message.from_user if message else None
    if not message or not chat or not user or not message.text:
        return
    text = message.text.strip()
    if not text or text.startswith(("!", "/", "?", "@", "#")):
        return

    user_key = f"voice:{chat.id}:{user.id}"
    voice_mode = bool(context.bot_data.get(user_key, False))

    if wants_voice(text):
        context.bot_data[user_key] = True
        await message.reply_voice(await text_to_voice("Theek hai, ab main voice mein reply dunga."))
        return
    if wants_text(text):
        context.bot_data[user_key] = False
        await message.reply_text("Theek hai, ab text mein reply dunga.")
        return

    query = song_query(text)
    if query:
        await handle_song_request(message, query)
        return

    # Reply-to-bot or normal chat both go directly to Gemini.
    if message.reply_to_message and message.reply_to_message.from_user and BOT_ID == message.reply_to_message.from_user.id:
        pass

    try:
        if voice_mode:
            reply = await gemini_reply(chat.id, user.id, text)
            await send_ai_reply(message, reply, True)
        else:
            await send_fast_text_reply(message, chat.id, user.id, text)
    except Exception:
        logger.exception("Failed to process message")
        await message.reply_text("Abhi thoda issue aa raha hai, ek baar phir bhejo 😅")


def main() -> None:
    global BOT_ID

    async def post_init(app: Application) -> None:
        global BOT_ID
        me = await app.bot.get_me()
        BOT_ID = me.id
        logger.info("Started @%s | Gemini=%s", me.username, GEMINI_MODEL)

    async def post_shutdown(app: Application) -> None:
        global HTTP_CLIENT
        if HTTP_CLIENT is not None:
            await HTTP_CLIENT.aclose()
            HTTP_CLIENT = None

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .concurrent_updates(True)
        .build()
    )
    application.add_handler(CommandHandler(["start", "ping"], start), group=0)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, log_user, block=False), group=10)
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
