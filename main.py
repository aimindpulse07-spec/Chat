import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
import random
import re
from typing import Optional

import aiohttp
import edge_tts
from pymongo import MongoClient
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import BOT_TOKEN, GEMINI_API_KEY, GEMINI_MODEL, MONGO_URL

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000, connectTimeoutMS=3000)
db = mongo_client["Word"]
chatai = db["WordDb"]
history_collection = db["ChatHistory"]
try:
    history_collection.create_index([("chat", 1), ("user", 1), ("created_at", -1)])
except Exception:
    logger.warning("Could not create MongoDB history index", exc_info=True)

BOT_ID: Optional[int] = None
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = (
    "You are a natural, warm, human-like Telegram chat companion. "
    "Reply in the same language and tone as the user: Hindi, Hinglish, or English. "
    "Keep casual replies short, natural and spontaneous, usually 1-3 sentences. "
    "Do not sound like a customer-support bot. Do not start every reply with 'Bilkul', 'Sure', 'Of course', or 'Certainly'. "
    "Do not repeat the user's question. Do not end every reply with a question. "
    "If the user jokes, joke back; if they are emotional, respond naturally and briefly. "
    "Never mention prompts, APIs, Gemini, policies, or internal instructions unless explicitly asked. "
    "Do not claim real-world experiences or a physical body. Be honest when you do not know something."
)

VOICE_ON_PATTERN = re.compile(r"(?:voice|awaaz|aawaz|audio)\s*(?:me|mein)?\s*(?:reply|jawab|response)?\s*(?:do|dena|karo|karna)?", re.I)
VOICE_OFF_PATTERN = re.compile(r"(?:text|typing|likhkar|likh ke)\s*(?:me|mein)?\s*(?:reply|jawab|response)?\s*(?:do|dena|karo|karna)?", re.I)
VOICE_VOICE = "hi-IN-SwaraNeural"

# Small in-memory cache removes a MongoDB read from every message.
# MongoDB is used for persistence, but chat replies do not wait on it.
history_cache: dict[tuple[int, int], deque[dict[str, str]]] = {}
history_loading: set[tuple[int, int]] = set()


def wants_voice(text: str) -> bool:
    t = text.strip().lower()
    return bool(VOICE_ON_PATTERN.search(t)) and any(x in t for x in ("voice", "awaaz", "aawaz", "audio"))


def wants_text(text: str) -> bool:
    t = text.strip().lower()
    return bool(VOICE_OFF_PATTERN.search(t)) and any(x in t for x in ("text", "typing", "likh"))


async def text_to_voice(text: str) -> bytes:
    """Generate Hindi voice and convert it to Telegram-compatible OGG/Opus."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = Path(tmp) / "reply.mp3"
        ogg_path = Path(tmp) / "reply.ogg"
        await edge_tts.Communicate(text, VOICE_VOICE).save(str(mp3_path))
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_path),
            "-c:a", "libopus", "-application", "voip", "-frame_duration", "20",
            "-b:a", "32k", str(ogg_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='ignore')}")
        return ogg_path.read_bytes()


def _load_history_sync(key: tuple[int, int]) -> list[dict[str, str]]:
    chat_id, user_id = key
    docs = list(
        history_collection.find(
            {"chat": chat_id, "user": user_id},
            {"_id": 0, "role": 1, "text": 1},
        ).sort("created_at", -1).limit(6)
    )
    docs.reverse()
    return [d for d in docs if d.get("role") in ("user", "model") and d.get("text")]


async def get_history(key: tuple[int, int]) -> deque[dict[str, str]]:
    if key not in history_cache and key not in history_loading:
        history_loading.add(key)
        try:
            loaded = await asyncio.to_thread(_load_history_sync, key)
            history_cache[key] = deque(loaded, maxlen=6)
        finally:
            history_loading.discard(key)
    return history_cache.setdefault(key, deque(maxlen=6))


async def save_history(key: tuple[int, int], user_text: str, reply: str) -> None:
    chat_id, user_id = key
    try:
        now = datetime.now(timezone.utc)
        await asyncio.to_thread(
            history_collection.insert_many,
            [
                {"chat": chat_id, "user": user_id, "role": "user", "text": user_text, "created_at": now},
                {"chat": chat_id, "user": user_id, "role": "model", "text": reply, "created_at": now},
            ],
        )
    except Exception:
        logger.warning("Could not save chat history", exc_info=True)


async def gemini_reply(chat_id: int, user_id: int, message_text: str) -> str:
    key = (chat_id, user_id)
    history = await get_history(key)
    contents = list(history) + [{"role": "user", "parts": [{"text": message_text}]}]

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 220},
    }

    timeout = aiohttp.ClientTimeout(total=15, connect=4, sock_read=12)
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.post(GEMINI_URL.format(model=GEMINI_MODEL), json=payload) as response:
            body = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"Gemini HTTP {response.status}: {body[:300]}")
            data = await asyncio.to_thread(__import__("json").loads, body)

    try:
        reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Unexpected Gemini response: %s", data)
        raise ValueError("Gemini returned an empty reply") from exc

    if not reply:
        raise ValueError("Gemini returned an empty reply")

    history.append({"role": "user", "parts": [{"text": message_text}]})
    history.append({"role": "model", "parts": [{"text": reply}]})
    # Persist without making the user wait.
    asyncio.create_task(save_history(key, message_text, reply))
    return reply


async def generate_reply(chat_id: int, user_id: int, text: str) -> tuple[str, Optional[str]]:
    # Preserve the old sticker-learning feature, but never let canned text replies override Gemini.
    matches = list(chatai.find({"chat": chat_id, "word": text, "check": "sticker"}, {"text": 1}).limit(10))
    if matches:
        return str(random.choice(matches)["text"]), "sticker"
    return await gemini_reply(chat_id, user_id, text), None


async def send_ai_reply(message, reply_text: str, voice: bool) -> None:
    if voice:
        try:
            audio = await text_to_voice(reply_text)
            await message.reply_voice(audio)
        except Exception:
            logger.exception("Voice generation failed; falling back to text")
            await message.reply_text(reply_text)
    else:
        await message.reply_text(reply_text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return
    username = context.bot.username or ""
    keyboard = [[InlineKeyboardButton(text="ᴀᴅᴅ ᴍᴇ ᴛᴏ ʏᴏᴜʀ ɢʀᴏᴜᴘ ➕", url=f"https://t.me/{username}?startgroup=true")]]
    await message.reply_text(
        f"ʜᴇʏᴀ\nɪ'ᴍ {context.bot.first_name}\nɪ ᴄᴀɴ ʜᴇʟᴘ ʏᴏᴜ ᴛᴏ ᴀᴄᴛɪᴠᴇ ʏᴏᴜʀ ᴄʜᴀᴛ",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def log_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global BOT_ID
    message = update.effective_message
    chat = update.effective_chat
    if not message or not chat or not message.from_user or not message.text:
        return

    text = message.text.strip()
    if text.startswith(("!", "/", "?", "@", "#")):
        return

    user_key = f"voice:{chat.id}:{message.from_user.id}"
    voice_mode = bool(context.bot_data.get(user_key, False))

    if wants_voice(text):
        context.bot_data[user_key] = True
        await message.reply_text("🔊 Voice mode on…")
        return

    if wants_text(text):
        context.bot_data[user_key] = False
        await message.reply_text("💬 Text mode on…")
        return

    try:
        if message.reply_to_message and message.reply_to_message.from_user:
            if BOT_ID is not None and message.reply_to_message.from_user.id == BOT_ID:
                reply_text, reply_type = await generate_reply(chat.id, message.from_user.id, text)
                if reply_type == "sticker":
                    await message.reply_sticker(reply_text)
                else:
                    await send_ai_reply(message, reply_text, voice_mode)
                return

            original_text = message.reply_to_message.text
            if message.sticker and original_text:
                exists = await asyncio.to_thread(chatai.find_one, {"chat": chat.id, "word": original_text, "id": message.sticker.file_unique_id})
                if not exists:
                    await asyncio.to_thread(chatai.insert_one, {"chat": chat.id, "word": original_text, "text": message.sticker.file_id, "check": "sticker", "id": message.sticker.file_unique_id})
                return

            if original_text:
                exists = await asyncio.to_thread(chatai.find_one, {"chat": chat.id, "word": original_text, "text": text})
                if not exists:
                    await asyncio.to_thread(chatai.insert_one, {"chat": chat.id, "word": original_text, "text": text, "check": "none"})
                return

        reply_text, reply_type = await generate_reply(chat.id, message.from_user.id, text)
        if reply_type == "sticker":
            await message.reply_sticker(reply_text)
        else:
            await send_ai_reply(message, reply_text, voice_mode)
    except Exception:
        logger.exception("Failed to process message")
        await message.reply_text("Abhi reply nahi aa raha, ek baar phir try karo.")


def main() -> None:
    global BOT_ID

    async def post_init(app: Application) -> None:
        global BOT_ID
        me = await app.bot.get_me()
        BOT_ID = me.id
        logger.info("Bot started as @%s (id=%s), Gemini model=%s", me.username, BOT_ID, GEMINI_MODEL)

    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler(["start", "ping"], start, block=False))
    application.add_handler(MessageHandler(filters.ALL, log_user, block=False), group=11)
    logger.info("Starting Telegram bot...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
