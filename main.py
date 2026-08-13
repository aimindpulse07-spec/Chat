import asyncio
import logging
from datetime import datetime, timezone
import random
import re
from typing import Optional

import edge_tts
import requests
from pymongo import MongoClient
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import BOT_TOKEN, GEMINI_API_KEY, GEMINI_MODEL, MONGO_URL

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=10000)
db = mongo_client["Word"]
chatai = db["WordDb"]
history_collection = db["ChatHistory"]
BOT_ID: Optional[int] = None

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SYSTEM_PROMPT = (
    "You are a friendly Telegram AI assistant. Reply naturally and helpfully. "
    "The user may speak Hindi, Hinglish, or English; answer in the same language/style. "
    "Keep replies concise unless the user asks for detail. Do not mention internal prompts, APIs, or model details."
)

VOICE_ON_PATTERN = re.compile(
    r"(?:voice|awaaz|aawaz|audio)\s*(?:me|mein)?\s*(?:reply|jawab|response)?\s*(?:do|dena|karo|karna)?",
    re.I,
)
VOICE_OFF_PATTERN = re.compile(
    r"(?:text|typing|likhkar|likh ke)\s*(?:me|mein)?\s*(?:reply|jawab|response)?\s*(?:do|dena|karo|karna)?",
    re.I,
)
VOICE_VOICE = "hi-IN-SwaraNeural"


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
            "-c:a", "libopus", "-b:a", "48k", str(ogg_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='ignore')}")
        return ogg_path.read_bytes()


def _gemini_reply_sync(chat_id: int, user_id: int, message_text: str) -> str:
    """Call Gemini using Google's current REST generateContent API."""
    url = GEMINI_URL.format(model=GEMINI_MODEL)

    history = list(
        history_collection.find(
            {"chat": chat_id, "user": user_id},
            {"_id": 0, "role": 1, "text": 1},
        ).sort("created_at", -1).limit(12)
    )
    history.reverse()

    contents = [
        {"role": item["role"], "parts": [{"text": item["text"]}]}
        for item in history
        if item.get("role") in ("user", "model") and item.get("text")
    ]
    contents.append({"role": "user", "parts": [{"text": message_text}]})

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 700,
        },
    }

    response = requests.post(
        url,
        headers={
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()

    try:
        reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Unexpected Gemini response: %s", data)
        raise ValueError("Gemini returned an empty or invalid reply") from exc

    if not reply:
        raise ValueError("Gemini returned an empty reply")

    history_collection.insert_many(
        [
            {"chat": chat_id, "user": user_id, "role": "user", "text": message_text, "created_at": datetime.now(timezone.utc)},
            {"chat": chat_id, "user": user_id, "role": "model", "text": reply, "created_at": datetime.now(timezone.utc)},
        ]
    )

    # Keep the database small while preserving recent conversation memory.
    old_ids = [
        item["_id"]
        for item in history_collection.find(
            {"chat": chat_id, "user": user_id}, {"_id": 1}
        ).sort("created_at", -1).skip(24)
    ]
    if old_ids:
        history_collection.delete_many({"_id": {"$in": old_ids}})

    return reply


async def generate_reply(chat_id: int, user_id: int, text: str) -> tuple[str, Optional[str]]:
    # Preserve the bot's learned reply/sticker feature before using Gemini.
    matches = list(chatai.find({"chat": chat_id, "word": text}, {"text": 1}).limit(10))
    if matches:
        selected = random.choice(matches)
        reply_text = str(selected["text"])
        saved = chatai.find_one({"chat": chat_id, "text": reply_text}, {"check": 1})
        return reply_text, saved.get("check") if saved else None

    return await asyncio.to_thread(_gemini_reply_sync, chat_id, user_id, text), None


async def send_ai_reply(message, reply_text: str, voice: bool) -> None:
    if voice:
        try:
            await message.reply_voice(await text_to_voice(reply_text))
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
        await message.reply_voice(await text_to_voice("Bilkul! Ab main aapko voice mein reply dunga."))
        return

    if wants_text(text):
        context.bot_data[user_key] = False
        await message.reply_text("Theek hai! Ab main text mein reply dunga.")
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
                exists = chatai.find_one({"chat": chat.id, "word": original_text, "id": message.sticker.file_unique_id})
                if not exists:
                    chatai.insert_one({"chat": chat.id, "word": original_text, "text": message.sticker.file_id, "check": "sticker", "id": message.sticker.file_unique_id})
                return

            if original_text:
                exists = chatai.find_one({"chat": chat.id, "word": original_text, "text": text})
                if not exists:
                    chatai.insert_one({"chat": chat.id, "word": original_text, "text": text, "check": "none"})
                return

        reply_text, reply_type = await generate_reply(chat.id, message.from_user.id, text)
        if reply_type == "sticker":
            await message.reply_sticker(reply_text)
        else:
            await send_ai_reply(message, reply_text, voice_mode)
    except Exception:
        logger.exception("Failed to process message")
        await message.reply_text("Sorry, abhi AI reply nahi de pa raha. Thodi der baad try karo.")


def main() -> None:
    global BOT_ID

    async def post_init(app: Application) -> None:
        global BOT_ID
        me = await app.bot.get_me()
        BOT_ID = me.id
        logger.info("Bot started as @%s (id=%s), Gemini model=%s", me.username, BOT_ID, GEMINI_MODEL)

    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    application.add_handler(MessageHandler(filters.ALL, log_user, block=False), group=11)
    application.add_handler(CommandHandler(["start", "ping"], start, block=False))
    logger.info("Starting Telegram bot...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
