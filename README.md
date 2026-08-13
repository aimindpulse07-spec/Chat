# Gemini Human Fast Telegram Bot

Fast, casual, emotion-aware Telegram chat bot.

Required:
```env
BOT_TOKEN=...
MONGO_URL=...
GEMINI_API_KEY=...
```

Optional:
```env
GEMINI_API_KEY_2=...
GEMINI_MODEL=gemini-3.5-flash-lite
```

`GEMINI_API_KEY_2` is a second Gemini project/key used only as failover if the
first key hits a transient quota/server error. It is optional.

The bot uses short responses, minimal thinking, HTTP/2 keep-alive, in-memory
conversation history, background MongoDB writes, Telegram typing indicators,
streaming first-token replies, and instant local replies for common greetings.

Exact 1-second latency cannot be guaranteed because Telegram/network/API
latency is outside the code's control.

Voice and YouTube metadata search are retained from the original project.
