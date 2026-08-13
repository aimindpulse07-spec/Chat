# Telegram Gemini AI Bot

Telegram AI chatbot using Google's Gemini API, MongoDB conversation memory, and optional Hindi voice replies.

## Environment variables

```env
BOT_TOKEN=your_telegram_bot_token
MONGO_URL=your_mongodb_connection_url
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

`GEMINI_MODEL` is optional. The default is `gemini-2.5-flash`.

## Voice mode

Send:
- `mujhe voice me reply do`
- `voice me reply karo`
- `awaaz me jawab do`

The bot will switch to voice replies for that user. To switch back:
- `text me reply do`
- `likhkar reply do`

## VPS

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python3 main.py
```

## Railway

Deploy the repository with the included Dockerfile. Add the four environment variables above in Railway Variables.
