# Fast Gemini Telegram Chat Bot

Fast, casual Gemini chat bot with optional Hindi voice replies and song search.

## Variables

```env
BOT_TOKEN=your_telegram_bot_token
MONGO_URL=your_mongodb_connection_url
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

`GEMINI_MODEL` is optional. `gemini-3.5-flash-lite` is the default low-latency model.

## Voice

- `mujhe voice me reply do`
- `voice me reply karo`
- `awaaz me jawab do`
- `text me reply do`

## Song requests

Examples:
- `mujhe Arijit Singh ka song sunao`
- `Kesariya bajao`
- `play Believer`

The bot searches YouTube and sends the result link. It does **not** rip or download copyrighted music. For audio playback, use audio files/URLs you have permission to distribute.

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

Deploy with the included Dockerfile and set the variables above.
