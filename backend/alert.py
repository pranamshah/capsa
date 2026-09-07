"""
Telegram alert helper (per anchor paper 2's alerting design).

Setup: create a bot via @BotFather, get the token; message the bot, then check
https://api.telegram.org/bot<TOKEN>/getUpdates for your chat_id.
"""

import requests

BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"


def send_alert(message: str):
    if BOT_TOKEN == "YOUR_BOT_TOKEN":
        print(f"[alert - Telegram not configured] {message}")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=5)
    except requests.RequestException as e:
        print(f"Failed to send alert: {e}")


if __name__ == "__main__":
    send_alert("Test alert from vibration monitor")
