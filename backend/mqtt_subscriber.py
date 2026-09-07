"""Subscribes to the ESP32's MQTT topic and writes feature windows into SQLite."""

import json
import paho.mqtt.client as mqtt

from db import init_db, insert_window

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "vibration/yourname/data"  # must match the ESP32 firmware


def on_connect(client, userdata, flags, rc):
    print(f"Connected (rc={rc}), subscribing to {MQTT_TOPIC}")
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    try:
        p = json.loads(msg.payload.decode())
        insert_window(float(p["rms"]), float(p["peak"]), float(p["std"]),
                      float(p.get("vbat", 0)))
        print(f"Stored rms={p['rms']} peak={p['peak']} vbat={p.get('vbat')}")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Bad message skipped: {msg.payload} ({e})")


def main():
    init_db()
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
