# Project Responsibilities & Full Information
## Computer-Aided Vibration Signal Analysis and Anomaly Detection using ESP32, MPU6050, and Machine Learning

This file explains exactly who does what, so the split between your work and the
work already done for you (by Claude / Claude Code) is clear.

---

## A. What has ALREADY been done for you (by Claude)

All of the software in this zip is written and ready:

- **ESP32 firmware** (`firmware/esp32_mpu6050_monitor.ino`)
  Reads the MPU6050 acceleration, computes time-domain features (RMS, peak, std),
  reads the battery voltage through the divider, and publishes everything over MQTT.

- **Full Python backend** (in `backend/`)
  - `db.py` – SQLite database schema and helpers
  - `mqtt_subscriber.py` – receives ESP32 data over MQTT and stores it
  - `features.py` – FFT and frequency-domain feature extraction
  - `isolation_forest.py` – Isolation Forest anomaly detection model
  - `autoencoder.py` – deep-learning autoencoder anomaly model
  - `api.py` – FastAPI service exposing the data and model outputs
  - `alert.py` – Telegram alerting helper
  - `dashboard.py` – Streamlit live remote dashboard

- **Deployment** (`Dockerfile`, `docker-compose.yml`)
  One command runs the whole backend so it can be hosted for remote access.

- **Research grounding**
  The whole design follows two real, peer-reviewed papers that use the same
  ESP32 + MPU6050 hardware (see the main guide document for full summaries).

In short: the architecture, all code, and the academic basis are complete.

---

## B. What YOU need to do

### Hardware (your part)
1. Wire the MPU6050 to the ESP32 over I2C:
   SCL→GPIO22, SDA→GPIO21, VCC→3.3V, GND→GND.
2. Build a 2-resistor voltage divider from the 3.7V battery to GPIO34 so the
   ESP32 can safely read the battery voltage (a full battery is ~4.2V, above the
   3.3V ADC limit). Example: R1 = 10kΩ (battery+ to pin), R2 = 33kΩ (pin to GND).
3. Connect a common ground between battery, divider, ESP32, and MPU6050.
4. Mount the MPU6050 on the device you want to monitor (a small fan or motor is ideal).

### Software configuration (your part — small edits only)
5. In `firmware/esp32_mpu6050_monitor.ino`, fill in:
   - your WiFi name and password
   - a unique MQTT topic, e.g. `vibration/yourname/data`
   - the VOLTAGE_DIVIDER_RATIO for your resistors
6. In `backend/mqtt_subscriber.py`, set the same MQTT topic.
7. (Optional) In `backend/alert.py`, add your Telegram bot token and chat ID.

### Running it (your part — follow the steps)
8. Flash the firmware to the ESP32 (Arduino IDE).
9. `pip install -r backend/requirements.txt`
10. `python mqtt_subscriber.py` (leave running)
11. Collect a few minutes of NORMAL vibration data.
12. Train/fit the models: `python isolation_forest.py` and `python autoencoder.py --train`
13. `uvicorn api:app --reload` and `streamlit run dashboard.py`
    (or `docker compose up --build` to run everything together)
14. Demonstrate: create anomalies (tap/shake/imbalance) and show the models
    detecting them live; let the battery drain to show the discharge curve.

### Using Claude Code (recommended for the software part)
Open this whole folder in Claude Code and ask it to help you with steps 5–14.
It can fill in the config, install dependencies, run each service, and help you
debug anything that doesn't work the first time.

---

## C. One-line summary of the project
The system watches the vibration signal of a battery-powered device and uses
machine learning to automatically detect when the device starts behaving
abnormally (a fault), while also monitoring the battery's voltage — all
viewable remotely.

## D. Honest scope note (say this in your viva)
The MPU6050 measures vibration, not electrical power. So this is
"computer-aided power signal analysis" in the signal-processing sense
(acquire a real signal → FFT/RMS → analyse with ML → visualise), plus genuine
battery-voltage monitoring for the electrical-power element. This framing is
accurate and matches the anchor research paper exactly.
