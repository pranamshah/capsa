# Computer-Aided Vibration Signal Analysis and Anomaly Detection using ESP32, MPU6050, and Machine Learning

> A truthful note on scope: the MPU6050 is a 6-axis accelerometer + gyroscope. It
> measures **vibration and motion**, not electrical power directly. So this project does
> "computer-aided power signal analysis" in the **signal-processing sense**: it acquires a
> real analog signal (triaxial vibration from a battery-powered device), runs FFT / RMS /
> feature extraction on it, and detects anomalies with machine learning. It also monitors
> the **3.7V battery's own voltage** over time via the ESP32 ADC, which is the genuine
> electrical-power angle. Both together give you a complete "signal analysis + power
> monitoring" story that matches your exact hardware.

## 1. Your hardware (used strictly, nothing else)

- ESP32 DevKit
- MPU6050 (I2C accelerometer + gyroscope)
- 3.7V Li-ion / LiPo battery (powers the setup and is itself monitored)
- Optional: 2 resistors for a voltage divider so the ESP32 ADC can read the 3.7V rail
  safely (a 3.7V battery can peak ~4.2V when full, above the 3.3V ADC limit — the divider
  keeps it in range).

### Wiring
MPU6050 -> ESP32 (I2C):
| MPU6050 | ESP32 |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SCL | GPIO22 |
| SDA | GPIO21 |

Battery voltage sensing (via divider):
| Node | ESP32 |
|---|---|
| Divider output (battery+ through R1/R2) | GPIO34 |
| Battery GND | GND |

Example divider for a 4.2V max battery -> 3.3V max at pin: R1 = 10kΩ (battery+ to pin),
R2 = 33kΩ (pin to GND) gives ratio ≈ 0.767, so 4.2V -> ~3.22V. Set VOLTAGE_DIVIDER_RATIO
in the firmware to match your actual resistors.

## 2. Research paper anchors (real, ESP32 + MPU6050 specific)

1. **"Low-Cost IoT-Based Predictive Maintenance Using Vibration"** (Sensors / MDPI, PMC12609400)
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12609400/
   Uses an ESP32 + MPU6050, processes vibration through RMS and FFT, and detects anomalies
   with the Isolation Forest algorithm (~73% accuracy on a small motor). This is your
   primary methodology anchor — almost identical hardware.

2. **"IoT device for detecting abnormal vibrations in motors using TinyML"** (Discover IoT / Springer, 2025)
   https://link.springer.com/article/10.1007/s43926-025-00142-4
   ESP32 + MPU6050, time- and frequency-domain feature extraction, FFT, MQTT streaming,
   K-Means / clustering for anomaly detection, Telegram alerts. Supports the MQTT + alerting
   + clustering parts of your pipeline.

Cite paper 1 for the RMS/FFT + Isolation Forest core, paper 2 for the MQTT streaming and
alerting design.

## 3. Software architecture (heavy)

```
ESP32 (Arduino/C++)
  - Reads MPU6050 accel (ax, ay, az) and gyro at a fixed sample rate (e.g. 500-1000 Hz)
  - Buffers a window of samples (e.g. 256/512)
  - Computes on-device time-domain features: RMS, peak, std, (optionally kurtosis)
  - Reads 3.7V battery voltage via ADC (divider)
  - Publishes each window's features + raw-ish summary + battery voltage over MQTT
      topic: vibration/<yourname>/data
      payload: {"rms":..., "peak":..., "std":..., "vbat":..., "ts":...}

MQTT Broker (broker.hivemq.com free, or your own Mosquitto)

Backend (Python, FastAPI + SQLite)
  - mqtt_subscriber.py : ingest -> SQLite
  - features.py        : full FFT + frequency-domain features on the server side
  - isolation_forest.py: Isolation Forest anomaly detection (paper 1)
  - autoencoder.py     : LSTM/Dense autoencoder, reconstruction-error anomaly detection
                         (deep-learning layer, heavier software)
  - api.py             : FastAPI endpoints for data + model outputs
  - alert.py           : Telegram alert on sustained anomaly (paper 2 style)

Dashboard (Streamlit)
  - Live triaxial vibration features + battery voltage
  - FFT spectrum plot
  - 3-way anomaly comparison: on-device threshold vs Isolation Forest vs Autoencoder
  - Battery discharge curve over time

Deployment
  - Dockerfile + docker-compose.yml to run subscriber + API + dashboard together
```

## 4. What's in this zip

- `firmware/esp32_mpu6050_monitor.ino` — reads MPU6050 + battery, computes features,
  publishes over MQTT.
- `backend/db.py` — SQLite schema + helpers.
- `backend/mqtt_subscriber.py` — MQTT -> SQLite ingest.
- `backend/features.py` — FFT and frequency-domain feature extraction.
- `backend/isolation_forest.py` — Isolation Forest anomaly model.
- `backend/autoencoder.py` — autoencoder (deep learning) anomaly model.
- `backend/api.py` — FastAPI service.
- `backend/alert.py` — Telegram alerts.
- `backend/dashboard.py` — Streamlit dashboard.
- `Dockerfile`, `docker-compose.yml` — containerized deployment.
- `backend/requirements.txt`

## 5. Setup steps (hand to Claude Code)

1. Install Arduino libraries: `MPU6050` (by Electronic Cats or Jeff Rowberg's I2Cdevlib),
   `PubSubClient`, `arduinoFFT` (optional, if doing FFT on-device).
2. Fill in WiFi + MQTT topic in `firmware/esp32_mpu6050_monitor.ino` (keep the topic
   unique and identical to `mqtt_subscriber.py`). Set VOLTAGE_DIVIDER_RATIO to your resistors.
3. Flash the ESP32.
4. `cd backend && pip install -r requirements.txt`
5. Run:
   - `python mqtt_subscriber.py`  (ingest)
   - collect a few minutes of NORMAL vibration, then
     `python autoencoder.py --train` and let Isolation Forest fit on the normal data.
   - `uvicorn api:app --reload`
   - `streamlit run dashboard.py`
   - or `docker compose up --build` to run all backend services at once.
6. Test anomalies: tap/shake the device, place it on something vibrating, or let the
   battery drain — watch the detectors light up in the dashboard.

## 6. Report structure

1. Introduction — low-cost computer-aided vibration signal analysis + battery monitoring
2. Related Work — cite both anchor papers
3. System Architecture — the pipeline above
4. Methodology — sampling, RMS/FFT features, Isolation Forest, autoencoder reconstruction
5. Experimental Setup — how you induced anomalies
6. Results — time/frequency plots, detector comparison, battery discharge analysis
7. Conclusion & Future Work — on-device TinyML (TFLite Micro) deployment
