# IoT-Based Remote DC Power Signal Monitoring and Anomaly Detection using ESP32 and Machine Learning

## 1. Research paper anchor

**"An Adaptable and Unsupervised TinyML Anomaly Detection System for Extreme Industrial Environments"**
Published in *Sensors* (MDPI). Full text: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9962960/

Core idea borrowed from the paper: an ESP32-based sensing node collects data, trains an
anomaly detection model **on-device** (no manually labeled dataset needed), flags anomalies
locally in real time, and alerts an external system. The paper uses Isolation Forest trained
directly on the microcontroller. This project mirrors that architecture but applies it to DC
voltage monitoring instead of industrial pump vibration, and adds a remote dashboard layer.

Cite this paper in your report's "Related Work" / "Methodology basis" section.

## 2. Hardware — minimal, just ESP32

- ESP32 DevKit — the only required hardware.
- Optional (only if you want to monitor a voltage above 3.3V, e.g. a 5V/9V/12V supply or
  battery): a simple 2-resistor voltage divider. This is two passive resistors, not a sensor
  module.
  - Example for a 0-12V input scaled to 0-3.3V: R1 = 20kΩ (from source+ to ADC pin),
    R2 = 6.8kΩ (from ADC pin to GND). Output = Vin * R2/(R1+R2) ≈ Vin * 0.254.
  - Adjust R1/R2 for your actual source voltage so the max never exceeds 3.3V at the ADC pin.
- DC source to test with: a variable bench power supply, a battery, or a USB power bank —
  anything you can manually vary to create "anomalies" (voltage sag, spike, disconnect).

### Wiring (if using the divider)
| Signal | ESP32 pin |
|---|---|
| Divider output (scaled voltage) | GPIO34 (ADC1, input-only, safe with WiFi active) |
| GND | GND |

If your DC source is already within 0-3.3V, skip the divider and wire it straight into GPIO34.

## 3. Software architecture

```
ESP32 (Arduino/C++)
  - Sample ADC voltage periodically
  - Maintain rolling mean/std (adaptive baseline of "normal" behavior)
  - Flag on-device anomaly via z-score threshold  (lightweight stand-in for the
    paper's on-device Isolation Forest — cheap enough to run continuously on the MCU)
  - Push each reading + anomaly flag to ThingSpeak over WiFi (HTTP GET)

ThingSpeak (free cloud, gives you instant public/remote live-graph URL)
  - Field 1: raw voltage
  - Field 2: on-device anomaly flag (0/1)

Python backend (heavier, "real" ML — this is where the software depth lives)
  - Pulls historical data from ThingSpeak's public API
  - Trains an Isolation Forest (sklearn) on the pulled readings for deeper, retrospective
    anomaly detection (catches subtler anomalies the simple on-device z-score misses)
  - Streamlit dashboard: shows live/historical voltage, on-device flags, and the
    Isolation Forest's flags side by side — accessible remotely once deployed
    (e.g. Streamlit Community Cloud, Render, or Railway free tier)
```

## 4. What's in this zip

- `firmware/esp32_dc_monitor.ino` — ESP32 sketch: ADC sampling, on-device adaptive
  z-score anomaly detection, pushes to ThingSpeak.
- `backend/thingspeak_fetch.py` — pulls channel data from ThingSpeak's REST API into a
  pandas DataFrame.
- `backend/isolation_forest_analysis.py` — trains/runs sklearn's IsolationForest on the
  fetched data, returns anomaly labels.
- `backend/dashboard.py` — Streamlit app: fetches from ThingSpeak, runs the Isolation
  Forest, plots everything, refreshes periodically.
- `backend/requirements.txt` — Python dependencies.
- `data/README.md` — where to drop any exported CSVs if you want offline analysis.

## 5. Setup steps (hand this whole list to Claude Code)

1. **Create a ThingSpeak account** (free) at thingspeak.com, create a new Channel with:
   - Field 1 = "Voltage"
   - Field 2 = "OnDeviceAnomaly"
   Copy your **Write API Key** and **Channel ID** (and Read API Key if channel is private).
2. In `firmware/esp32_dc_monitor.ino`, fill in:
   - `WIFI_SSID`, `WIFI_PASSWORD`
   - `THINGSPEAK_WRITE_API_KEY`
3. Flash the firmware to the ESP32 (Arduino IDE or PlatformIO).
4. In `backend/thingspeak_fetch.py` and `backend/dashboard.py`, fill in your
   `CHANNEL_ID` and `READ_API_KEY` (leave READ_API_KEY as None if your channel is public).
5. `pip install -r backend/requirements.txt`
6. `streamlit run backend/dashboard.py` — this is your remote-viewable dashboard
   (deploy it to Streamlit Community Cloud for a public URL, or run locally and share
   your ThingSpeak channel's own public view URL as the "remote" access point).
7. Vary your DC source manually (turn a supply knob, half-disconnect a battery, etc.)
   and watch both the on-device flag and the Isolation Forest flag react.

## 6. Report structure suggestion

1. Introduction — need for low-cost, hardware-minimal remote DC power monitoring
2. Related Work — cite PMC9962960 (TinyML Isolation Forest on ESP32) as your core basis
3. Methodology — ADC sampling, on-device z-score baseline, ThingSpeak transmission,
   offline Isolation Forest, Streamlit dashboard
4. Results — plots of normal vs. induced-anomaly voltage traces, detection latency,
   comparison of on-device flag vs. Isolation Forest flag accuracy
5. Conclusion — cost (~one ESP32, no sensors), scalability to solar/battery systems,
   future work (on-device Isolation Forest instead of z-score, per the anchor paper)
