/*
  IoT-Based Remote DC Power Signal Monitoring and Anomaly Detection
  ESP32 Firmware

  Hardware: ESP32 only (+ optional 2-resistor voltage divider if monitoring >3.3V)
    Divider output / DC signal -> GPIO34 (ADC1, safe alongside WiFi)

  What it does:
    1. Samples DC voltage on GPIO34 every SAMPLE_INTERVAL_MS
    2. Maintains an adaptive rolling mean/std of "normal" behavior
    3. Flags an anomaly if the current reading's z-score exceeds Z_THRESHOLD
       (this is a lightweight on-device stand-in for the Isolation Forest approach
       described in the anchor paper -- cheap enough to run continuously on the MCU)
    4. Pushes voltage + anomaly flag to a ThingSpeak channel over WiFi

  TODO before flashing:
    - WIFI_SSID / WIFI_PASSWORD
    - THINGSPEAK_WRITE_API_KEY
    - VOLTAGE_DIVIDER_RATIO (set to 1.0 if wiring the DC source directly, no divider)
*/

#include <WiFi.h>
#include <HTTPClient.h>

const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* THINGSPEAK_WRITE_API_KEY = "YOUR_WRITE_API_KEY";
const char* THINGSPEAK_URL = "http://api.thingspeak.com/update";

const int   ADC_PIN               = 34;
const float ADC_MAX_VOLTAGE       = 3.3;
const float ADC_MAX_COUNTS        = 4095.0;
const float VOLTAGE_DIVIDER_RATIO = 0.254;  // set to 1.0 if no divider used

const unsigned long SAMPLE_INTERVAL_MS   = 2000;   // one reading every 2s
const unsigned long BASELINE_WARMUP_N    = 15;     // readings before flagging starts
const float          Z_THRESHOLD          = 3.0;    // anomaly if |z| > this

// Rolling (exponential) mean/std for adaptive baseline
float runningMean = 0;
float runningVar  = 0;
unsigned long sampleCount = 0;
const float alpha = 0.05; // smoothing factor for the running stats

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nConnected. IP: " + WiFi.localIP().toString());
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  connectWiFi();
}

float readVoltage() {
  int raw = analogRead(ADC_PIN);
  float vAtPin = (raw / ADC_MAX_COUNTS) * ADC_MAX_VOLTAGE;
  return vAtPin / VOLTAGE_DIVIDER_RATIO;  // scaled back to actual source voltage
}

// Update adaptive mean/std, return current z-score
float updateBaselineAndGetZ(float value) {
  sampleCount++;
  if (sampleCount == 1) {
    runningMean = value;
    runningVar = 0;
    return 0;
  }
  float diff = value - runningMean;
  runningMean += alpha * diff;
  runningVar = (1 - alpha) * (runningVar + alpha * diff * diff);
  float std = sqrt(runningVar) + 1e-6;
  return diff / std;
}

void sendToThingSpeak(float voltage, bool anomaly) {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }
  HTTPClient http;
  String url = String(THINGSPEAK_URL) +
               "?api_key=" + THINGSPEAK_WRITE_API_KEY +
               "&field1=" + String(voltage, 3) +
               "&field2=" + String(anomaly ? 1 : 0);
  http.begin(url);
  int httpCode = http.GET();
  if (httpCode > 0) {
    Serial.println("ThingSpeak update sent, code: " + String(httpCode));
  } else {
    Serial.println("ThingSpeak update failed: " + http.errorToString(httpCode));
  }
  http.end();
}

void loop() {
  float voltage = readVoltage();
  float z = updateBaselineAndGetZ(voltage);
  bool anomaly = (sampleCount > BASELINE_WARMUP_N) && (fabs(z) > Z_THRESHOLD);

  Serial.printf("Voltage: %.3f V | z-score: %.2f | anomaly: %s\n",
                voltage, z, anomaly ? "YES" : "no");

  sendToThingSpeak(voltage, anomaly);

  delay(SAMPLE_INTERVAL_MS);
}
