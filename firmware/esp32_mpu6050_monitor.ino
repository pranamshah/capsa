/*
  Computer-Aided Vibration Signal Analysis and Anomaly Detection
  ESP32 + MPU6050 + 3.7V battery

  Reads triaxial acceleration from MPU6050 over I2C, buffers a window, computes
  time-domain features (RMS, peak, std), reads battery voltage via ADC, and
  publishes everything over MQTT.

  Wiring:
    MPU6050 SCL -> GPIO22, SDA -> GPIO21, VCC -> 3.3V, GND -> GND
    Battery voltage (through R1/R2 divider) -> GPIO34

  Libraries (install via Arduino Library Manager):
    - MPU6050 (Electronic Cats) or I2Cdevlib MPU6050
    - PubSubClient (Nick O'Leary)

  TODO before flashing:
    - WIFI_SSID / WIFI_PASSWORD
    - MQTT_TOPIC (unique, matching mqtt_subscriber.py)
    - VOLTAGE_DIVIDER_RATIO (match your resistors)
*/

#include <Wire.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <MPU6050.h>
#include <math.h>

const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

const char* MQTT_BROKER = "broker.hivemq.com";
const int   MQTT_PORT   = 1883;
const char* MQTT_TOPIC  = "vibration/yourname/data";  // make unique, match subscriber

const int   BAT_ADC_PIN           = 34;
const float ADC_MAX_VOLTAGE       = 3.3;
const float ADC_MAX_COUNTS        = 4095.0;
const float VOLTAGE_DIVIDER_RATIO = 0.767;  // e.g. R1=10k(top), R2=33k(bottom) -> 33/43

const int   WINDOW_SIZE       = 256;   // samples per feature window
const int   SAMPLE_DELAY_US   = 2000;  // ~500 Hz sampling

MPU6050 mpu;
WiFiClient espClient;
PubSubClient mqttClient(espClient);

float axBuf[WINDOW_SIZE];

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.println(" connected: " + WiFi.localIP().toString());
}

void connectMQTT() {
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  while (!mqttClient.connected()) {
    String clientId = "esp32-vib-" + String(random(0xffff), HEX);
    if (mqttClient.connect(clientId.c_str())) {
      Serial.println("MQTT connected");
    } else {
      Serial.print("MQTT failed rc="); Serial.println(mqttClient.state());
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed! Check wiring.");
  }
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  connectWiFi();
  connectMQTT();
}

float readBatteryVoltage() {
  int raw = analogRead(BAT_ADC_PIN);
  float vAtPin = (raw / ADC_MAX_COUNTS) * ADC_MAX_VOLTAGE;
  return vAtPin / VOLTAGE_DIVIDER_RATIO;
}

// Collect a window of one accel axis (magnitude of acceleration works well too)
void collectWindow() {
  int16_t ax, ay, az, gx, gy, gz;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    // acceleration magnitude in g (MPU6050 default +/-2g -> 16384 LSB/g)
    float axg = ax / 16384.0;
    float ayg = ay / 16384.0;
    float azg = az / 16384.0;
    axBuf[i] = sqrt(axg * axg + ayg * ayg + azg * azg);
    delayMicroseconds(SAMPLE_DELAY_US);
  }
}

void computeFeatures(float &rms, float &peak, float &stddev) {
  float sumsq = 0, mean = 0, mx = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    mean += axBuf[i];
    if (axBuf[i] > mx) mx = axBuf[i];
  }
  mean /= WINDOW_SIZE;
  float varsum = 0;
  for (int i = 0; i < WINDOW_SIZE; i++) {
    sumsq += axBuf[i] * axBuf[i];
    varsum += (axBuf[i] - mean) * (axBuf[i] - mean);
  }
  rms = sqrt(sumsq / WINDOW_SIZE);
  peak = mx;
  stddev = sqrt(varsum / WINDOW_SIZE);
}

void publish(float rms, float peak, float stddev, float vbat) {
  char payload[160];
  snprintf(payload, sizeof(payload),
           "{\"rms\":%.4f,\"peak\":%.4f,\"std\":%.4f,\"vbat\":%.3f,\"ts\":%lu}",
           rms, peak, stddev, vbat, millis());
  mqttClient.publish(MQTT_TOPIC, payload);
  Serial.println(payload);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqttClient.connected()) connectMQTT();
  mqttClient.loop();

  collectWindow();
  float rms, peak, stddev;
  computeFeatures(rms, peak, stddev);
  float vbat = readBatteryVoltage();
  publish(rms, peak, stddev, vbat);
}
