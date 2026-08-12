# 🤖 AI Integrated Smart Home

> **AI-powered security, environmental monitoring and home automation using ESP32, Computer Vision, MQTT and Node-RED.**

## 📌 Project Overview

**AI Integrated Smart Home** is a locally hosted AI + IoT system designed to combine **intelligent security, environmental monitoring and automated home control** into a single platform.

The system uses an **ESP32-CAM as a network camera**, while computationally heavier computer-vision processing is performed on a **Python server/laptop**. The processed decisions are communicated to ESP32 hardware through the **MQTT protocol**.

A **Node-RED dashboard** provides centralized monitoring and control of the connected devices.

### Core workflow

```text
ESP32-CAM
    │
    │ HTTP MJPEG Stream
    ▼
Python Processing Server
    │
    ├── MediaPipe Face Detection
    ├── Face Preprocessing
    ├── Cosine Similarity Recognition
    └── Decision Engine
            │
            │ MQTT
            ▼
      Mosquitto Broker
            │
            ▼
      ESP32 Controller
       ├── Door Servo
       ├── Fan
       ├── Lights
       └── Alarm
            │
            ▼
       Node-RED Dashboard
```

---

## ✨ Features

### 🔐 AI-Based Door Security

* Live video captured using ESP32-CAM
* Real-time face detection using MediaPipe
* Face recognition using cosine similarity
* Authorized-person verification
* Unknown/intruder detection
* Multiple-frame confirmation before access
* Automatic door unlocking using servo control
* Door remains locked for unauthorized users

### 🌡️ Environmental Monitoring

The system integrates multiple sensors including:

* **DHT11** — temperature/humidity monitoring
* **MQ135** — air-quality/gas monitoring
* **PIR** — motion detection

### 🚨 Emergency Automation

When a dangerous environmental condition is detected, the system can automatically trigger predefined actions such as:

```text
Hazard detected
      ↓
MQTT event
      ↓
ESP32 Controller
      ↓
Fan / Door / Alarm
      ↓
Node-RED Alert
```

### 🏠 Home Automation

The system provides control of:

* 💡 Lights
* 🌀 Exhaust/fan
* 🔒 Door lock
* 🚨 Alarm
* 📡 Sensors

through the Node-RED dashboard.

---

# 🧠 AI Pipeline

The face-recognition pipeline is implemented on the Python processing server.

```text
ESP32-CAM Frame
       ↓
JPEG/MJPEG decoding
       ↓
MediaPipe Face Detection
       ↓
Face Cropping
       ↓
Grayscale Conversion
       ↓
Resize to 100 × 100
       ↓
Normalization
       ↓
Face Vector
       ↓
Cosine Similarity
       ↓
Best Match
       ↓
Decision
```

### Recognition Logic

The system compares the processed face vector with previously stored face encodings.

Cosine similarity is calculated as:

```text
             A · B
Similarity = ───────
             |A||B|
```

A higher similarity score indicates greater similarity between the vectors.

The system uses:

```python
CONFIDENCE_THRESHOLD = 0.75
UNKNOWN_PERSON_THRESHOLD = 0.65
REQUIRED_MATCHES = 3
```

Multiple consecutive matches are required before granting access to reduce accidental recognition.

---

# 📡 MQTT Communication

MQTT is used as the communication layer between the Python AI server, Node-RED and ESP32 controllers.

### Example

Python publishes:

```text
Topic: home/lock
Message: true
```

The ESP32 subscribes to:

```text
home/lock
```

and activates the door servo when the appropriate command is received.

Face-recognition results are also published through:

```text
smartdoor/face
```

Example payload:

```json
{
  "detected": true,
  "name": "Shivam",
  "confidence": 0.82,
  "category": "allowed",
  "status": "access_granted"
}
```

---

# 🧵 Multithreaded Architecture

The Python server separates major operations into independent threads.

### 1. Frame Capture Thread

Receives the ESP32-CAM stream and extracts JPEG frames.

### 2. Processing Thread

Runs:

* Face detection
* Face preprocessing
* Recognition
* Decision logic

### 3. HTTP Proxy/Stream Thread

Serves the processed video stream to Node-RED.

### 4. Monitoring Thread

Tracks:

* Connection status
* Processed frames
* Camera availability
* System health

This architecture prevents video capture, AI processing and streaming from blocking each other.

---

# 🌐 Network Architecture

The system operates primarily over a local Wi-Fi network.

```text
                 Wi-Fi Network
                      │
        ┌─────────────┴─────────────┐
        │                           │
   ESP32-CAM                   Laptop / Server
        │                           │
   MJPEG Stream ───────────────► Python AI
                                    │
                                    ▼
                              MQTT Broker
                               Mosquitto
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                     ESP32 Controller     Node-RED
                          │                   │
                     Actuators/Sensors    Dashboard
```

The camera itself performs **video capture and streaming**. The AI processing is performed on the laptop/Python server rather than directly on the ESP32-CAM.

---

# 🛠️ Technologies Used

| Category          | Technology        |
| ----------------- | ----------------- |
| Microcontroller   | ESP32             |
| Camera            | ESP32-CAM         |
| Programming       | Python, C/C++     |
| Computer Vision   | OpenCV            |
| Face Detection    | MediaPipe         |
| Face Recognition  | Cosine Similarity |
| IoT Communication | MQTT              |
| MQTT Broker       | Mosquitto         |
| Automation        | Node-RED          |
| Sensors           | MQ135, DHT11, PIR |
| Actuator          | Servo Motor       |
| Video Protocol    | HTTP MJPEG        |
| Networking        | Wi-Fi             |

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Integrated-Smart-Home.git
cd AI-Integrated-Smart-Home
```

## 2. Create Python environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure ESP32-CAM

Update the ESP32-CAM IP address in the Python configuration:

```python
ESP32_IP = "YOUR_ESP32_CAM_IP"
```

## 5. Configure MQTT

Update:

```python
MQTT_BROKER = "YOUR_MQTT_BROKER_IP"
MQTT_PORT = 1883
```

## 6. Start Mosquitto

Make sure the Mosquitto MQTT broker is running before starting the Python server.

## 7. Start Node-RED

```bash
node-red
```

Open:

```text
http://localhost:1880
```

## 8. Start the AI server

```bash
python src/esp32_proxy_server_realtime_debug.py
```

The processed stream will be available at:

```text
http://localhost:8082/stream
```

---

# 📊 Performance

During testing, the prototype achieved approximately:

| Parameter                 | Result               |
| ------------------------- | -------------------- |
| Face recognition accuracy | ~70–75%              |
| Response time             | ~0.5–0.7 seconds     |
| Communication             | Wi-Fi + MQTT         |
| Processing                | Laptop/Python server |
| Camera                    | ESP32-CAM            |

> Performance depends on lighting, camera positioning, network conditions and the quality of the stored face samples.

---

# 🔒 Security Considerations

This project is a prototype and should not be considered a production-grade access-control system.

Future security improvements include:

* Liveness detection
* Stronger face embeddings
* Encrypted MQTT communication
* MQTT authentication
* Secure credential storage
* Event logging
* Fail-safe door control
* Anti-spoofing mechanisms

**Do not commit real face images, biometric encodings, passwords, Wi-Fi credentials or private IP configurations to a public repository.**

---

# 🚀 Future Improvements

### AI

* FaceNet / ArcFace embeddings
* Improved recognition accuracy
* Liveness detection
* Intruder tracking
* Object detection

### Edge AI

Move the AI processing from the laptop to an edge-computing platform such as:

* Raspberry Pi
* NVIDIA Jetson
* Other AI-capable edge devices

### IoT

* MQTT TLS
* Authentication
* Mobile notifications
* Cloud logging
* Remote monitoring

### Automation

* Predictive automation
* Energy optimization
* Voice control
* Advanced emergency response

---

# 👨‍💻 Team

**Varad Jaiswal**
**Shivam Raut**

**Class:** 2U2 — Electronics Engineering, 2nd Year

**Institution:**
Shri Sant Gajanan Maharaj College of Engineering, Shegaon

---

# 🎓 Project Type

**Academic Team Project**

### Domains

`Artificial Intelligence` · `Computer Vision` · `IoT` · `Embedded Systems` · `Home Automation` · `MQTT` · `Edge Computing`

---

## ⭐ Project Concept

> **Capture → Analyze → Decide → Communicate → Automate**

The project demonstrates how **AI and embedded IoT systems can work together as a distributed real-time automation system**.

---
