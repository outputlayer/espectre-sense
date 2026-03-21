# ESPectre

**WiFi CSI room sensing — presence detection, activity classification, vital signs & sleep monitoring with ESP32-S3.**

ESPectre uses WiFi Channel State Information (CSI) to detect people, classify their activity (empty / lying / sitting / walking), estimate heart & breathing rate, and track sleep quality. No cameras, no wearables — just WiFi signals.

## Architecture

```
ESP32-S3 Nodes (x3)          Rust Server (Docker)              Browser
┌──────────────┐        ┌───────────────────────────┐    ┌──────────────┐
│ WiFi CSI data │──UDP──→│  CSINetLight CNN (pure    │    │   Heatmap    │
│ 56 subcarrier │  5005  │  Rust, no ML frameworks)  │─WS→│   Sleep      │
│ amplitudes    │        │  ├─ 4-class presence      │4001│   Training   │
│ ~7 fps/node   │        │  ├─ Vital signs (HR, BR)  │    │              │
│               │        │  └─ Sleep quality scoring  │    │              │
└──────────────┘        └───────────────────────────┘    └──────────────┘
```

### Deep Learning Pipeline

The classification uses **CSINetLight** — a CNN trained on WiFi CSI amplitude data:

| Stage | Details |
|-------|---------|
| **Input** | 3 nodes × 56 subcarriers = 168 features × 100 frames (~4.5s window) |
| **Preprocessing** | Baseline subtraction (empty room) → normalization |
| **Model** | Conv1d(168→128) → Conv1d(128→256) → Conv1d(256→128) → AdaptiveAvgPool → Dense(128→64→4) |
| **Output** | 4 classes: `empty`, `lying`, `sitting`, `walking` |
| **Accuracy** | 100% on validation set (4433 labeled windows) |
| **Inference** | Pure Rust, zero ML dependencies, ~1ms per frame |

## Quick Start

### 1. Flash ESP32-S3 Firmware

3x ESP32-S3 boards in a triangle layout:

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash

# Write WiFi creds to NVS (never stored in code)
python provision.py --port /dev/ttyUSB0 \
  --ssid "YourWiFi" --password "YourPass" \
  --target-ip YOUR_SERVER_IP
```

### 2. Deploy Server

```bash
docker build -t espectre -f docker/Dockerfile .
docker run -d --name espectre \
  --network host \
  -v espectre-data:/app/data \
  -e CSI_SOURCE=esp32 \
  espectre
```

Or with explicit ports:

```bash
docker run -d --name espectre \
  -p 3030:4000 -p 3031:4001 -p 5005:5005/udp \
  -v espectre-data:/app/data \
  -e CSI_SOURCE=esp32 \
  espectre
```

Open: `http://YOUR_SERVER:3030/ui/heatmap.html`

### 3. (Optional) Train Your Own Model

Record labeled data via `http://YOUR_SERVER:3030/ui/train.html`:

| Activity | What to do | Duration |
|----------|-----------|----------|
| **Empty** | Leave room, close door | 3-5 min |
| **Lying** | Lie in bed, stay still | 3-5 min |
| **Sitting** | Sit at desk normally | 3-5 min |
| **Walking** | Walk around the room | 3-5 min |

Then train:

```bash
cd training
pip install -r requirements.txt

# 1. Preprocess raw CSI recordings
python prepare_data.py

# 2. Train CSINetLight CNN
python train_dl.py
```

Trained weights are exported to `server/models/csi_light_weights.json`. Rebuild Docker to deploy.

## Project Structure

```
espectre/
├── firmware/                 # ESP32-S3 firmware (ESP-IDF, C)
│   ├── main/main.c           #   CSI collection + UDP streaming
│   └── provision.py          #   Flash WiFi credentials
├── server/                   # Rust server (Axum + Tokio)
│   ├── src/main.rs           #   UDP receiver, WS, REST API, classification
│   ├── src/dl_classifier.rs  #   CSINetLight CNN inference (pure Rust)
│   ├── src/espectre.rs       #   ESPectre motion detection (legacy fallback)
│   ├── src/vital_signs.rs    #   Heart rate & breathing estimation
│   └── models/               #   Model weights & normalization params
│       ├── csi_light_weights.json
│       ├── baseline.npy
│       ├── feat_mean.npy
│       └── feat_std.npy
├── training/                 # PyTorch training pipeline
│   ├── prepare_data.py       #   JSONL → sliding windows → numpy
│   ├── csi_model.py          #   CSINet (CNN+LSTM) & CSINetLight (CNN)
│   ├── train_dl.py           #   Training loop with class balancing
│   └── train_sklearn.py      #   Legacy MLP/RF/GB training
├── ui/                       # Web dashboards (HTML/JS/Canvas)
│   ├── heatmap.html          #   Real-time heatmap + classification
│   ├── sleep.html            #   Sleep quality dashboard
│   └── train.html            #   Data recording UI
└── docker/Dockerfile         # Multi-stage build (Rust 1.85 → Debian slim)
```

## Web Dashboard

The heatmap UI shows:

- **Spatial heatmap** — IDW-interpolated motion intensity across the room
- **Classification** — DL model output with confidence (empty / lying / sitting / walking)
- **Vital signs** — heart rate & breathing rate with confidence bars
- **Node activity** — per-node motion energy bars
- **Activity timeline** — scrolling classification history
- **CSI spectrograms** — raw subcarrier amplitudes per node (expandable to fullscreen)

Everforest dark theme. WebSocket updates at ~7 fps.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health |
| `/api/v1/sensing/latest` | GET | Latest sensing data |
| `/api/v1/vital-signs` | GET | Heart rate & breathing |
| `/api/v1/sleep/history?hours=N` | GET | Sleep log |
| `/api/v1/recording/start` | POST | Start CSI recording |
| `/api/v1/recording/stop` | POST | Stop recording |
| `/api/v1/recording/list` | GET | List recordings |
| `/ws/sensing` | WS | Real-time data stream |

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `auto` | `esp32`, `simulate`, `auto` |
| `--http-port` | `4000` | HTTP API port |
| `--ws-port` | `4001` | WebSocket port |
| `--tick-ms` | `100` | Processing interval |
| `--bind-addr` | `127.0.0.1` | Bind address |
| `--ui-path` | `ui` | Static UI path |

## Hardware

- 3x ESP32-S3 dev boards (any with WiFi CSI)
- WiFi access point in the room
- Linux server or VPS (Docker, ~128MB RAM)

```
        [Node 3]
       /        \
      /   Room    \
     /              \
[Node 2] ────── [Node 1]
```

## License

MIT
