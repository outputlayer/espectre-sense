#!/bin/bash
# Flash ESP32-S3 CSI Node with prebuilt firmware
# Usage: ./flash.sh /dev/ttyUSB0

PORT=${1:-/dev/ttyUSB0}

if [ ! -c "$PORT" ] && [ ! -e "$PORT" ]; then
    echo "Error: Port $PORT not found"
    echo "Usage: ./flash.sh /dev/ttyUSB0"
    echo "  macOS: ./flash.sh /dev/cu.usbserial-*"
    echo "  Linux: ./flash.sh /dev/ttyUSB0"
    echo "  Windows: ./flash.sh COM3"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Flashing ESP32-S3 CSI Node firmware to $PORT..."
python3 -m esptool --chip esp32s3 \
    -p "$PORT" \
    -b 460800 \
    --before default_reset \
    --after hard_reset \
    write_flash \
    --flash_mode dio \
    --flash_size 8MB \
    --flash_freq 80m \
    0x0     "$SCRIPT_DIR/bootloader.bin" \
    0x8000  "$SCRIPT_DIR/partition-table.bin" \
    0xf000  "$SCRIPT_DIR/ota_data_initial.bin" \
    0x20000 "$SCRIPT_DIR/esp32-csi-node.bin"

echo ""
echo "Firmware flashed! Now provision the node:"
echo "  python3 provision.py --port $PORT --ssid YOUR_WIFI --password YOUR_PASS --target-ip SERVER_IP --target-port 5005 --node-id 1"
