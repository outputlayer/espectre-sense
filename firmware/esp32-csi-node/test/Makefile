# Makefile for ESP32 CSI firmware fuzz testing targets (ADR-061 Layer 6).
#
# Requirements:
#   - clang with libFuzzer support (clang 6.0+)
#   - Linux or macOS (host-based fuzzing, no ESP-IDF needed)
#
# Usage:
#   make all                  # Build all fuzz targets
#   make fuzz_serialize       # Build serialize target only
#   make fuzz_edge            # Build edge enqueue target only
#   make fuzz_nvs             # Build NVS config target only
#   make run_serialize        # Build and run serialize fuzzer (30s)
#   make run_edge             # Build and run edge fuzzer (30s)
#   make run_nvs              # Build and run NVS fuzzer (30s)
#   make run_all              # Run all fuzzers (30s each)
#   make clean                # Remove build artifacts
#
# Environment variables:
#   FUZZ_DURATION=60          # Override fuzz duration in seconds
#   FUZZ_JOBS=4               # Parallel fuzzing jobs

CC       = clang
CFLAGS   = -fsanitize=fuzzer,address,undefined -g -O1 \
           -Istubs -I../main \
           -DCONFIG_CSI_NODE_ID=1 \
           -DCONFIG_CSI_WIFI_CHANNEL=6 \
           -DCONFIG_CSI_WIFI_SSID=\"test\" \
           -DCONFIG_CSI_TARGET_IP=\"192.168.1.1\" \
           -DCONFIG_CSI_TARGET_PORT=5500 \
           -DCONFIG_ESP_WIFI_CSI_ENABLED=1 \
           -Wno-unused-function

STUBS_SRC = stubs/esp_stubs.c
MAIN_DIR  = ../main

# Default fuzz duration (seconds) and jobs
FUZZ_DURATION ?= 30
FUZZ_JOBS     ?= 1

.PHONY: all clean run_serialize run_edge run_nvs run_all

all: fuzz_serialize fuzz_edge fuzz_nvs

# --- Serialize fuzzer ---
# Tests csi_serialize_frame() with random wifi_csi_info_t inputs.
# Links against the real csi_collector.c (with stubs for ESP-IDF).
fuzz_serialize: fuzz_csi_serialize.c $(MAIN_DIR)/csi_collector.c $(STUBS_SRC)
	$(CC) $(CFLAGS) $^ -o $@ -lm

# --- Edge enqueue fuzzer ---
# Tests the SPSC ring buffer push/pop logic with rapid-fire enqueues.
# Self-contained: reproduces ring buffer logic from edge_processing.c.
fuzz_edge: fuzz_edge_enqueue.c $(STUBS_SRC)
	$(CC) $(CFLAGS) $^ -o $@ -lm

# --- NVS config validation fuzzer ---
# Tests all NVS config validation ranges with random values.
# Self-contained: reproduces validation logic from nvs_config.c.
fuzz_nvs: fuzz_nvs_config.c $(STUBS_SRC)
	$(CC) $(CFLAGS) $^ -o $@ -lm

# --- Run targets ---
run_serialize: fuzz_serialize
	@mkdir -p corpus_serialize
	./fuzz_serialize corpus_serialize/ -max_total_time=$(FUZZ_DURATION) -max_len=2048 -jobs=$(FUZZ_JOBS)

run_edge: fuzz_edge
	@mkdir -p corpus_edge
	./fuzz_edge corpus_edge/ -max_total_time=$(FUZZ_DURATION) -max_len=4096 -jobs=$(FUZZ_JOBS)

run_nvs: fuzz_nvs
	@mkdir -p corpus_nvs
	./fuzz_nvs corpus_nvs/ -max_total_time=$(FUZZ_DURATION) -max_len=256 -jobs=$(FUZZ_JOBS)

run_all: run_serialize run_edge run_nvs

clean:
	rm -f fuzz_serialize fuzz_edge fuzz_nvs
	rm -rf corpus_serialize/ corpus_edge/ corpus_nvs/
