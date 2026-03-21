/**
 * @file mock_csi.h
 * @brief ADR-061 Mock CSI generator for ESP32-S3 QEMU testing.
 *
 * Generates synthetic CSI frames at 20 Hz using an esp_timer, injecting
 * them directly into the edge processing pipeline via edge_enqueue_csi().
 * Ten scenarios exercise the full signal processing and edge intelligence
 * pipeline without requiring real WiFi hardware.
 *
 * Signal model per subcarrier k at time t:
 *   A_k(t) = A_base + A_person * exp(-d_k^2 / sigma^2) + noise
 *   phi_k(t) = phi_base + (2*pi*d / lambda) + breathing_mod(t) + noise
 *
 * Enable via: idf.py menuconfig -> CSI Mock Generator -> Enable
 * Or add CONFIG_CSI_MOCK_ENABLED=y to sdkconfig.defaults.
 */

#ifndef MOCK_CSI_H
#define MOCK_CSI_H

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Timing ---- */

/** Mock CSI frame interval in milliseconds (20 Hz). */
#define MOCK_CSI_INTERVAL_MS    50

/* ---- HT20 subcarrier geometry ---- */

/** Number of OFDM subcarriers for HT20 (802.11n). */
#define MOCK_N_SUBCARRIERS      52

/** I/Q data length in bytes: 52 subcarriers * 2 bytes (I + Q). */
#define MOCK_IQ_LEN             (MOCK_N_SUBCARRIERS * 2)

/* ---- Scenarios ---- */

/** Scenario identifiers for mock CSI generation. */
typedef enum {
    MOCK_SCENARIO_EMPTY         = 0,  /**< Empty room: low-noise baseline. */
    MOCK_SCENARIO_STATIC_PERSON = 1,  /**< Static person: amplitude dip, no motion. */
    MOCK_SCENARIO_WALKING       = 2,  /**< Walking person: moving reflector. */
    MOCK_SCENARIO_FALL          = 3,  /**< Fall event: abrupt phase acceleration. */
    MOCK_SCENARIO_MULTI_PERSON  = 4,  /**< Multiple people at different positions. */
    MOCK_SCENARIO_CHANNEL_SWEEP = 5,  /**< Sweep through channels 1, 6, 11, 36. */
    MOCK_SCENARIO_MAC_FILTER    = 6,  /**< Alternate correct/wrong MAC for filter test. */
    MOCK_SCENARIO_RING_OVERFLOW = 7,  /**< Burst 1000 frames rapidly to overflow ring. */
    MOCK_SCENARIO_BOUNDARY_RSSI = 8,  /**< Sweep RSSI from -90 to -10 dBm. */
    MOCK_SCENARIO_ZERO_LENGTH   = 9,  /**< Zero-length I/Q payload (error case). */

    MOCK_SCENARIO_COUNT         = 10, /**< Total number of individual scenarios. */
    MOCK_SCENARIO_ALL           = 255 /**< Meta: run all scenarios sequentially. */
} mock_scenario_t;

/* ---- State ---- */

/** Internal state for the mock CSI generator. */
typedef struct {
    uint8_t  scenario;          /**< Current active scenario. */
    uint32_t frame_count;       /**< Total frames emitted since init. */
    float    person_x;          /**< Person X position in meters (walking). */
    float    person_speed;      /**< Person movement speed in m/s. */
    float    breathing_phase;   /**< Breathing oscillator phase in radians. */
    float    person2_x;        /**< Second person X position (multi-person). */
    float    person2_speed;    /**< Second person movement speed. */
    uint8_t  channel_idx;       /**< Index into channel sweep table. */
    int8_t   rssi_sweep;        /**< Current RSSI for boundary sweep. */
    int64_t  scenario_start_ms; /**< Timestamp when current scenario started. */
    uint8_t  all_idx;           /**< Current scenario index in SCENARIO_ALL mode. */
} mock_state_t;

/**
 * Initialize and start the mock CSI generator.
 *
 * Creates a periodic esp_timer that fires every MOCK_CSI_INTERVAL_MS
 * and injects synthetic CSI frames into edge_enqueue_csi().
 *
 * @param scenario  Scenario to run (0-9), or MOCK_SCENARIO_ALL (255)
 *                  to run all scenarios sequentially.
 * @return ESP_OK on success, ESP_ERR_INVALID_STATE if already running.
 */
esp_err_t mock_csi_init(uint8_t scenario);

/**
 * Stop and destroy the mock CSI timer.
 *
 * Safe to call even if the timer is not running.
 */
void mock_csi_stop(void);

/**
 * Get the total number of mock frames emitted since init.
 *
 * @return Frame count (useful for test validation).
 */
uint32_t mock_csi_get_frame_count(void);

#ifdef __cplusplus
}
#endif

#endif /* MOCK_CSI_H */
