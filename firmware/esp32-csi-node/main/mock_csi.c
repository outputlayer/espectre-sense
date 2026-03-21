/**
 * @file mock_csi.c
 * @brief ADR-061 Mock CSI generator for ESP32-S3 QEMU testing.
 *
 * Generates synthetic CSI frames at 20 Hz using an esp_timer callback,
 * injecting them directly into the edge processing pipeline. This allows
 * full-stack testing of the CSI signal processing, vitals extraction,
 * and presence detection pipeline under QEMU without WiFi hardware.
 *
 * Signal model per subcarrier k at time t:
 *   A_k(t) = A_base + A_person * exp(-d_k^2 / sigma^2) + noise
 *   phi_k(t) = phi_base + (2*pi*d / lambda) + breathing_mod(t) + noise
 *
 * The entire file is guarded by CONFIG_CSI_MOCK_ENABLED so it compiles
 * to nothing on production builds.
 */

#include "sdkconfig.h"

#ifdef CONFIG_CSI_MOCK_ENABLED

#include "mock_csi.h"
#include "edge_processing.h"
#include "nvs_config.h"

#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"

static const char *TAG = "mock_csi";

/* ---- Configuration defaults ---- */

/** Scenario duration in ms. Kconfig-overridable. */
#ifndef CONFIG_CSI_MOCK_SCENARIO_DURATION_MS
#define CONFIG_CSI_MOCK_SCENARIO_DURATION_MS 5000
#endif

/* ---- Physical constants ---- */

#define SPEED_OF_LIGHT_MHZ  300.0f   /**< c in m * MHz (simplified). */
#define FREQ_CH6_MHZ        2437.0f  /**< Center frequency of WiFi channel 6. */
#define LAMBDA_CH6          (SPEED_OF_LIGHT_MHZ / FREQ_CH6_MHZ)  /**< ~0.123 m */

/** Breathing rate: ~15 breaths/min = 0.25 Hz. */
#define BREATHING_FREQ_HZ   0.25f

/** Breathing modulation amplitude in radians. */
#define BREATHING_AMP_RAD   0.3f

/** Walking speed in m/s. */
#define WALK_SPEED_MS       1.0f

/** Room width for position wrapping (meters). */
#define ROOM_WIDTH_M        6.0f

/** Gaussian sigma for person influence on subcarriers. */
#define PERSON_SIGMA        8.0f

/** Base amplitude for all subcarriers. */
#define A_BASE              80.0f

/** Person-induced amplitude perturbation. */
#define A_PERSON            40.0f

/** Noise amplitude (peak). */
#define NOISE_AMP           3.0f

/** Phase noise amplitude (radians). */
#define PHASE_NOISE_AMP     0.05f

/** Number of frames in the ring overflow burst (scenario 7). */
#define OVERFLOW_BURST_COUNT 1000

/** Fall detection: number of frames with abrupt phase jump. */
#define FALL_FRAME_COUNT    5

/** Fall phase acceleration magnitude (radians). */
#define FALL_PHASE_JUMP     3.14f

/** Pi constant. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Channel sweep table ---- */

static const uint8_t s_sweep_channels[] = {1, 6, 11, 36};
#define SWEEP_CHANNEL_COUNT (sizeof(s_sweep_channels) / sizeof(s_sweep_channels[0]))

/* ---- MAC addresses for filter test ---- */

/** "Correct" MAC that matches a typical filter_mac. */
static const uint8_t s_good_mac[6] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};

/** "Wrong" MAC that should be rejected by the filter. */
static const uint8_t s_bad_mac[6] __attribute__((unused)) = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};

/* ---- LFSR pseudo-random number generator ---- */

/**
 * 32-bit Galois LFSR for deterministic pseudo-random noise.
 * Avoids stdlib rand() which may not be available on ESP32 bare-metal.
 * Taps: bits 32, 31, 29, 1 (Galois LFSR polynomial 0xD0000001).
 */
static uint32_t s_lfsr = 0xDEADBEEF;

static uint32_t lfsr_next(void)
{
    uint32_t lsb = s_lfsr & 1u;
    s_lfsr >>= 1;
    if (lsb) {
        s_lfsr ^= 0xD0000001u;  /* x^32 + x^31 + x^29 + x^1 */
    }
    return s_lfsr;
}

/**
 * Return a pseudo-random float in [-1.0, +1.0].
 */
static float lfsr_float(void)
{
    uint32_t r = lfsr_next();
    /* Map [0, 65535] to [-1.0, +1.0] using 65535/2 = 32767.5 */
    return ((float)(r & 0xFFFF) / 32768.0f) - 1.0f;
}

/* ---- Module state ---- */

static mock_state_t  s_state;
static esp_timer_handle_t s_timer = NULL;

/** Tracks whether the MAC filter has been set up in gen_mac_filter. */
static bool s_mac_filter_initialized = false;

/** Tracks whether the overflow burst has fired in gen_ring_overflow. */
static bool s_overflow_burst_done = false;

/* External NVS config (for MAC filter scenario). */
extern nvs_config_t g_nvs_config;

/* ---- Helper: compute channel frequency ---- */

static uint32_t channel_to_freq_mhz(uint8_t channel)
{
    if (channel >= 1 && channel <= 13) {
        return 2412 + (channel - 1) * 5;
    } else if (channel == 14) {
        return 2484;
    } else if (channel >= 36 && channel <= 177) {
        return 5000 + channel * 5;
    }
    return 2437;  /* Default to ch 6. */
}

/* ---- Helper: compute wavelength for a channel ---- */

static float channel_to_lambda(uint8_t channel)
{
    float freq = (float)channel_to_freq_mhz(channel);
    return SPEED_OF_LIGHT_MHZ / freq;
}

/* ---- Helper: elapsed ms since scenario start ---- */

static int64_t scenario_elapsed_ms(void)
{
    int64_t now = esp_timer_get_time() / 1000;
    return now - s_state.scenario_start_ms;
}

/* ---- Helper: clamp int8 ---- */

static int8_t clamp_i8(int32_t val)
{
    if (val < -128) return -128;
    if (val >  127) return  127;
    return (int8_t)val;
}

/* ---- Core signal generation ---- */

/**
 * Generate one I/Q frame for a single person at position person_x.
 *
 * @param iq_buf       Output buffer (MOCK_IQ_LEN bytes).
 * @param person_x     Person X position in meters.
 * @param breathing    Breathing phase in radians.
 * @param has_person   Whether a person is present.
 * @param lambda       Wavelength in meters.
 */
static void generate_person_iq(uint8_t *iq_buf, float person_x,
                                float breathing, bool has_person,
                                float lambda)
{
    for (int k = 0; k < MOCK_N_SUBCARRIERS; k++) {
        /* Distance of subcarrier k's spatial sample from person. */
        float d_k = (float)k - person_x * (MOCK_N_SUBCARRIERS / ROOM_WIDTH_M);

        /* Amplitude model. */
        float amp = A_BASE;
        if (has_person) {
            float gauss = expf(-(d_k * d_k) / (2.0f * PERSON_SIGMA * PERSON_SIGMA));
            amp += A_PERSON * gauss;
        }
        amp += NOISE_AMP * lfsr_float();

        /* Phase model. */
        float phase = (float)k * 0.1f;  /* Base phase gradient. */
        if (has_person) {
            float d_meters = fabsf(d_k) * (ROOM_WIDTH_M / MOCK_N_SUBCARRIERS);
            phase += (2.0f * M_PI * d_meters) / lambda;
            phase += BREATHING_AMP_RAD * sinf(breathing);
        }
        phase += PHASE_NOISE_AMP * lfsr_float();

        /* Convert to I/Q (int8). */
        float i_f = amp * cosf(phase);
        float q_f = amp * sinf(phase);

        iq_buf[k * 2]     = (uint8_t)clamp_i8((int32_t)i_f);
        iq_buf[k * 2 + 1] = (uint8_t)clamp_i8((int32_t)q_f);
    }
}

/* ---- Scenario generators ---- */

/**
 * Scenario 0: Empty room.
 * Low-amplitude noise on all subcarriers, no person present.
 */
static void gen_empty(uint8_t *iq_buf, uint8_t *channel, int8_t *rssi)
{
    generate_person_iq(iq_buf, 0.0f, 0.0f, false, LAMBDA_CH6);
    *channel = 6;
    *rssi = -60;
}

/**
 * Scenario 1: Static person.
 * Person at fixed position with breathing modulation.
 */
static void gen_static_person(uint8_t *iq_buf, uint8_t *channel, int8_t *rssi)
{
    s_state.breathing_phase += 2.0f * M_PI * BREATHING_FREQ_HZ
                               * (MOCK_CSI_INTERVAL_MS / 1000.0f);
    if (s_state.breathing_phase > 2.0f * M_PI) {
        s_state.breathing_phase -= 2.0f * M_PI;
    }

    generate_person_iq(iq_buf, 3.0f, s_state.breathing_phase, true, LAMBDA_CH6);
    *channel = 6;
    *rssi = -45;
}

/**
 * Scenario 2: Walking person.
 * Person moves across the room and wraps around.
 */
static void gen_walking(uint8_t *iq_buf, uint8_t *channel, int8_t *rssi)
{
    s_state.breathing_phase += 2.0f * M_PI * BREATHING_FREQ_HZ
                               * (MOCK_CSI_INTERVAL_MS / 1000.0f);
    if (s_state.breathing_phase > 2.0f * M_PI) {
        s_state.breathing_phase -= 2.0f * M_PI;
    }

    s_state.person_x += s_state.person_speed * (MOCK_CSI_INTERVAL_MS / 1000.0f);
    if (s_state.person_x > ROOM_WIDTH_M) {
        s_state.person_x -= ROOM_WIDTH_M;
    }

    generate_person_iq(iq_buf, s_state.person_x, s_state.breathing_phase,
                       true, LAMBDA_CH6);
    *channel = 6;
    *rssi = -40;
}

/**
 * Scenario 3: Fall event.
 * Normal walking for most frames, then an abrupt phase discontinuity
 * simulating a fall (rapid vertical displacement).
 */
static void gen_fall(uint8_t *iq_buf, uint8_t *channel, int8_t *rssi)
{
    int64_t elapsed = scenario_elapsed_ms();
    uint32_t duration = CONFIG_CSI_MOCK_SCENARIO_DURATION_MS;

    /* Fall occurs at 70% of scenario duration. */
    uint32_t fall_start = (duration * 70) / 100;
    uint32_t fall_end   = fall_start + (FALL_FRAME_COUNT * MOCK_CSI_INTERVAL_MS);

    s_state.breathing_phase += 2.0f * M_PI * BREATHING_FREQ_HZ
                               * (MOCK_CSI_INTERVAL_MS / 1000.0f);

    s_state.person_x += 0.5f * (MOCK_CSI_INTERVAL_MS / 1000.0f);
    if (s_state.person_x > ROOM_WIDTH_M) {
        s_state.person_x = ROOM_WIDTH_M;
    }

    float extra_phase = 0.0f;
    if (elapsed >= fall_start && elapsed < fall_end) {
        /* Abrupt phase jump simulating rapid downward motion. */
        extra_phase = FALL_PHASE_JUMP;
    }

    /* Build I/Q with fall perturbation. */
    float lambda = LAMBDA_CH6;
    for (int k = 0; k < MOCK_N_SUBCARRIERS; k++) {
        float d_k = (float)k - s_state.person_x * (MOCK_N_SUBCARRIERS / ROOM_WIDTH_M);
        float gauss = expf(-(d_k * d_k) / (2.0f * PERSON_SIGMA * PERSON_SIGMA));

        float amp = A_BASE + A_PERSON * gauss + NOISE_AMP * lfsr_float();

        float d_meters = fabsf(d_k) * (ROOM_WIDTH_M / MOCK_N_SUBCARRIERS);
        float phase = (float)k * 0.1f
                    + (2.0f * M_PI * d_meters) / lambda
                    + BREATHING_AMP_RAD * sinf(s_state.breathing_phase)
                    + extra_phase * gauss  /* Fall affects nearby subcarriers. */
                    + PHASE_NOISE_AMP * lfsr_float();

        iq_buf[k * 2]     = (uint8_t)clamp_i8((int32_t)(amp * cosf(phase)));
        iq_buf[k * 2 + 1] = (uint8_t)clamp_i8((int32_t)(amp * sinf(phase)));
    }

    *channel = 6;
    *rssi = -42;
}

/**
 * Scenario 4: Multiple people.
 * Two people at different positions with independent breathing.
 */
static void gen_multi_person(uint8_t *iq_buf, uint8_t *channel, int8_t *rssi)
{
    float dt = MOCK_CSI_INTERVAL_MS / 1000.0f;

    s_state.breathing_phase += 2.0f * M_PI * BREATHING_FREQ_HZ * dt;
    float breathing2 = s_state.breathing_phase * 1.3f;  /* Slightly different rate. */

    s_state.person_x  += s_state.person_speed * dt;
    s_state.person2_x += s_state.person2_speed * dt;

    /* Wrap positions. */
    if (s_state.person_x > ROOM_WIDTH_M) s_state.person_x -= ROOM_WIDTH_M;
    if (s_state.person2_x > ROOM_WIDTH_M) s_state.person2_x -= ROOM_WIDTH_M;

    float lambda = LAMBDA_CH6;

    for (int k = 0; k < MOCK_N_SUBCARRIERS; k++) {
        /* Superpose contributions from both people. */
        float d1 = (float)k - s_state.person_x * (MOCK_N_SUBCARRIERS / ROOM_WIDTH_M);
        float d2 = (float)k - s_state.person2_x * (MOCK_N_SUBCARRIERS / ROOM_WIDTH_M);

        float g1 = expf(-(d1 * d1) / (2.0f * PERSON_SIGMA * PERSON_SIGMA));
        float g2 = expf(-(d2 * d2) / (2.0f * PERSON_SIGMA * PERSON_SIGMA));

        float amp = A_BASE + A_PERSON * g1 + (A_PERSON * 0.7f) * g2
                  + NOISE_AMP * lfsr_float();

        float dm1 = fabsf(d1) * (ROOM_WIDTH_M / MOCK_N_SUBCARRIERS);
        float dm2 = fabsf(d2) * (ROOM_WIDTH_M / MOCK_N_SUBCARRIERS);

        float phase = (float)k * 0.1f
                    + (2.0f * M_PI * dm1) / lambda * g1
                    + (2.0f * M_PI * dm2) / lambda * g2
                    + BREATHING_AMP_RAD * sinf(s_state.breathing_phase) * g1
                    + BREATHING_AMP_RAD * sinf(breathing2) * g2
                    + PHASE_NOISE_AMP * lfsr_float();

        iq_buf[k * 2]     = (uint8_t)clamp_i8((int32_t)(amp * cosf(phase)));
        iq_buf[k * 2 + 1] = (uint8_t)clamp_i8((int32_t)(amp * sinf(phase)));
    }

    *channel = 6;
    *rssi = -38;
}

/**
 * Scenario 5: Channel sweep.
 * Cycles through channels 1, 6, 11, 36 every 20 frames.
 */
static void gen_channel_sweep(uint8_t *iq_buf, uint8_t *channel, int8_t *rssi)
{
    /* Switch channel every 20 frames (1 second at 20 Hz). */
    if ((s_state.frame_count % 20) == 0 && s_state.frame_count > 0) {
        s_state.channel_idx = (s_state.channel_idx + 1) % SWEEP_CHANNEL_COUNT;
    }

    uint8_t ch = s_sweep_channels[s_state.channel_idx];
    float lambda = channel_to_lambda(ch);

    generate_person_iq(iq_buf, 3.0f, 0.0f, true, lambda);
    *channel = ch;
    *rssi = -50;
}

/**
 * Scenario 6: MAC filter test.
 * Alternates between a "good" MAC (should pass filter) and a "bad" MAC
 * (should be rejected). Even frames use good MAC, odd frames use bad MAC.
 *
 * Note: Since we inject via edge_enqueue_csi() which bypasses the MAC
 * filter (that happens in wifi_csi_callback), this scenario instead
 * sets/clears the NVS filter_mac and logs which frames would pass.
 * The test harness can verify frame_count vs expected.
 */
static void gen_mac_filter(uint8_t *iq_buf, uint8_t *channel, int8_t *rssi,
                           bool *skip_inject)
{
    /* Set up the filter MAC to match s_good_mac on first frame of this scenario. */
    if (!s_mac_filter_initialized) {
        memcpy(g_nvs_config.filter_mac, s_good_mac, 6);
        g_nvs_config.filter_mac_set = 1;
        s_mac_filter_initialized = true;
        ESP_LOGI(TAG, "MAC filter scenario: filter set to %02X:%02X:%02X:%02X:%02X:%02X",
                 s_good_mac[0], s_good_mac[1], s_good_mac[2],
                 s_good_mac[3], s_good_mac[4], s_good_mac[5]);
    }

    generate_person_iq(iq_buf, 3.0f, 0.0f, true, LAMBDA_CH6);
    *channel = 6;
    *rssi = -50;

    /* Odd frames: simulate "wrong" MAC by skipping injection. */
    if ((s_state.frame_count & 1) != 0) {
        *skip_inject = true;
        ESP_LOGD(TAG, "MAC filter: frame %lu skipped (bad MAC)",
                 (unsigned long)s_state.frame_count);
    } else {
        *skip_inject = false;
    }
}

/**
 * Scenario 7: Ring buffer overflow.
 * Burst OVERFLOW_BURST_COUNT frames as fast as possible to test
 * the SPSC ring buffer's overflow handling.
 */
static void gen_ring_overflow(uint8_t *iq_buf, uint8_t *channel, int8_t *rssi,
                              uint16_t *burst_count)
{
    generate_person_iq(iq_buf, 3.0f, 0.0f, true, LAMBDA_CH6);
    *channel = 6;
    *rssi = -50;

    /* Burst once on the first timer tick of this scenario. */
    if (!s_overflow_burst_done) {
        *burst_count = OVERFLOW_BURST_COUNT;
        s_overflow_burst_done = true;
    } else {
        *burst_count = 1;
    }
}

/**
 * Scenario 8: Boundary RSSI sweep.
 * Sweeps RSSI from -90 dBm to -10 dBm linearly over the scenario duration.
 */
static void gen_boundary_rssi(uint8_t *iq_buf, uint8_t *channel, int8_t *rssi)
{
    int64_t elapsed = scenario_elapsed_ms();
    uint32_t duration = CONFIG_CSI_MOCK_SCENARIO_DURATION_MS;

    /* Linear sweep: -90 to -10 dBm. */
    float frac = (float)elapsed / (float)duration;
    if (frac > 1.0f) frac = 1.0f;
    int8_t sweep_rssi = (int8_t)(-90.0f + 80.0f * frac);

    generate_person_iq(iq_buf, 3.0f, 0.0f, true, LAMBDA_CH6);
    *channel = 6;
    *rssi = sweep_rssi;
}

/**
 * Scenario 9: Zero-length I/Q.
 * Injects a frame with iq_len = 0 to test error handling.
 */
/* Handled inline in the timer callback. */

/* ---- Scenario transition ---- */

/**
 * Advance to the next scenario when running SCENARIO_ALL.
 */
/** Flag: set when all scenarios are done so timer callback exits early. */
static bool s_all_done = false;

static void advance_scenario(void)
{
    s_state.all_idx++;
    if (s_state.all_idx >= MOCK_SCENARIO_COUNT) {
        ESP_LOGI(TAG, "All %d scenarios complete (%lu total frames)",
                 MOCK_SCENARIO_COUNT, (unsigned long)s_state.frame_count);
        s_all_done = true;
        return;  /* Stop generating — timer callback will check s_all_done. */
    }

    s_state.scenario = s_state.all_idx;
    s_state.scenario_start_ms = esp_timer_get_time() / 1000;

    /* Reset per-scenario state. */
    s_state.person_x = 1.0f;
    s_state.person_speed = WALK_SPEED_MS;
    s_state.person2_x = 4.0f;
    s_state.person2_speed = WALK_SPEED_MS * 0.6f;
    s_state.breathing_phase = 0.0f;
    s_state.channel_idx = 0;
    s_state.rssi_sweep = -90;

    ESP_LOGI(TAG, "=== Scenario %u started ===", (unsigned)s_state.scenario);
}

/* ---- Timer callback ---- */

static void mock_timer_cb(void *arg)
{
    (void)arg;

    /* All scenarios finished — stop generating. */
    if (s_all_done) {
        return;
    }

    /* Check for scenario timeout in SCENARIO_ALL mode. */
    if (s_state.scenario == MOCK_SCENARIO_ALL ||
        (s_state.all_idx > 0 && s_state.all_idx < MOCK_SCENARIO_COUNT)) {
        /* We're running in sequential mode. */
        int64_t elapsed = scenario_elapsed_ms();
        if (elapsed >= CONFIG_CSI_MOCK_SCENARIO_DURATION_MS) {
            advance_scenario();
        }
    }

    uint8_t  iq_buf[MOCK_IQ_LEN];
    uint8_t  channel = 6;
    int8_t   rssi = -50;
    uint16_t iq_len = MOCK_IQ_LEN;
    uint16_t burst = 1;
    bool     skip = false;

    uint8_t active_scenario = s_state.scenario;

    switch (active_scenario) {
    case MOCK_SCENARIO_EMPTY:
        gen_empty(iq_buf, &channel, &rssi);
        break;

    case MOCK_SCENARIO_STATIC_PERSON:
        gen_static_person(iq_buf, &channel, &rssi);
        break;

    case MOCK_SCENARIO_WALKING:
        gen_walking(iq_buf, &channel, &rssi);
        break;

    case MOCK_SCENARIO_FALL:
        gen_fall(iq_buf, &channel, &rssi);
        break;

    case MOCK_SCENARIO_MULTI_PERSON:
        gen_multi_person(iq_buf, &channel, &rssi);
        break;

    case MOCK_SCENARIO_CHANNEL_SWEEP:
        gen_channel_sweep(iq_buf, &channel, &rssi);
        break;

    case MOCK_SCENARIO_MAC_FILTER:
        gen_mac_filter(iq_buf, &channel, &rssi, &skip);
        break;

    case MOCK_SCENARIO_RING_OVERFLOW:
        gen_ring_overflow(iq_buf, &channel, &rssi, &burst);
        break;

    case MOCK_SCENARIO_BOUNDARY_RSSI:
        gen_boundary_rssi(iq_buf, &channel, &rssi);
        break;

    case MOCK_SCENARIO_ZERO_LENGTH:
        /* Deliberately inject zero-length data to test error path. */
        iq_len = 0;
        memset(iq_buf, 0, sizeof(iq_buf));
        break;

    default:
        ESP_LOGW(TAG, "Unknown scenario %u, defaulting to empty", active_scenario);
        gen_empty(iq_buf, &channel, &rssi);
        break;
    }

    /* Inject frame(s) into the edge processing pipeline. */
    if (!skip) {
        for (uint16_t i = 0; i < burst; i++) {
            edge_enqueue_csi(iq_buf, iq_len, rssi, channel);
            s_state.frame_count++;
        }
    } else {
        /* Count skipped frames for MAC filter validation. */
        s_state.frame_count++;
    }

    /* Periodic logging (every 20 frames = 1 second). */
    if ((s_state.frame_count % 20) == 0) {
        ESP_LOGI(TAG, "scenario=%u frames=%lu ch=%u rssi=%d",
                 active_scenario, (unsigned long)s_state.frame_count,
                 (unsigned)channel, (int)rssi);
    }
}

/* ---- Public API ---- */

esp_err_t mock_csi_init(uint8_t scenario)
{
    if (s_timer != NULL) {
        ESP_LOGW(TAG, "Mock CSI already running");
        return ESP_ERR_INVALID_STATE;
    }

    /* Initialize state. */
    memset(&s_state, 0, sizeof(s_state));
    s_state.person_x = 1.0f;
    s_state.person_speed = WALK_SPEED_MS;
    s_state.person2_x = 4.0f;
    s_state.person2_speed = WALK_SPEED_MS * 0.6f;
    s_state.scenario_start_ms = esp_timer_get_time() / 1000;
    s_all_done = false;
    s_mac_filter_initialized = false;
    s_overflow_burst_done = false;

    /* Reset LFSR to deterministic seed. */
    s_lfsr = 0xDEADBEEF;

    if (scenario == MOCK_SCENARIO_ALL) {
        s_state.scenario = 0;
        s_state.all_idx = 0;
        ESP_LOGI(TAG, "Mock CSI: running ALL %d scenarios sequentially (%u ms each)",
                 MOCK_SCENARIO_COUNT, CONFIG_CSI_MOCK_SCENARIO_DURATION_MS);
    } else {
        s_state.scenario = scenario;
        s_state.all_idx = 0;
        ESP_LOGI(TAG, "Mock CSI: scenario=%u, interval=%u ms, duration=%u ms",
                 (unsigned)scenario, MOCK_CSI_INTERVAL_MS,
                 CONFIG_CSI_MOCK_SCENARIO_DURATION_MS);
    }

    /* Create periodic timer. */
    esp_timer_create_args_t timer_args = {
        .callback = mock_timer_cb,
        .arg      = NULL,
        .name     = "mock_csi",
    };

    esp_err_t err = esp_timer_create(&timer_args, &s_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create mock CSI timer: %s", esp_err_to_name(err));
        return err;
    }

    uint64_t period_us = (uint64_t)MOCK_CSI_INTERVAL_MS * 1000;
    err = esp_timer_start_periodic(s_timer, period_us);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start mock CSI timer: %s", esp_err_to_name(err));
        esp_timer_delete(s_timer);
        s_timer = NULL;
        return err;
    }

    ESP_LOGI(TAG, "Mock CSI generator started (20 Hz, %u subcarriers, %u bytes/frame)",
             MOCK_N_SUBCARRIERS, MOCK_IQ_LEN);
    return ESP_OK;
}

void mock_csi_stop(void)
{
    if (s_timer == NULL) {
        return;
    }

    esp_timer_stop(s_timer);
    esp_timer_delete(s_timer);
    s_timer = NULL;

    ESP_LOGI(TAG, "Mock CSI stopped after %lu frames",
             (unsigned long)s_state.frame_count);
}

uint32_t mock_csi_get_frame_count(void)
{
    return s_state.frame_count;
}

#endif /* CONFIG_CSI_MOCK_ENABLED */
