/* One capture interrupt writes audio chunks. The main loop reads them. */
#ifndef DG_AUDIO_H
#define DG_AUDIO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DG_AUDIO_CHUNK_SAMPLES
#ifdef DG_MFCC_HOP
#define DG_AUDIO_CHUNK_SAMPLES DG_MFCC_HOP
#else
#define DG_AUDIO_CHUNK_SAMPLES 320 /* 20 ms at 16 kHz */
#endif
#endif
#ifndef DG_AUDIO_RING_SLOTS
#define DG_AUDIO_RING_SLOTS 32 /* 31 queued chunks and one slot for the writer. */
#endif

typedef struct {
    int16_t chunks[DG_AUDIO_RING_SLOTS][DG_AUDIO_CHUNK_SAMPLES];
    volatile uint8_t write_index, read_index;
    volatile uint32_t dropped_chunks;
} dg_audio_ring_t;

/* Reset the ring before capture starts. */
void dg_audio_ring_init(dg_audio_ring_t *ring);

/* Capture interrupt: reserve a chunk, fill it, then commit it.
 * Fill exactly DG_AUDIO_CHUNK_SAMPLES samples before each commit.
 * Reserve always returns a writable chunk, even when the ring is full.
 * If the ring is full at commit, the new chunk is dropped and counted. */
int16_t *dg_audio_ring_reserve(dg_audio_ring_t *ring);
void dg_audio_ring_commit(dg_audio_ring_t *ring);

/* Main loop: peek at the oldest chunk (NULL if empty), then release it
 * after reading. The returned samples stay valid until release. */
const int16_t *dg_audio_ring_peek(const dg_audio_ring_t *ring);
void dg_audio_ring_release(dg_audio_ring_t *ring);

#ifdef __cplusplus
}
#endif
#endif
