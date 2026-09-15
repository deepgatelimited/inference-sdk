#include "dg_audio.h"

#include <string.h>
#include <stdatomic.h>

void dg_audio_ring_init(dg_audio_ring_t *ring)
{
    memset(ring, 0, sizeof *ring);
}

int16_t *dg_audio_ring_reserve(dg_audio_ring_t *ring)
{
    return ring->chunks[ring->write_index];
}

void dg_audio_ring_commit(dg_audio_ring_t *ring)
{
    uint8_t next = (uint8_t)((ring->write_index + 1) % DG_AUDIO_RING_SLOTS);
    if (next == ring->read_index) {
        ring->dropped_chunks++;
    } else {
        /* Finish writing the samples before making this chunk visible. */
        atomic_signal_fence(memory_order_release);
        ring->write_index = next;
    }
}

const int16_t *dg_audio_ring_peek(const dg_audio_ring_t *ring)
{
    if (ring->read_index == ring->write_index) return NULL;
    atomic_signal_fence(memory_order_acquire);
    return ring->chunks[ring->read_index];
}

void dg_audio_ring_release(dg_audio_ring_t *ring)
{
    if (ring->read_index != ring->write_index) {
        /* Finish reading the samples before the interrupt can reuse this slot. */
        atomic_signal_fence(memory_order_release);
        ring->read_index = (uint8_t)((ring->read_index + 1) % DG_AUDIO_RING_SLOTS);
    }
}
