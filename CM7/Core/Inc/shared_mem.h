#ifndef __SHARED_MEM_CM7_H
#define __SHARED_MEM_CM7_H

#include <stdint.h>
#include <stdbool.h>
#include "stm32h745xx.h"

#define SHARED_FRAMES_COUNT 256u

typedef struct {
    int16_t x;
    int16_t y;
    int16_t z;
    uint32_t ts;
} sensor_frame_t;

typedef struct {
    volatile uint32_t head;
    volatile uint32_t tail;
    sensor_frame_t frames[SHARED_FRAMES_COUNT];
} shared_ring_t;

/* Instance is defined by CM4 in .shared_ram, we just extern it here */
extern volatile shared_ring_t shared_ring;

static inline bool shared_pop_frame_cm7(sensor_frame_t *out)
{
    if (!out) return false;
    uint32_t tail = shared_ring.tail;
    uint32_t head = shared_ring.head;
    if (head == tail) return false;
    *out = shared_ring.frames[tail];
    __DSB();
    shared_ring.tail = (tail + 1) % SHARED_FRAMES_COUNT;
    return true;
}

#endif /* __SHARED_MEM_CM7_H */


