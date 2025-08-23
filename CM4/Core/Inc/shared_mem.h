#ifndef __SHARED_MEM_H
#define __SHARED_MEM_H

#include <stdint.h>
#include <stdbool.h>

#define SHARED_FRAMES_COUNT 256u

typedef struct {
    int16_t x;
    int16_t y;
    int16_t z;
    uint32_t ts; /* timestamp ms */
} sensor_frame_t;

/* Simple shared ring buffer declaration.
   If you plan to share between CM4/CM7 place this variable into a shared linker
   section (.shared_ram) mapped to D2 SRAM. For now a plain declaration works. */

typedef struct {
    volatile uint32_t head;
    volatile uint32_t tail;
    sensor_frame_t frames[SHARED_FRAMES_COUNT];
} shared_ring_t;

/* single instance, defined in shared_mem.c (placed in .shared_ram) */
extern volatile shared_ring_t shared_ring;

/* helper prototypes (optional) */
bool shared_push_frame(const sensor_frame_t *f);
bool shared_pop_frame(sensor_frame_t *out);

#endif /* __SHARED_MEM_H */
