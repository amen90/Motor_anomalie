#include "shared_mem.h"
#include "main.h"
#include "cmsis_os.h"

#if defined(__GNUC__)
#define SHARED_LINK __attribute__((section(".shared_ram"), aligned(32)))
#else
#define SHARED_LINK
#endif

/* Shared ring buffer placed in D2 shared RAM */
SHARED_LINK volatile shared_ring_t shared_ring;

bool shared_push_frame(const sensor_frame_t *f)
{
    uint32_t next = (shared_ring.head + 1) % SHARED_FRAMES_COUNT;
    if (next == shared_ring.tail) return false; /* full */
    shared_ring.frames[shared_ring.head] = *f;
    __DSB(); /* ensure memory writes complete */
    shared_ring.head = next;
    return true;
}

bool shared_pop_frame(sensor_frame_t *out)
{
    if (shared_ring.head == shared_ring.tail) return false; /* empty */
    *out = shared_ring.frames[shared_ring.tail];
    __DSB();
    shared_ring.tail = (shared_ring.tail + 1) % SHARED_FRAMES_COUNT;
    return true;
}
