#ifndef __AI_DATA_COLLECTION_H
#define __AI_DATA_COLLECTION_H

#include "main.h"
#include <stdint.h>
#include <stdbool.h>

/* AI Data Collection Configuration */
#define AI_SAMPLE_RATE_HZ           1000    /* 1kHz sampling for AI training */
#define AI_SAMPLE_DURATION_SEC      10      /* 10 seconds per sample */
#define AI_SAMPLES_PER_COLLECTION   (AI_SAMPLE_RATE_HZ * AI_SAMPLE_DURATION_SEC)
#define AI_BUFFER_SIZE              (AI_SAMPLES_PER_COLLECTION * 3)  /* 3 axes */

/* Motor fault types for labeling */
typedef enum {
    MOTOR_NORMAL = 0,
    MOTOR_IMBALANCE = 1,
    MOTOR_BEARING_FAULT = 2,
    MOTOR_MISALIGNMENT = 3,
    MOTOR_UNKNOWN = 255
} motor_fault_type_t;

/* AI training sample structure */
typedef struct {
    uint32_t sample_id;
    uint32_t timestamp;
    motor_fault_type_t fault_type;
    uint16_t sample_rate;
    uint16_t duration_ms;
    uint32_t num_samples;
    int16_t data[AI_BUFFER_SIZE];  /* Interleaved X,Y,Z data */
} ai_training_sample_t;

/* Collection status */
typedef enum {
    AI_COLLECTION_IDLE = 0,
    AI_COLLECTION_ACTIVE = 1,
    AI_COLLECTION_COMPLETE = 2,
    AI_COLLECTION_ERROR = 3
} ai_collection_status_t;

/* Function prototypes */
void ai_data_collection_init(void);
bool ai_start_collection(motor_fault_type_t fault_type);
bool ai_stop_collection(void);
ai_collection_status_t ai_get_collection_status(void);
bool ai_get_sample_data(ai_training_sample_t *sample);
void ai_reset_collection(void);

/* USB communication functions */
void ai_send_sample_via_usb(const ai_training_sample_t *sample);
void ai_process_usb_commands(void);

/* High-speed acquisition task */
void AIDataCollectionTask(void *argument);

#endif /* __AI_DATA_COLLECTION_H */
