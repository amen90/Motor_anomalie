#include "ai_data_collection.h"
#include "msa301.h"
#include "cmsis_os.h"
#include "stm32h7xx_hal.h"
#include "stm32h7xx_hal_tim.h"
#include "stm32h7xx_it.h"
#include <string.h>
#include <stdio.h>

/* External I2C handle */
extern I2C_HandleTypeDef hi2c1;

/* Static variables for data collection */
static ai_training_sample_t current_sample;
static volatile ai_collection_status_t collection_status = AI_COLLECTION_IDLE;
static volatile uint32_t sample_counter = 0;
static volatile uint32_t sample_id_counter = 0;
static volatile bool data_ready = false;

/* Timer for precise timing */
static TIM_HandleTypeDef htim_ai;

/* Mutex for thread safety */
static osMutexId_t ai_collection_mutex;

/* Function to initialize AI data collection system */
void ai_data_collection_init(void)
{
    /* Initialize mutex */
    ai_collection_mutex = osMutexNew(NULL);
    
    /* Initialize timer for precise 1kHz sampling */
    /* TIM6 is typically available and suitable for this purpose */
    __HAL_RCC_TIM6_CLK_ENABLE();
    
    htim_ai.Instance = TIM6;
    htim_ai.Init.Prescaler = (SystemCoreClock / 1000000) - 1;  /* 1MHz timer clock */
    htim_ai.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim_ai.Init.Period = 1000 - 1;  /* 1000us = 1ms = 1kHz */
    htim_ai.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim_ai.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    
    if (HAL_TIM_Base_Init(&htim_ai) != HAL_OK) {
        /* Timer initialization error */
        collection_status = AI_COLLECTION_ERROR;
        return;
    }
    
    /* Enable TIM6 clock and NVIC */
    __HAL_RCC_TIM6_CLK_ENABLE();
    HAL_NVIC_SetPriority(TIM6_DAC_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(TIM6_DAC_IRQn);
    
    /* Initialize sample structure */
    memset(&current_sample, 0, sizeof(ai_training_sample_t));
    collection_status = AI_COLLECTION_IDLE;
}

/* Start data collection for specified fault type */
bool ai_start_collection(motor_fault_type_t fault_type)
{
    if (osMutexAcquire(ai_collection_mutex, 100) != osOK) {
        return false;
    }
    
    if (collection_status != AI_COLLECTION_IDLE) {
        osMutexRelease(ai_collection_mutex);
        return false;
    }
    
    /* Initialize sample data */
    current_sample.sample_id = ++sample_id_counter;
    current_sample.timestamp = HAL_GetTick();
    current_sample.fault_type = fault_type;
    current_sample.sample_rate = AI_SAMPLE_RATE_HZ;
    current_sample.duration_ms = AI_SAMPLE_DURATION_SEC * 1000;
    current_sample.num_samples = AI_SAMPLES_PER_COLLECTION;
    
    /* Reset counters */
    sample_counter = 0;
    data_ready = false;
    
    /* Start timer */
    collection_status = AI_COLLECTION_ACTIVE;
    HAL_TIM_Base_Start_IT(&htim_ai);
    
    osMutexRelease(ai_collection_mutex);
    
    printf("AI: Started collection for fault type %d, Sample ID: %lu\r\n", 
           fault_type, current_sample.sample_id);
    
    return true;
}

/* Stop data collection */
bool ai_stop_collection(void)
{
    if (osMutexAcquire(ai_collection_mutex, 100) != osOK) {
        return false;
    }
    
    HAL_TIM_Base_Stop_IT(&htim_ai);
    
    if (collection_status == AI_COLLECTION_ACTIVE) {
        collection_status = AI_COLLECTION_COMPLETE;
        data_ready = true;
    }
    
    osMutexRelease(ai_collection_mutex);
    
    printf("AI: Collection stopped. Samples collected: %lu\r\n", sample_counter);
    return true;
}

/* Get collection status */
ai_collection_status_t ai_get_collection_status(void)
{
    return collection_status;
}

/* Get sample data (non-blocking) */
bool ai_get_sample_data(ai_training_sample_t *sample)
{
    if (!sample || !data_ready) {
        return false;
    }
    
    if (osMutexAcquire(ai_collection_mutex, 100) != osOK) {
        return false;
    }
    
    memcpy(sample, &current_sample, sizeof(ai_training_sample_t));
    data_ready = false;
    
    osMutexRelease(ai_collection_mutex);
    
    return true;
}

/* Reset collection system */
void ai_reset_collection(void)
{
    if (osMutexAcquire(ai_collection_mutex, 100) != osOK) {
        return;
    }
    
    HAL_TIM_Base_Stop_IT(&htim_ai);
    collection_status = AI_COLLECTION_IDLE;
    sample_counter = 0;
    data_ready = false;
    memset(&current_sample, 0, sizeof(ai_training_sample_t));
    
    osMutexRelease(ai_collection_mutex);
}

/* Timer interrupt handler for precise 1kHz sampling */
void TIM6_DAC_IRQHandler(void)
{
    if (__HAL_TIM_GET_FLAG(&htim_ai, TIM_FLAG_UPDATE) != RESET && __HAL_TIM_GET_IT_SOURCE(&htim_ai, TIM_IT_UPDATE) != RESET) {
        __HAL_TIM_CLEAR_IT(&htim_ai, TIM_IT_UPDATE);
        
        if (collection_status == AI_COLLECTION_ACTIVE && sample_counter < AI_SAMPLES_PER_COLLECTION) {
            int16_t x, y, z;
            
            /* Fast I2C read - this should be optimized for speed */
            if (msa301_read_raw(&hi2c1, &x, &y, &z)) {
                uint32_t index = sample_counter * 3;
                current_sample.data[index] = x;
                current_sample.data[index + 1] = y;
                current_sample.data[index + 2] = z;
                
                sample_counter++;
                
                /* Check if collection is complete */
                if (sample_counter >= AI_SAMPLES_PER_COLLECTION) {
                    HAL_TIM_Base_Stop_IT(&htim_ai);
                    collection_status = AI_COLLECTION_COMPLETE;
                    data_ready = true;
                }
            } else {
                /* I2C read failed - handle error */
                collection_status = AI_COLLECTION_ERROR;
                HAL_TIM_Base_Stop_IT(&htim_ai);
            }
        }
    }
}

/* Send sample data via USB (CDC) */
void ai_send_sample_via_usb(const ai_training_sample_t *sample)
{
    if (!sample) return;
    
    /* Send header information */
    printf("AI_SAMPLE_START\r\n");
    printf("ID:%lu\r\n", sample->sample_id);
    printf("TIMESTAMP:%lu\r\n", sample->timestamp);
    printf("FAULT_TYPE:%d\r\n", sample->fault_type);
    printf("SAMPLE_RATE:%d\r\n", sample->sample_rate);
    printf("DURATION:%d\r\n", sample->duration_ms);
    printf("NUM_SAMPLES:%lu\r\n", sample->num_samples);
    printf("DATA_START\r\n");
    
    /* Send data in chunks to avoid buffer overflow */
    const uint32_t chunk_size = 100;  /* Send 100 samples at a time */
    
    for (uint32_t i = 0; i < sample->num_samples; i++) {
        uint32_t index = i * 3;
        printf("%d,%d,%d\r\n", 
               sample->data[index], 
               sample->data[index + 1], 
               sample->data[index + 2]);
        
        /* Small delay every chunk to prevent USB buffer overflow */
        if ((i + 1) % chunk_size == 0) {
            HAL_Delay(10);
        }
    }
    
    printf("DATA_END\r\n");
    printf("AI_SAMPLE_END\r\n");
}

/* Process USB commands for data collection control */
void ai_process_usb_commands(void)
{
    /* This function should be called from main loop to process incoming commands */
    /* Implementation depends on your USB CDC setup */
    /* Commands format:
     * "START_NORMAL" - Start collection for normal motor
     * "START_IMBALANCE" - Start collection for imbalanced motor
     * "START_BEARING" - Start collection for bearing fault
     * "START_MISALIGN" - Start collection for misalignment
     * "STOP" - Stop current collection
     * "GET_DATA" - Send collected data via USB
     * "RESET" - Reset collection system
     * "STATUS" - Get current status
     */
}

/* High-speed data collection task */
void AIDataCollectionTask(void *argument)
{
    ai_training_sample_t sample_buffer;
    
    /* Initialize AI data collection system */
    ai_data_collection_init();
    
    printf("AI: Data collection task started\r\n");
    
    for (;;) {
        /* Process USB commands */
        ai_process_usb_commands();
        
        /* Check if data is ready to send */
        if (ai_get_collection_status() == AI_COLLECTION_COMPLETE) {
            if (ai_get_sample_data(&sample_buffer)) {
                printf("AI: Sending sample data via USB\r\n");
                ai_send_sample_via_usb(&sample_buffer);
                ai_reset_collection();
            }
        }
        
        /* Small delay to prevent excessive CPU usage */
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
