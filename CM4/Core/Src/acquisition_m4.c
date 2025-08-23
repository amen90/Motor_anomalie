#include "main.h"
#include "cmsis_os.h"
#include "msa301.h"
#include "stm32h7xx_hal.h"
#include "stm32h7xx_hal_hsem.h"
#include "shared_mem.h"
#include "stm32h745xx.h"
#include "ai_data_collection.h"

/* HSEM ID definition */
#ifndef HSEM_ID_0
#define HSEM_ID_0 (0U) /* HW semaphore 0*/
#endif


extern I2C_HandleTypeDef hi2c1;
/* Local variables to store sensor data */
static int16_t sensor_x, sensor_y, sensor_z;

/* Acquisition task prototype (create this task in CubeMX-generated RTOS init or add here) */
void AcquisitionTask(void *argument)
{
    sensor_frame_t frame;
    /* Initialize sensor */
    if (!msa301_probe(&hi2c1)) {
        /* Sensor not present: blink an LED or log via SWO/USB if available. */
    } else {
        msa301_configure(&hi2c1);
    }

    /* Start initial sensor read */
    if (!msa301_read_raw(&hi2c1, &sensor_x, &sensor_y, &sensor_z)) {
        /* handle error */
    }

    for (;;) {
        /* Wait for periodic sensor reading (e.g., every 100ms) */
        vTaskDelay(pdMS_TO_TICKS(100));

        /* Read sensor data */
        if (msa301_read_raw(&hi2c1, &sensor_x, &sensor_y, &sensor_z)) {
            frame.x = sensor_x;
            frame.y = sensor_y;
            frame.z = sensor_z;
        } else {
            /* Use previous values on error */
        }
        frame.ts = HAL_GetTick();

        /* Protect shared buffer update:
           - Option A: Use HSEM to protect both cores (recommended if you use HSEM)
           - Option B: set shared area non-cacheable / then use simple push
           Here we disable IRQs briefly to make update atomic on this core
           (still need cache maintenance if region is cacheable). */
        taskENTER_CRITICAL();
        shared_push_frame(&frame);
        taskEXIT_CRITICAL();

        /* Notify CM7: release HSEM (example). On CM7 side you must enable HSEM notification */
        HAL_HSEM_FastTake(HSEM_ID_0);
        HAL_HSEM_Release(HSEM_ID_0, 0);

        /* Sensor reading will happen in next loop iteration */
    }
}

/* I2C complete callback - not used in this implementation but kept for future use */
void HAL_I2C_MemRxCpltCallback(I2C_HandleTypeDef *hi2c)
{
    /* This callback can be used if implementing DMA-based I2C reading */
    /* For now, we use blocking reads in the task loop */
}

/* I2C error callback */
void HAL_I2C_ErrorCallback(I2C_HandleTypeDef *hi2c)
{
    /* Log error or handle I2C error */
    /* Error handling can be implemented here if needed */
}
