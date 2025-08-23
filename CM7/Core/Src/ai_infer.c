#include "ai_infer.h"
#include "motor_anomalie.h"
#include "motor_anomalie_data.h"
#include "cmsis_os.h"
#include "shared_mem.h"
#include "stm32h7xx_hal.h"
#include "stm32h7xx_hal_gpio.h"
#include <string.h>

static ai_handle s_network = AI_HANDLE_NULL;
static AI_ALIGNED(4) uint8_t s_activations[10000];

void AI_Init(void)
{
    const ai_handle acts[] = { s_activations };
    const ai_handle wts[]  = { AI_HANDLE_NULL };
    ai_error err = ai_motor_anomalie_create_and_init(&s_network, acts, wts);
    (void)err;
}

void AI_DeInit(void)
{
    if (s_network) {
        ai_motor_anomalie_destroy(s_network);
        s_network = AI_HANDLE_NULL;
    }
}

bool AI_RunOnce(const int8_t *input_s8_60x3, int8_t *out_s8_4)
{
    if (!s_network || !input_s8_60x3 || !out_s8_4) return false;

    ai_i8 in_buffer_mem[180];
    memcpy(in_buffer_mem, input_s8_60x3, 180);

    ai_buffer in = AI_BUFFER_INIT(
        AI_FLAG_NONE, AI_MOTOR_ANOMALIE_IN_1_FORMAT,
        AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, AI_MOTOR_ANOMALIE_IN_1_CHANNEL, 1, AI_MOTOR_ANOMALIE_IN_1_HEIGHT),
        AI_MOTOR_ANOMALIE_IN_1_SIZE, NULL, in_buffer_mem);

    ai_i8 out_buffer_mem[4];
    ai_buffer out = AI_BUFFER_INIT(
        AI_FLAG_NONE, AI_MOTOR_ANOMALIE_OUT_1_FORMAT,
        AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, AI_MOTOR_ANOMALIE_OUT_1_CHANNEL, 1, 1),
        AI_MOTOR_ANOMALIE_OUT_1_SIZE, NULL, out_buffer_mem);

    ai_i32 nbatch = ai_motor_anomalie_run(s_network, &in, &out);
    if (nbatch != 1) {
        return false;
    }
    memcpy(out_s8_4, out_buffer_mem, 4);
    return true;
}

void AiTask(void *argument)
{
    AI_Init();
    int8_t input_s8[180];
    int8_t out_s8[4];
    /* Simple GPIO feedback mapping: assumes LEDs on GPIOB PIN0/PIN1 */
    __HAL_RCC_GPIOB_CLK_ENABLE();
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
    const float scale = 0.025338666513562202f; /* input quant scale */
    /* Embedded normalization means/stds from training (update with your stats) */
    const float mean_x = 0.0f, mean_y = 0.0f, mean_z = 1000.0f;
    const float std_x  = 50.0f, std_y  = 50.0f, std_z  = 80.0f;
    const int zp = 12;                          /* input zero point */

    for (;;) {
        /* Build a 60x3 window from shared ring (most recent 60 frames) */
        sensor_frame_t buf[60];
        int count = 0;
        /* Drain up to 60 frames; if fewer, pad with zeros around zp */
        while (count < 60) {
            sensor_frame_t f;
            if (!shared_pop_frame_cm7(&f)) break;
            buf[count++] = f;
        }
        if (count > 0) {
            /* Normalize (simple mg to standardization can be added later); here assume data already roughly centered */
            /* Pack as int8 using quantization: q = round(x/scale) + zp */
            for (int i = 0; i < 60; ++i) {
                float xf = (i < count) ? (float)buf[i].x : 0.f;
                float yf = (i < count) ? (float)buf[i].y : 0.f;
                float zf = (i < count) ? (float)buf[i].z : 0.f;
                /* z-score normalize */
                xf = (xf - mean_x) / (std_x + 1e-6f);
                yf = (yf - mean_y) / (std_y + 1e-6f);
                zf = (zf - mean_z) / (std_z + 1e-6f);
                /* quantize */
                int32_t qxi = (int32_t)((xf / scale) + zp);
                int32_t qyi = (int32_t)((yf / scale) + zp);
                int32_t qzi = (int32_t)((zf / scale) + zp);
                if (qxi > 127) qxi = 127; if (qxi < -128) qxi = -128;
                if (qyi > 127) qyi = 127; if (qyi < -128) qyi = -128;
                if (qzi > 127) qzi = 127; if (qzi < -128) qzi = -128;
                int8_t qx = (int8_t)qxi;
                int8_t qy = (int8_t)qyi;
                int8_t qz = (int8_t)qzi;
                input_s8[i*3 + 0] = qx;
                input_s8[i*3 + 1] = qy;
                input_s8[i*3 + 2] = qz;
            }
            if (AI_RunOnce(input_s8, out_s8)) {
                /* Dequantize logits to pick class (softmax int8: scale 1/256, zp=-128) */
                int best = 0; int8_t bestv = out_s8[0];
                for (int k = 1; k < 4; ++k) { if (out_s8[k] > bestv) { bestv = out_s8[k]; best = k; } }
                /* Map classes: 0=normal, >0=fault: LED0 normal ON, LED1 fault ON */
                if (best == 0) {
                    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);
                    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, GPIO_PIN_RESET);
                } else {
                    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
                    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, GPIO_PIN_SET);
                }
            }
        }
        osDelay(20);
    }
}


