#ifndef __MSA301_H
#define __MSA301_H

#include "main.h"
#include <stdint.h>
#include <stdbool.h>

/* 7-bit I2C address */
#define MSA301_ADDR_7BIT    0x26
/* HAL wants 8-bit address (read/write bit included) */
#define MSA301_ADDR         (MSA301_ADDR_7BIT << 1)

/* MSA301 register definitions (basic set used here) */
#define MSA301_REG_PARTID    0x01
#define MSA301_REG_OUT_X_L   0x02
#define MSA301_REG_OUT_X_H   0x03
#define MSA301_REG_OUT_Y_L   0x04
#define MSA301_REG_OUT_Y_H   0x05
#define MSA301_REG_OUT_Z_L   0x06
#define MSA301_REG_OUT_Z_H   0x07
#define MSA301_REG_RESRANGE  0x0F
#define MSA301_REG_ODR       0x10
#define MSA301_REG_POWERMODE 0x11

/* Driver API */
bool msa301_probe(I2C_HandleTypeDef *hi2c);
bool msa301_configure(I2C_HandleTypeDef *hi2c);
bool msa301_read_raw(I2C_HandleTypeDef *hi2c, int16_t *x, int16_t *y, int16_t *z);

#endif /* __MSA301_H */
