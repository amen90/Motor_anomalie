#include "msa301.h"
#include "stm32h7xx_hal.h"
#include "stm32h7xx_hal_hsem.h"
#include "stm32h745xx.h"


/* Simple blocking read/write driver for MSA301 */

/* probe by reading PART ID */
bool msa301_probe(I2C_HandleTypeDef *hi2c)
{
    uint8_t id = 0;
    if (HAL_I2C_Mem_Read(hi2c, MSA301_ADDR, MSA301_REG_PARTID,
                         I2C_MEMADD_SIZE_8BIT, &id, 1, 200) != HAL_OK) {
        return false;
    }
    return (id != 0);
}

/* configure example: range +/-2g, ODR 125Hz, normal power mode
   NOTE: replace ODR/range values with datasheet mapping if needed */
bool msa301_configure(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef st;
    uint8_t v;

    /* Range: ±2g (example) */
    v = 0x00;
    st = HAL_I2C_Mem_Write(hi2c, MSA301_ADDR, MSA301_REG_RESRANGE,
                           I2C_MEMADD_SIZE_8BIT, &v, 1, 200);
    if (st != HAL_OK) return false;

    /* ODR: example code for 125 Hz (check datasheet mapping) */
    v = 0x07;
    st = HAL_I2C_Mem_Write(hi2c, MSA301_ADDR, MSA301_REG_ODR,
                           I2C_MEMADD_SIZE_8BIT, &v, 1, 200);
    if (st != HAL_OK) return false;

    /* Power mode: normal */
    v = 0x00;
    st = HAL_I2C_Mem_Write(hi2c, MSA301_ADDR, MSA301_REG_POWERMODE,
                           I2C_MEMADD_SIZE_8BIT, &v, 1, 200);
    if (st != HAL_OK) return false;

    HAL_Delay(5);
    return true;
}

/* blocking read 6 bytes (X L/H, Y L/H, Z L/H) */
bool msa301_read_raw(I2C_HandleTypeDef *hi2c, int16_t *x, int16_t *y, int16_t *z)
{
    uint8_t buf[6];
    if (HAL_I2C_Mem_Read(hi2c, MSA301_ADDR, MSA301_REG_OUT_X_L,
                         I2C_MEMADD_SIZE_8BIT, buf, sizeof(buf), 200) != HAL_OK) {
        return false;
    }

    *x = (int16_t)((buf[1] << 8) | buf[0]);
    *y = (int16_t)((buf[3] << 8) | buf[2]);
    *z = (int16_t)((buf[5] << 8) | buf[4]);
    return true;
}
