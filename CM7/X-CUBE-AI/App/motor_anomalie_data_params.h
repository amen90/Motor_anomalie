/**
  ******************************************************************************
  * @file    motor_anomalie_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-08-23T15:13:00+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef MOTOR_ANOMALIE_DATA_PARAMS_H
#define MOTOR_ANOMALIE_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_MOTOR_ANOMALIE_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_motor_anomalie_data_weights_params[1]))
*/

#define AI_MOTOR_ANOMALIE_DATA_CONFIG               (NULL)


#define AI_MOTOR_ANOMALIE_DATA_ACTIVATIONS_SIZES \
  { 9984, }
#define AI_MOTOR_ANOMALIE_DATA_ACTIVATIONS_SIZE     (9984)
#define AI_MOTOR_ANOMALIE_DATA_ACTIVATIONS_COUNT    (1)
#define AI_MOTOR_ANOMALIE_DATA_ACTIVATION_1_SIZE    (9984)



#define AI_MOTOR_ANOMALIE_DATA_WEIGHTS_SIZES \
  { 40944, }
#define AI_MOTOR_ANOMALIE_DATA_WEIGHTS_SIZE         (40944)
#define AI_MOTOR_ANOMALIE_DATA_WEIGHTS_COUNT        (1)
#define AI_MOTOR_ANOMALIE_DATA_WEIGHT_1_SIZE        (40944)



#define AI_MOTOR_ANOMALIE_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_motor_anomalie_activations_table[1])

extern ai_handle g_motor_anomalie_activations_table[1 + 2];



#define AI_MOTOR_ANOMALIE_DATA_WEIGHTS_TABLE_GET() \
  (&g_motor_anomalie_weights_table[1])

extern ai_handle g_motor_anomalie_weights_table[1 + 2];


#endif    /* MOTOR_ANOMALIE_DATA_PARAMS_H */
