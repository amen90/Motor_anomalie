#ifndef __AI_INFER_H
#define __AI_INFER_H

#include <stdint.h>
#include <stdbool.h>

void AI_Init(void);
void AI_DeInit(void);
bool AI_RunOnce(const int8_t *input_s8_60x3, int8_t *out_s8_4);
void AiTask(void *argument);

#endif /* __AI_INFER_H */


