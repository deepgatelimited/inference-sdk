#ifndef TEST_MODEL_H
#define TEST_MODEL_H

#include <stdint.h>

// The fake model uses the same storage for input and output.
#define MODEL_INPUT0_ELEMENTS 490
#define MODEL_OUTPUT0_ELEMENTS 3
#define MODEL_INPUT0_BYTES MODEL_INPUT0_ELEMENTS
#define MODEL_OUTPUT0_BYTES MODEL_OUTPUT0_ELEMENTS

int8_t *model_get_input0(void);
const int8_t *model_get_output0(void);
int run_model(void);

#endif
