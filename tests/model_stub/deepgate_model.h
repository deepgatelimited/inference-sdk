#pragma once
#include <deepgate.h>

// Test model metadata. Only _unknown_ is a background label.
#define DG_MODEL_N_LABELS 3
#define DG_MODEL_INPUT_SCALE 1.33f
#define DG_MODEL_INPUT_ZERO_POINT 17
#define DG_MODEL_OUTPUT_SCALE 0.1f
#define DG_MODEL_OUTPUT_ZERO_POINT 0
static const char *const DG_MODEL_LABELS[] = {"_unknown_", "lights", "unknown"};
static const bool DG_MODEL_IS_BACKGROUND[] = {true, false, false};
static const float DG_MODEL_THRESHOLDS[] = {0.0f, 0.4f, 0.6f};
