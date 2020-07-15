#pragma once

#include <math.h>
#include "layer/layers.h"

float NoneForward(float val);
float NoneBackward(float val);
static activation None = { NoneForward, NoneBackward};

float SigmoidForward(float val);
float SigmoidBackward(float val);
static activation Sigmoid = { SigmoidForward, SigmoidBackward};

float ReLUForward(float val);
float ReLUBackward(float val);
static activation ReLU = { ReLUForward, ReLUBackward};

