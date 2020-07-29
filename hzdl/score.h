#pragma once

#include "hzdl/dnn.h"

float Accuracy(dnn* net, float* labels);
float Loss(dnn* net, float* labels);

