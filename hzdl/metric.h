#pragma once

#include "hzdl/dnn.h"


void GetMetricName(char* name, float (*metric_function)(struct _dnn*, float*));

float Accuracy(dnn* net, float* labels);
float Loss(dnn* net, float* labels);

