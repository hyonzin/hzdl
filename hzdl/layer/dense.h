#pragma once

#include "../dnn.h"

void Dense(dnn* net, int dim, float (*activation)(float));

void ForwardDense(layer* p);
