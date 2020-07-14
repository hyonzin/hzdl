#pragma once

#include "../dnn.h"

void Dense(dnn* net, int dim, float (*activation)(float));
void DenseForward(layer* p);
void DenseBackward(layer* p);

