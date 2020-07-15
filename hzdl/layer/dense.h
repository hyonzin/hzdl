#pragma once

#include "../dnn.h"
#include "../util.h"

void Dense(dnn* net, int dim, activation act);
void DenseForward(layer* p);
void DenseBackward(layer* p, float* labels);
void DenseUpdateWeight(layer* p, float eta);
void DenseDestroy(layer* p);

