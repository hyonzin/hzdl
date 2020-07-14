#pragma once

#include "../dnn.h"

void Softmax(dnn* net);
void SoftmaxForward(layer* p);
void SoftmaxBackward(layer* p);

