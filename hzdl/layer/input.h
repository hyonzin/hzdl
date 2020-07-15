#pragma once

#include "../dnn.h"
#include "../util.h"

void Input(dnn* net, int n, int c, int h, int w);
void InputDestroy(layer* p);

