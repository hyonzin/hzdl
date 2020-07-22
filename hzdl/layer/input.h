#pragma once

#include "hzdl/dnn.h"
#include "hzdl/util.h"

void Input(dnn* net, int n, int c, int h, int w);
void InputDestroy(layer* p);

