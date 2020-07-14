#include "activation.h"

float None(float val) {
    return val;
}

float Sigmoid(float val) {
    return 1 / (1 + exp(-val));
}

