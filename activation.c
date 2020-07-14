#include "activation.h"

float None(float val) {
    return val;
}

float Sigmoid(float val) {
    return 1 / (1 + exp(-val));
}

float ReLU(float val) {
    return fmax(val, 0);
}

