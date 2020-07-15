#include "activation.h"

float NoneForward(float val) {
    return val;
}

float NoneBackward(float val) {
    return 1;
}

float SigmoidForward(float val) {
    return 1 / (1 + exp(-val));
}

float SigmoidBackward(float val) {
    return SigmoidForward(val) * (1 - SigmoidForward(val));
}

float ReLUForward(float val) {
    return fmax(val, 0);
}

float ReLUBackward(float val) {
    return (val>0)? 1:0;
}

