#include "activation.h"


float NoneForward(struct _layer* l, int batch_idx, float val) {
    return val;
}

float NoneBackward(struct _layer* l, int batch_idx, float val) {
    return 1;
}


float SigmoidForward(struct _layer* l, int batch_idx, float val) {
    return 1 / (1 + exp(-val));
}

float SigmoidBackward(struct _layer* l, int batch_idx, float val) {
    float res = 1 / (1 + exp(-val));
    res = res * (1 - res);
    return res;
}


float ReLUForward(struct _layer* l, int batch_idx, float val) {
    return fmax(val, 0);
}

float ReLUBackward(struct _layer* l, int batch_idx, float val) {
    return (val>0)? 1:0;
}


float SoftmaxForward(struct _layer* l, int batch_idx, float val) {
    int i, dim = _get_num_element(l);
    float *out, sum = 0;

    out = &l->out[batch_idx * dim];

    // Sum of exponential
    for (i=0; i<dim; ++i) {
        sum += exp(out[i]);
    }
    
    // Calculate ratio
    return exp(val) / sum;
}

float SoftmaxBackward(struct _layer* l, int batch_idx, float val) {
    int i, dim = _get_num_element(l);
    float *out, sum = 0;

    out = &l->out[batch_idx * dim];

    // Sum of exponential
    for (i=0; i<dim; ++i) {
        sum += exp(out[i]);
    }
    
    // Calculate ratio
    return (exp(val) * (sum - exp(val))) / (sum * sum);
}

