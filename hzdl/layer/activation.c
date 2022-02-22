#include "hzdl/layer/activation.h"

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
    return val * (1 - val);
}


float ReLUForward(struct _layer* l, int batch_idx, float val) {
    return fmax(val, 0);
}

float ReLUBackward(struct _layer* l, int batch_idx, float val) {
    return (val>0)? 1:0;
}


float SoftmaxForward(struct _layer* l, int batch_idx, float val) {
    int i, dim = _get_num_element(l);
    float *out, sum;

    out = &l->out[batch_idx * dim];

    // Sum of exponential
    if(l->delta[batch_idx] == -1) {
        sum = 0;
        for (i=0; i<dim; ++i) {
            sum += exp(out[i]);
        }
        l->delta[batch_idx] = sum;
    } else {
        sum = l->delta[batch_idx];
    }

    if (sum == 0) return 0;

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

    if (val == 0 || sum == 0) return 0;

    // Calculate ratio
    return (exp(val) * (sum - exp(val))) / (sum * sum);
}


float TanhForward(struct _layer* l, int batch_idx, float val) {
    return 2 * (1 / (1 + exp(-2 * val))) - 1;  // 2 * sigmoid(2x) - 1
}

float TanhBackward(struct _layer* l, int batch_idx, float val) {
    return (1 - (2 * (1 / (1 + exp(-2 * val))) - 1)) *
           (1 + (2 * (1 / (1 + exp(-2 * val))) - 1));  // (1 - tanh(x)) * (1 + tanh(x))
}
