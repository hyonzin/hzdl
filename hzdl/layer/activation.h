#pragma once

#include <math.h>
#include "hzdl/util.h"

struct _layer;

typedef struct _activation {
    float (*forward)(struct _layer*, int, float);
    float (*backward)(struct _layer*, int, float);
} activation;

float NoneForward(struct _layer* l, int batch_idx, float val);
float NoneBackward(struct _layer* l, int batch_idx, float val);
static activation None = { NoneForward, NoneBackward};

float SigmoidForward(struct _layer* l, int batch_idx, float val);
float SigmoidBackward(struct _layer* l, int batch_idx, float val);
static activation Sigmoid = { SigmoidForward, SigmoidBackward};

float ReLUForward(struct _layer* l, int batch_idx, float val);
float ReLUBackward(struct _layer* l, int batch_idx, float val);
static activation ReLU = { ReLUForward, ReLUBackward};

float SoftmaxForward(struct _layer* l, int batch_idx, float val);
float SoftmaxBackward(struct _layer* l, int batch_idx, float val);
static activation Softmax = { SoftmaxForward, SoftmaxBackward};

float TanhForward(struct _layer* l, int batch_idx, float val);
float TanhBackward(struct _layer* l, int batch_idx, float val);
static activation Tanh = { TanhForward, TanhBackward};

#include "hzdl/layer/layers.h"
