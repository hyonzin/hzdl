#pragma once

#include "hzdl/layer/activation.h"

enum layer_type {
    layer_type_input,
    layer_type_dense,
    layer_type_softmax
};

typedef struct _layer {
    int n;
    int c;
    int h;
    int w;
    enum layer_type type;
    void (*forward)(struct _layer*);
    void (*backward)(struct _layer*, float* labels);
    void (*update_weight)(struct _layer*, float eta);
    void (*destroy)(struct _layer*);
    struct _activation act;

    float* in;
    float* out;
    float* weight;
    float* bias;
    float* delta;

    struct _layer* prev;
    struct _layer* next;
    struct _dnn* dnn;
} layer;

#include "input.h"
#include "dense.h"

