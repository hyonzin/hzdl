#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

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
    void (*backward)(struct _layer*);
    float (*activation)(float);

    float* in;
    float* out;
    float* weight;
    float* bias;

    struct _layer* prev;
    struct _layer* next;
    struct _dnn* dnn;
} layer;

typedef struct _dnn {
    struct _layer* next;
    struct _layer* edge;
} dnn;


#include "util.h"
#include "activation.h"


void CreateDNN(dnn** net);
void DestroyDNN(dnn** net);
void Input(dnn* net, int n, int c, int h, int w);
void Dense(dnn* net, int dim, float (*activation)(float));
float None(float val);
float Sigmoid(float val);

void Softmax(dnn* net);

void Train(dnn* net, float* train_images, float* train_labels,
        int train_size, int batch_size, int epochs, float learning_rate);


void Forward(dnn* net, int batch_size);
void Backward(dnn* net, int batch_size, float learning_rate, float* labels);

