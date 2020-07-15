#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>


typedef struct _dnn {
    struct _layer* next;
    struct _layer* edge;
} dnn;


#include "util.h"
#include "activation.h"
#include "layer/layers.h"

void CreateDNN(dnn** net);
void DestroyDNN(dnn** net);

void Train(dnn* net, float* train_images, float* train_labels,
        int train_size, int batch_size, int epochs, float learning_rate);

void Forward(dnn* net, int batch_size);
void Backward(dnn* net, int batch_size, float learning_rate, float* labels);

