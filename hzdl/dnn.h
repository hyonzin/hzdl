#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <omp.h>


typedef struct _dnn {
    struct _layer* next;
    struct _layer* edge;
} dnn;


#include "util.h"
#include "layer/layers.h"

void CreateDNN(dnn** net);
void DestroyDNN(dnn** net);

void Train(dnn* net,
        float* train_images, float* train_labels, int train_size,
        float* test_images, float* test_labels, int test_size,
        int epochs, float learning_rate);

void Test(dnn* net,
        float* test_images, float* test_labels, int test_size);

void Forward(dnn* net);
void Backward(dnn* net, float* labels);
void UpdateWeight(dnn* net, float learning_rate);

