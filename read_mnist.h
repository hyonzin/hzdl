#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float* read_mnist_train_images();
float* read_mnist_train_labels();
float* read_mnist_test_images();
float* read_mnist_test_labels();
void test_mnist(float* label, float* image, int idx);

