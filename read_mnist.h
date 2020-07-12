#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned char* read_train_images();
unsigned char* read_train_labels();
unsigned char* read_test_images();
unsigned char* read_test_labels();
void test_mnist(unsigned char* label, unsigned char* image, int idx);

