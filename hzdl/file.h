#pragma once

#include "hzdl/dnn.h"
#include "hzdl/layer/layers.h"
#include <string.h>

// void SaveStructure(dnn* net, char* filename);
void SaveWeight(dnn* net, char* filename);
void SaveDNN(dnn* net, char* filename);

// void LoadStructure(dnn** net, char* filename);
void LoadWeight(dnn** net, char* filename);
void LoadDNN(dnn** net, char* filename);

