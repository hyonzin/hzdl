#include <stdio.h>
#include "hzdl/dnn.h"


int test_simple(int argc, char* argv[]) {
    float train_images[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float train_labels[8] = {1, 0, 1, 0, 0, 1, 0, 1};
    float test_images[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float test_labels[8] = {1, 0, 1, 0, 0, 1, 0, 1};

    int train_size = 4, test_size = 4;
    int batch_size = 1, epochs = 10;
    float learning_rate = 0.300;

    dnn* net;
    CreateDNN(&net);

    Input(net, batch_size, 1, 1, 4);
    Dense(net, 2, Softmax);
   
    Train(net, train_images, train_labels, train_size,
            test_images, test_labels, test_size,
            epochs, learning_rate,
            Accuracy);

    DestroyDNN(&net);
    return 0;
}

int main(int argc, char* argv[]) {
    return test_simple(argc, argv);
}

