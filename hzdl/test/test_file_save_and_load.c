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

    // DNN to save
    dnn* net;
    CreateDNN(&net);
    Input(net, batch_size, 1, 1, 4);
    Dense(net, 2, Softmax);
   
    Train(net, train_images, train_labels, train_size,
            NULL, NULL, 0,
            epochs, batch_size, learning_rate,
            Accuracy);

    Test(net, test_images, test_labels, test_size,
            batch_size, Accuracy);

    SaveDNN(net, "test_net");

    // DNN to load
    dnn* loaded_net;
    CreateDNN(&loaded_net);
    Input(loaded_net, batch_size, 1, 1, 4);
    Dense(loaded_net, 2, Softmax);

    LoadDNN(&loaded_net, "test_net");

    Test(loaded_net, test_images, test_labels, test_size,
            batch_size, Accuracy);

    DestroyDNN(&net);
    DestroyDNN(&loaded_net);

    return 0;
}

int main(int argc, char* argv[]) {
    return test_simple(argc, argv);
}

